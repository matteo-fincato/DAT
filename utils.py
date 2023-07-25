# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

from os import path as osp
import os
import torch
import torch.distributed as dist
import subprocess
import cv2
import numpy as np
import tqdm
import torchvision.transforms as T
from PIL import Image


def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch']
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy

def load_pretrained(ckpt_path, model, logger):
    logger.info(f"==============> Loading pretrained form {ckpt_path}....................")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    msg = model.load_pretrained(checkpoint['model'])
    logger.info(msg)
    logger.info(f"=> Loaded successfully {ckpt_path} ")
    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, model, max_accuracy, optimizer, lr_scheduler, logger, epoch=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}

    # if epoch:
    #     save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    # else:
    save_path = os.path.join(config.OUTPUT, f'ckpt.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def init_dist_slurm():
    """Initialize slurm distributed training environment.
    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.
    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    # specify master port
    if 'MASTER_PORT' in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    # use MASTER_ADDR in the environment variable if it already exists
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = addr
    
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
    os.environ['RANK'] = str(proc_id)

    dist.init_process_group(backend='nccl')


def plot_sequence(img, result, target, img_path, output_dir, write_images, generate_attention_maps):
    """Plots a whole sequence

    Args:
        tracks (dict): The dictionary containing the track dictionaries in the form tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images belonging to the tracks (e.g. MOT_Sequence object)
        output_dir (String): Directory where to save the resulting images
    """

    # if generate_attention_maps:
    #     attention_maps_per_track = {
    #         track_id: (np.concatenate([t['attention_map'] for t in track.values()])
    #                    if len(track) > 1
    #                    else list(track.values())[0]['attention_map'])
    #         for track_id, track in tracks.items()}
    #     attention_map_thresholds = {
    #         track_id: np.histogram(maps, bins=2)[1][1]
    #         for track_id, maps in attention_maps_per_track.items()}

    # _, attention_maps_bin_edges = np.histogram(all_attention_maps, bins=2)
            
    # img = cv2.imread(img_path)[:, :, (2, 1, 0)]    

    # fig = plt.figure()
    # fig.set_size_inches(width / 96, height / 96)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    # ax.imshow(img)

    img = cv2.imread(img_path)
    height, width, _ = img.shape

    if generate_attention_maps:
        attention_map_img = np.zeros((height, width, 4))

    bboxs = result['boxes']
    kpts = result['kpts']
    scores = result['scores']

    det_keep = torch.logical_and(scores > 0.2, result['labels'] == 0)
    
    bboxs = bboxs[det_keep].cpu().numpy()
    scores = scores[det_keep].cpu().numpy()
    # new_det_indices = det_keep.float().nonzero()
    kpts = kpts[det_keep].cpu().numpy()
    
    for kpt, bbox in zip(kpts, bboxs):    
        new_shape = (kpt.shape[-1]//2, 2)
        kpt = np.reshape(kpt, new_shape)    
        num_keypoint = kpt.shape[0]
        if num_keypoint == 14:
            edges = [[0, 2],[2, 4],[1, 3],[3, 5],[0, 1],[0, 6],[1, 7],[6, 8],[8, 10],[7, 9],[9, 11],[12, 13]
            ]  # neck
            ec = [(169, 209, 142),
                (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                (252, 176, 243), (252, 176, 243), (236, 6, 124)]
        elif num_keypoint == 17:
            edges = [[0, 1],[0, 2],[1, 3],[2, 4],[5, 7],[7, 9],[6, 8],[8, 10],[5, 6],[11, 12],[5, 11],[6, 12],[11, 13],[13, 15],[12, 14],[14, 16]]
            ec = [(236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124),
                (169, 209, 142), (169, 209, 142), (255, 255, 0), (255, 255, 0), 
                (0, 176, 240), (0, 176, 240), (0, 176, 240), (0, 176, 240),
                (252, 176, 243), (252, 176, 243), (255, 102, 0), (255, 102, 0)]
        elif num_keypoint == 22:
            edges = [[1, 2], [2, 3], [3, 4], [3, 8], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11], [3, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [16, 20], [20, 21], [21, 22]]
            edges = (np.array(edges) - 1).tolist()
            ec = [(236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124),
                (169, 209, 142),
                (169, 209, 142), (255, 255, 0), (255, 255, 0), (255, 102, 0),
                (0, 176, 240), (252, 176, 243), (0, 176, 240), (0, 176, 240),
                (252, 176, 243), (252, 176, 243), (252, 176, 243),(252, 176, 243),(252, 176, 243),
                (252, 176, 243), (252, 176, 243) , (252, 176, 243)]
        elif num_keypoint == 19:
            edges = [[0,1],[1,15],[15,16],[1,17],[17,18],[0,2],[0,3],[3,4],[4,5],[2,6],[6,7],[7,8],[0,9],[9,10],[10,11],[2,12],[12,13],[13,14]]
            edges = (np.array(edges)).tolist()
            ec = [(236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124), (236, 6, 124),
                (169, 209, 142),
                (255, 255, 0), (255, 255, 0), (255, 255, 0),
                (255, 102, 0), (255, 102, 0), (255, 102, 0),
                (0, 176, 240), (0, 176, 240), (0, 176, 240),
                (252, 176, 243), (252, 176, 243), (252, 176, 243),]
        else:
            raise ValueError(f'unsupported keypoint amount {num_keypoint}')
        ec = [color[::-1] for color in ec]
        bones_colors=[ec]
        
        keypoints_list = []
        kpt = kpt[:, :2].transpose(1,0).reshape(-1)
        pose2d = [int(x) for x in kpt]
        keypoints = list(zip(pose2d[:num_keypoint], pose2d[num_keypoint:2*num_keypoint])) # keypoints in format  (x_i, y_i)
        keypoints_list.append(keypoints)
        edges = [edges]
        for bone_id, bone in enumerate(edges):
            for idx,(i, j) in enumerate(bone):
                keypoint_a, keypoint_b = keypoints[i], keypoints[j]

                line_colors = bones_colors[bone_id % len(bones_colors)]
                if isinstance(line_colors, list):
                    line_color = line_colors[idx % len(line_colors)]
                else:
                    line_color = line_colors
                
                cv2.circle(img, keypoint_a, 5, (0, 200, 20), thickness=-1)
                cv2.line(img, keypoint_a, keypoint_b, line_color, 5)
                # ax.imshow(img)             

        bbox = [int(el) for el in bbox]
        x0 = (bbox[0], bbox[1])
        x3 = (bbox[2], bbox[3])
        cv2.rectangle(img, x0, x3, color=(255,255,255), thickness=4)

    # if write_images == 'debug':
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     cv2.putText(img, f"{track_data[frame_id]['score']:.2f}",
    #         (bbox[0], bbox[1]), color=annotate_color, fontFace=font, fontScale=2)

    if 'attention_map' in result:
        attention_map = result['attention_map']
        attention_map = cv2.resize(attention_map, (width, height))

        # attention_map_img = np.ones((height, width, 4)) * cmap(track_id)
        # # max value will be at 0.75 transparency
        # attention_map_img[:, :, 3] = attention_map * 0.75 / attention_map.max()

        # _, bin_edges = np.histogram(attention_map, bins=2)
        # attention_map_img[:, :][attention_map < bin_edges[1]] = 0.0

        # attention_map_img += attention_map_img

        # _, bin_edges = np.histogram(attention_map, bins=2)

        norm_attention_map = attention_map / attention_map.max()

        high_att_mask = norm_attention_map > 0.25 # bin_edges[1]
        attention_map_img[:, :][high_att_mask] = cmap(track_id)
        attention_map_img[:, :, 3][high_att_mask] = norm_attention_map[high_att_mask] * 0.5

        # attention_map_img[:, :] += (np.tile(attention_map[..., np.newaxis], (1,1,4)) / attention_map.max()) * cmap(track_id)
        # attention_map_img[:, :, 3] = 0.75

    if generate_attention_maps:
        ax.imshow(attention_map_img, vmin=0.0, vmax=1.0)

    # plt.axis('off')
    # # plt.tight_layout()
    # plt.draw()
    # plt.savefig(osp.join(output_dir, osp.basename(img_path)))
    # plt.close()
    cv2.imwrite(osp.join(output_dir, osp.basename(img_path)), img)