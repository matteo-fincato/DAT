# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np
from os import path as osp

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter
import tqdm

from config import get_config
from models import build_model, build_criterion
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained, plot_sequence
from data import get_coco_api_from_dataset
from data.coco_eval import CocoEvaluator
from data.panoptic_eval import PanopticEvaluator

from torch.cuda.amp import GradScaler, autocast
from util import misc as utils
from util.vis import build_visualizers, vis_results
import warnings
warnings.filterwarnings('ignore')


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    # easy config modification
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--eval_output', default='eval_output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--pretrained', type=str, help='Finetune 384 initial checkpoint.', default='')
    parser.add_argument('--mot_path_train', default='/mnt/beegfs/work/ToyotaHPE/code/Tracking/trackformer/data/Panoptic')
    parser.add_argument('--mot_path_val', default='/mnt/beegfs/work/ToyotaHPE/code/Tracking/trackformer/data/Panoptic')
    parser.add_argument('--train_split', default='train')
    parser.add_argument('--train_split_json', default='train_100')
    parser.add_argument('--val_split', default='val')
    parser.add_argument('--val_split_json', default='val_100')
    parser.add_argument('--aux_loss', default=False, action='store_true')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main():
    
    args, config = parse_option()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend='gloo', init_method='env://', world_size=world_size, rank=rank) 
    torch.cuda.set_device(local_rank)
    dist.barrier()

    seed = config.SEED + dist.get_rank() 
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device('cuda')
    
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.LOCAL_RANK = local_rank
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    
    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model, postprocessors = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True, find_unused_parameters=True)
    model_without_ddp = model.module
    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    # if config.AUG.MIXUP > 0.:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif config.MODEL.LABEL_SMOOTHING > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    # else:
    #     criterion = nn.CrossEntropyLoss()

    criterion = build_criterion(config)
    criterion.cuda()
    max_accuracy = 0.0
    
    visualizers = build_visualizers(config, list(criterion.weight_dict.keys()))

    if args.pretrained != '':
        load_pretrained(args.pretrained, model_without_ddp, logger)

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        with torch.no_grad():
            val_stats, _ = evaluate(
                    model, criterion, postprocessors, data_loader_val, device,
                    config.EVAL_OUTPUT, None, config, test=True)
        torch.cuda.empty_cache()
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, logger, visualizers['train'], postprocessors)
        torch.cuda.empty_cache()
        # if dist.get_rank() == 0 and ((epoch + 1) % config.SAVE_FREQ == 0 or (epoch + 1) == (config.TRAIN.EPOCHS)):
        #     save_checkpoint(config, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, epoch=epoch + 1)
        if dist.get_rank() == 0:
            save_checkpoint(config, model_without_ddp, max_accuracy, optimizer, lr_scheduler, logger, epoch=epoch+1)

        if epoch > 1 and not epoch % config.TRAIN.VAL_INTERVAL:
            val_stats, _ = evaluate(
                    model, criterion, postprocessors, data_loader_val, device,
                    config.OUTPUT, visualizers['val'], config, epoch)
        torch.cuda.empty_cache()
        # logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        # max_accuracy = max(max_accuracy, acc1)
        # logger.info(f'Max accuracy: {max_accuracy:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, logger, visualizers, postprocessors=None):
    model.train()
    optimizer.zero_grad()

    vis_iter_metrics = None
    if visualizers:
        vis_iter_metrics = visualizers['iter_metrics']

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    device = torch.device('cuda')

    start = time.time()
    end = time.time()

    scaler = GradScaler()
    
    metric_logger = utils.MetricLogger(
        config.VISDOM.vis_and_log_interval,
        delimiter="  ",
        vis=vis_iter_metrics,
        debug=False)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    for idx, (samples, targets) in enumerate(metric_logger.log_every(data_loader, epoch)):
        
        optimizer.zero_grad()
        samples = samples.tensors.cuda(non_blocking=True)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        
        if config.AMP: 
            with autocast():
                outputs, _, _ = model(samples)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs, _, _, _, _, _, _ = model(samples)
            loss_dict = criterion(outputs, targets)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {
                f'{k}_unscaled': v for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            losses.backward()
            if config.TRAIN.CLIP_GRAD:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            optimizer.step()

            metric_logger.update(loss=loss_value,
                                **loss_dict_reduced_scaled,
                                **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"],
                                lr_backbone=optimizer.param_groups[1]["lr"])
            
        lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        # loss_meter.update(losses.item(), targets.size(0))
        # norm_meter.update(grad_norm)
        # batch_time.update(time.time() - end)
        # end = time.time()

        # if (idx + 1) % config.PRINT_FREQ == 0:
        #     lr = optimizer.param_groups[0]['lr']
        #     memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        #     etas = batch_time.avg * (num_steps - idx)
        #     logger.info(
        #         f'Train: [{epoch + 1}/{config.TRAIN.EPOCHS}][{idx + 1}/{num_steps}]\t'
        #         f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
        #         f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        #         f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        #         f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
        #         f'mem {memory_used:.0f}MB')
            
        if visualizers and (idx == 0 or not idx % config.VISDOM.vis_and_log_interval):
                _, results = make_results(
                    outputs, targets, postprocessors, return_only_orig=False)

                vis_results(
                    visualizers['example_results'],
                    samples.detach()[0],
                    results[0],
                    targets[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch + 1} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, device,
             output_dir: str, visualizers: dict, args, epoch: int = None, test = False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(
        args.VISDOM.vis_and_log_interval,
        delimiter="  ",
        debug=False)
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    base_ds = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = tuple(k for k in ('bbox', 'segm') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 'Test:')):
        samples = samples.tensors.cuda(non_blocking=True)
        targets = [utils.nested_dict_to_device(t, device) for t in targets]

        outputs, _, _, _, _, _, _ = model(samples)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if visualizers and (i == 0 or not i % args.VISDOM.vis_and_log_interval):
            results_orig, results = make_results(
                outputs, targets, postprocessors, return_only_orig=False)

            vis_results(
                visualizers['example_results'],
                samples.detach()[0],
                results[0],
                targets[0])
        else:
            results_orig, _ = make_results(outputs, targets, postprocessors)

        if test:
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
            for img, tgt, res in tqdm.tqdm(zip(samples, targets, results_orig), total=len(samples)):
                root = data_loader.dataset.root
                filename = data_loader.dataset.coco.imgs[tgt['image_id'].item()]['file_name']
                img_path = osp.join(root, filename)
                plot_sequence(img, res, tgt, img_path, output_dir, "debug", False)
            
        # TODO. remove cocoDts from coco eval and change example results output
        if coco_evaluator is not None:
            results_orig = {
                target['image_id'].item(): output
                for target, output in zip(targets, results_orig)}

            coco_evaluator.update(results_orig)

        if panoptic_evaluator is not None:
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for j, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[j]["image_id"] = image_id
                res_pano[j]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in coco_evaluator.coco_eval:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in coco_evaluator.coco_eval:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]


    eval_stats = stats['coco_eval_bbox'][:3]
    if 'coco_eval_masks' in stats:
        eval_stats.extend(stats['coco_eval_masks'][:3])
    if 'track_bbox' in stats:
        eval_stats.extend(stats['track_bbox'])

    # VIS
    if visualizers:
        vis_epoch = visualizers['epoch_metrics']
        y_data = [stats[legend_name] for legend_name in vis_epoch.viz_opts['legend']]

        vis_epoch.plot(y_data, epoch)

        visualizers['epoch_eval'].plot(eval_stats, epoch)


    return eval_stats, coco_evaluator


def make_results(outputs, targets, postprocessors, return_only_orig=True):
    target_sizes = torch.stack([t["size"] for t in targets], dim=0)
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
   
    outputs['pred_keypoints'] = outputs['pred_kpts']
    outputs['pred_bboxes'] = outputs['pred_bboxes']
        
    results = None
    if not return_only_orig:
        results = postprocessors['bbox'](outputs, target_sizes)
    results_orig = postprocessors['bbox'](outputs, orig_target_sizes)

    if 'segm' in postprocessors:
        results_orig = postprocessors['segm'](
            results_orig, outputs, orig_target_sizes, target_sizes)
        if not return_only_orig:
            results = postprocessors['segm'](
                results, outputs, target_sizes, target_sizes)

    if results is None:
        return results_orig, results

    for i, result in enumerate(results):
        target = targets[i]
        target_size = target_sizes[i].unsqueeze(dim=0)

        result['target'] = {}
        result['boxes'] = result['boxes'].cpu()
        result['kpts'] = result['kpts'].cpu()
        
        # revert boxes for visualization
        # for key in ['keypoints', 'track_query_kpts']:
        #     if key in target:
        #         target[key] = postprocessors['bbox'].process_kpts(
        #             target[key], target['boxes'], target_size).cpu()
                
        for key in ['boxes', 'track_query_boxes']:
            if key in target:
                target[key] = postprocessors['bbox'].process_boxes(
                    target[key], target_size)[0].cpu()        
                

    return results_orig, results

if __name__ == '__main__':
    main()
