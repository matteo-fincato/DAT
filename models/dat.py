# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple
from util import box_ops
from .dat_blocks import *
from util.misc import inverse_sigmoid

class TransformerStage(nn.Module):

    def __init__(self, fmap_size, window_size, ns_per_pt,
                 dim_in, dim_embed, depths, stage_spec, n_groups, 
                 use_pe, sr_ratio, 
                 heads, stride, offset_range_factor, stage_idx,
                 dwc_pe, no_off, fixed_pe,
                 attn_drop, proj_drop, expansion, drop, drop_path_rate, use_dwc_mlp,
                 num_keypoints, num_classes):

        super().__init__()
        fmap_size = to_2tuple(fmap_size)
        self.depths = depths
        hc = dim_embed // heads
        assert dim_embed == heads * hc
        self.proj = nn.Conv2d(dim_in, dim_embed, 1, 1, 0) if dim_in != dim_embed else nn.Identity()

        self.layer_norms = nn.ModuleList(
            [LayerNormProxy(dim_embed) for _ in range(2 * depths)]
        )
        self.mlps = nn.ModuleList(
            [
                TransformerMLPWithConv(dim_embed, expansion, drop) 
                if use_dwc_mlp else TransformerMLP(dim_embed, expansion, drop)
                for _ in range(depths)
            ]
        )
        self.attns = nn.ModuleList()
        self.drop_path = nn.ModuleList()
        for i in range(depths):
            if stage_spec[i] == 'L':
                self.attns.append(
                    LocalAttention(dim_embed, heads, window_size, attn_drop, proj_drop)
                )
            elif stage_spec[i] == 'D':
                self.attns.append(
                    DAttentionBaseline(fmap_size, fmap_size, heads, 
                    hc, n_groups, attn_drop, proj_drop, 
                    stride, offset_range_factor, use_pe, dwc_pe, 
                    no_off, fixed_pe, stage_idx, num_keypoints, num_classes)
                )
            elif stage_spec[i] == 'S':
                shift_size = math.ceil(window_size / 2)
                self.attns.append(
                    ShiftWindowAttention(dim_embed, heads, window_size, attn_drop, proj_drop, shift_size, fmap_size)
                )
            else:
                raise NotImplementedError(f'Spec: {stage_spec[i]} is not supported.')
            
            self.drop_path.append(DropPath(drop_path_rate[i]) if drop_path_rate[i] > 0.0 else nn.Identity())
        
    def forward(self, x):
        
        x = self.proj(x)
        
        positions = []
        references = []
        output_joints = []
        output_classes = []
        output_bboxes = []
        for d in range(self.depths):
            x0 = x
            x, pos, ref, joints, classes, bbox = self.attns[d](self.layer_norms[2 * d](x))
            x = self.drop_path[d](x) + x0
            x0 = x
            x = self.mlps[d](self.layer_norms[2 * d + 1](x))
            x = self.drop_path[d](x) + x0
            positions.append(pos)
            references.append(ref)
            output_joints.append(joints)
            output_classes.append(classes)
            output_bboxes.append(bbox)

        return x, positions, references, output_joints, output_classes, output_bboxes

class DAT(nn.Module):

    def __init__(self, img_size=224, patch_size=4, num_classes=1000, expansion=4,
                 dim_stem=96, dims=[96, 192, 384, 768], depths=[2, 2, 6, 2], 
                 heads=[3, 6, 12, 24], 
                 window_sizes=[7, 7, 7, 7],
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, 
                 strides=[-1,-1,-1,-1], offset_range_factor=[1, 2, 3, 4], 
                 stage_spec=[['L', 'D'], ['L', 'D'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']], 
                 groups=[-1, -1, 3, 6],
                 use_pes=[False, False, False, False], 
                 dwc_pes=[False, False, False, False],
                 sr_ratios=[8, 4, 2, 1], 
                 fixed_pes=[False, False, False, False],
                 no_offs=[False, False, False, False],
                 ns_per_pts=[4, 4, 4, 4],
                 use_dwc_mlps=[False, False, False, False],
                 use_conv_patches=False,
                 num_keypoints=19,
                 aux_loss=False,
                 **kwargs):
        super().__init__()

        self.patch_proj = nn.Sequential(
            nn.Conv2d(3, dim_stem, 7, patch_size, 3),
            LayerNormProxy(dim_stem)
        ) if use_conv_patches else nn.Sequential(
            nn.Conv2d(3, dim_stem, patch_size, patch_size, 0),
            LayerNormProxy(dim_stem)
        ) 

        img_size = img_size // patch_size
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        for i in range(4):
            dim1 = dim_stem if i == 0 else dims[i - 1] * 2
            dim2 = dims[i]
            self.stages.append(
                TransformerStage(img_size, window_sizes[i], ns_per_pts[i],
                dim1, dim2, depths[i], stage_spec[i], groups[i], use_pes[i], 
                sr_ratios[i], heads[i], strides[i], 
                offset_range_factor[i], i,
                dwc_pes[i], no_offs[i], fixed_pes[i],
                attn_drop_rate, drop_rate, expansion, drop_rate, 
                dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_dwc_mlps[i], num_keypoints, num_classes)
            )
            img_size = img_size // 2

        self.down_projs = nn.ModuleList()
        for i in range(3):
            self.down_projs.append(
                nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 3, 2, 1, bias=False),
                    LayerNormProxy(dims[i + 1])
                ) if use_conv_patches else nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], 2, 2, 0, bias=False),
                    LayerNormProxy(dims[i + 1])
                )
            )
           
        # self.cls_norm = LayerNormProxy(dims[-1]) 
        # self.cls_head = nn.Linear(dims[-1], num_classes)
        
        self.hardtanh = nn.Hardtanh(min_val=0, max_val=1)
        self.aux_loss = aux_loss
        self.reset_parameters()
    
    def reset_parameters(self):

        for m in self.parameters():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
                
    @torch.no_grad()
    def load_pretrained(self, state_dict):
        
        new_state_dict = {}
        for state_key, state_value in state_dict.items():
            keys = state_key.split('.')
            m = self
            for key in keys:
                if key.isdigit():
                    m = m[int(key)]
                else:
                    m = getattr(m, key)
            if m.shape == state_value.shape:
                new_state_dict[state_key] = state_value
            else:
                # Ignore different shapes
                if 'relative_position_index' in keys:
                    new_state_dict[state_key] = m.data
                if 'q_grid' in keys:
                    new_state_dict[state_key] = m.data
                if 'reference' in keys:
                    new_state_dict[state_key] = m.data
                # Bicubic Interpolation
                if 'relative_position_bias_table' in keys:
                    n, c = state_value.size()
                    l = int(math.sqrt(n))
                    assert n == l ** 2
                    L = int(math.sqrt(m.shape[0]))
                    pre_interp = state_value.reshape(1, l, l, c).permute(0, 3, 1, 2)
                    post_interp = F.interpolate(pre_interp, (L, L), mode='bicubic')
                    new_state_dict[state_key] = post_interp.reshape(c, L ** 2).permute(1, 0)
                if 'rpe_table' in keys:
                    c, h, w = state_value.size()
                    C, H, W = m.data.size()
                    pre_interp = state_value.unsqueeze(0)
                    post_interp = F.interpolate(pre_interp, (H, W), mode='bicubic')
                    new_state_dict[state_key] = post_interp.squeeze(0)
        
        self.load_state_dict(new_state_dict, strict=False)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_kpts, output_joints, outputs_bboxes):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_kpts': b, 'pred_offset': c, 'pred_bboxes': d}
                for a, b, c, d in zip(outputs_class[:-1], outputs_kpts[:-1], output_joints[:-1], outputs_bboxes[:-1])]
    
    def forward(self, x):
        
        x = self.patch_proj(x)
        positions = []
        references = []
        joints = []
        classes = []
        bboxes = []
        for i in range(4):
            x, pos, ref, joint, classs, bbox = self.stages[i](x)
            if i < 3:
                x = self.down_projs[i](x)
            positions.append(pos)
            references.append(ref)
            joints.append(joint)
            classes.append(classs)
            bboxes.append(bbox)
        # x = self.cls_norm(x)
        # x = F.adaptive_avg_pool2d(x, 1)
        # x = torch.flatten(x, 1)
        # x = self.cls_head(x)
        

        output_poses = []
        output_classes = []
        output_bboxes = []
        output_joints = []
        for i in range(4):
            for pos, joint, classs, bbox in zip(positions[i], joints[i], classes[i], bboxes[i]):
                if pos is not None and joint is not None:
                    pos[..., 0].add_(1).div_(2)
                    pos[..., 1].add_(1).div_(2) 
                    pos = pos[..., (1, 0)]
                    root = pos.unsqueeze(-2) + joint[..., 0:1, :]
                    tmp_kpts = root + joint[..., 1:, :]
                    tmp_kpts = torch.cat([root, tmp_kpts], -2)
                    # tmp_kpts = self.hardtanh(tmp_kpts)

                    output_poses.append(tmp_kpts)
                    output_classes.append(classs)
                    output_bboxes.append(bbox)
                    output_joints.append(joint)

            # if ref_pose is not None:
            #     B, H, W, K, P = ref_pose.shape
            #     p = torch.zeros((B, H//2, W//2, K, P)).cuda()
            #     for i in range(H//2):
            #         for j in range(W//2):
            #             p[:,i,j] = torch.mean(ref_pose[:,i*2:i*2+2,j*2:j*2+2,:,:], (1,2))
            #     ref_pose = p
            # if ref_bbox is not None:
            #     B, H, W, K = ref_bbox.shape
            #     b = torch.zeros((B, H//2, W//2, K)).cuda()
            #     for i in range(H//2):
            #         for j in range(W//2):
            #             b[:,i,j] = torch.mean(ref_bbox[:,i*2:i*2+2,j*2:j*2+2,:], (1,2))
            #     ref_bbox = b

        # output_classes = torch.stack(output_classes_list)
        # output_poses = torch.stack(output_poses)

        out = {'pred_logits': output_classes[-1],
               'pred_kpts': output_poses[-1],
               'pred_offset': output_joints[-1],
               'pred_bboxes': output_bboxes[-1]}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(output_classes, output_poses, output_joints, output_bboxes)

        return out, positions, references, output_joints, output_poses, output_classes, output_bboxes


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def process_boxes(self, boxes, target_sizes):
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        return boxes

    def process_kpts(self, kpts, boxes, target_sizes):
        x_c, y_c, bw, bh = boxes.unbind(-1)
        if kpts.shape[-1] != 2 and kpts.shape[-1] != 3:
            kpts = kpts.reshape((kpts.shape[:-1]) + (kpts.shape[-1] // 2, 2))
        if kpts.shape[-1] == 3:
            kpts = kpts[..., :2]
        kpts = box_ops.joint_bbox_to_absolute(kpts, x_c, y_c, bw, bh)
        kpts = kpts.reshape((kpts.shape[:-1]) + (kpts.shape[-1] // 2, 2))
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h], dim=1)
        # boxes = boxes * scale_fct[:, None, :]
        kpts = kpts*scale_fct[:,None,:]
        return kpts
        
        
    @torch.no_grad()
    def forward(self, outputs, target_sizes, results_mask=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of
                          each images of the batch For evaluation, this must be the
                          original image size (before any data augmentation) For
                          visualization, this should be the image size after data
                          augment, but before padding
        """
        out_logits, out_bbox, out_kpts = outputs['pred_logits'], outputs['pred_bboxes'], outputs['pred_kpts']
        out_logits = out_logits.flatten(1, 3)
        out_bbox = out_bbox.flatten(1, 3)
        out_kpts = out_kpts.flatten(1, 3)
        out_kpts = out_kpts.flatten(-2, -1)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # prob = F.softmax(out_logits, -1)
        # scores, labels = prob[..., :-1].max(-1)

        prob = out_logits.sigmoid()

        ###
        # topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        # scores = topk_values

        # topk_boxes = topk_indexes // out_logits.shape[2]
        # labels = topk_indexes % out_logits.shape[2]

        # boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
        ###

        scores, labels = prob.max(-1)
        # scores, labels = prob[..., 0:1].max(-1)
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        out_kpts = out_kpts * scale_fct[:, :2].repeat(1, out_kpts.shape[-1]//2)[:,None,:]

        results = [
            {'scores': s, 'scores_no_object': 1 - s, 'labels': l, 'boxes': b, 'kpts': k}
            for s, l, b, k in zip(scores, labels, boxes, out_kpts)]

        if results_mask is not None:
            for i, mask in enumerate(results_mask):
                for k, v in results[i].items():
                    results[i][k] = v[mask]

        return results
