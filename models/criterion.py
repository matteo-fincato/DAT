# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, accuracy, dice_loss, get_world_size,
                         interpolate, is_dist_avail_and_initialized,
                         nested_tensor_from_tensor_list, sigmoid_focal_loss)
from .oks_loss import OKSLoss
from util import box_ops

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 focal_loss, focal_alpha, focal_gamma, num_keypoints):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their
                         relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of
                    available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_oks_ex = OKSLoss(num_keypoints=num_keypoints)


    def loss_labels(self, outputs, targets, indices, _, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].flatten(1,3)
        src_logits = src_logits.reshape(src_logits.shape[0], -1, src_logits.shape[-1])

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2),
                                  target_classes,
                                  weight=self.empty_weight,
                                  reduction='none')

        loss_ce = loss_ce.sum() / self.empty_weight[target_classes].sum()

        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses


    def loss_labels_focal(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits'].flatten(1,3)
        src_logits = src_logits.reshape(src_logits.shape[0], -1, src_logits.shape[-1])

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=self.focal_gamma)
            # , query_mask=query_mask)

        loss_ce *= src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of
            predicted non-empty boxes. This is not really a loss, it is intended
            for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits'].flatten(1,3)
        pred_logits = pred_logits.reshape(pred_logits.shape[0], -1, pred_logits.shape[-1])

        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses
        

    def loss_kpts(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_kpts' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs['pred_kpts'].flatten(1,3)[idx]
        # src_kpts = src_kpts.flatten(-2,-1)
        target_kpts = torch.cat([t['keypoints'][i,:,:2] for t, (_, i) in zip(targets, indices)], dim=0)
        target_kpts_vis = torch.cat([t['keypoints'][i,:,2:3] for t, (_, i) in zip(targets, indices)], dim=0) > 0

        loss_bbox = target_kpts_vis * F.l1_loss(src_kpts, target_kpts, reduction='none')

        losses = {}
        losses['loss_kpt'] = loss_bbox.sum() / num_boxes

        return losses
    

    def loss_offset(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_offset' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_offset = outputs['pred_offset'].flatten(1,3)[idx]
        src_offset = src_offset[:,1:,:]
        # src_offset = src_offset.flatten(-2,-1)
        
        target_kpts = torch.cat([t['keypoints'][i,:,:2] for t, (_, i) in zip(targets, indices)], dim=0)
        # target_kpts = target_kpts.reshape(target_kpts.shape[0], -1, 2)
        target_offset = target_kpts[:, 1:, :] - target_kpts[:, :1, :]
        # target_offset = target_offset.reshape(target_offset.shape[0], -1)

        target_kpts_vis = torch.cat([t['keypoints'][i,1:,2:3] for t, (_, i) in zip(targets, indices)], dim=0) > 0
    
        loss_bbox = target_kpts_vis * F.l1_loss(src_offset, target_offset, reduction='none')

        losses = {}
        losses['loss_offset'] = loss_bbox.sum() / num_boxes

        return losses
    

    def loss_root(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
           and the GIoU loss targets dicts must contain the key "boxes" containing
           a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
           format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_kpts' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_kpts = outputs['pred_kpts'].flatten(1,3)[idx]
        # src_kpts = src_kpts.flatten(-2,-1)
        src_root = src_kpts[:, 0, :]

        target_kpts = torch.cat([t['keypoints'][i,:,:2] for t, (_, i) in zip(targets, indices)], dim=0)
        target_root = target_kpts[:, 0, :]

        target_kpts_vis = torch.cat([t['keypoints'][i,:,2:3] for t, (_, i) in zip(targets, indices)], dim=0) > 0
        target_kpts_vis = target_kpts_vis[:, 0, :]

        loss_bbox = target_kpts_vis * F.l1_loss(src_root, target_root, reduction='none')

        losses = {}
        losses['loss_root'] = loss_bbox.sum() / num_boxes

        return losses
    

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss
            and the GIoU loss targets dicts must contain the key "boxes" containing
            a tensor of dim [nb_target_boxes, 4]. The target boxes are expected in
            format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_bboxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_bboxes'].flatten(1,3)[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses


    def loss_oks(self, outputs, targets, indices, num_boxes):
        # keypoint targets
        kpt_preds = outputs['pred_kpts'].flatten(1,3)
        kpt_preds = kpt_preds.flatten(-2,-1)
        idx = self._get_src_permutation_idx(indices)
        kpt_targets = torch.zeros_like(kpt_preds)
        kpt_weights = torch.zeros_like(kpt_preds)
        target_boxes = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # pos_gt_kpts = target_boxes.reshape(pos_gt_kpts.shape[0],
        #                                   pos_gt_kpts.shape[-1] // 3, 3)
        valid_idx = target_boxes[:, :, 2] > 0
        pos_kpt_weights = kpt_weights[idx].reshape(target_boxes.shape[0], kpt_weights.shape[-1] // 2, 2)
        pos_kpt_weights[valid_idx] = 1.0
        kpt_weights[idx] = pos_kpt_weights.reshape(pos_kpt_weights.shape[0], kpt_preds.shape[-1])

        factor = kpt_preds.new_tensor(targets[0]['orig_size']).unsqueeze(0)
        pos_gt_kpts_normalized = target_boxes[..., :2]
        # pos_gt_kpts_normalized[..., 0] = pos_gt_kpts_normalized[..., 0] / factor[:, 0:1]
        # pos_gt_kpts_normalized[..., 1] = pos_gt_kpts_normalized[..., 1] / factor[:, 1:2]
        kpt_targets[idx] = pos_gt_kpts_normalized.reshape(target_boxes.shape[0], kpt_preds.shape[-1])
        
        area_targets = kpt_preds.new_zeros(kpt_preds.shape[:-1])  # get areas for calculating oks
        target_areas = torch.cat([t['area'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        area_targets[idx] = target_areas
        
        # construct factors used for rescale keypoints
        factors = []
        for kpt_pred in kpt_preds:
            img_w, img_h = targets[0]['orig_size']
            factor = torch.tensor([img_w, img_h, img_w,
                                          img_h]).unsqueeze(0).repeat(
                                              kpt_pred.size(0), 1).cuda()
            factors.append(factor)
        factors = torch.cat(factors, 0)
        
        # keypoint oks loss
        pos_inds = kpt_weights.sum(-1) > 0
        factors = factors[pos_inds.flatten()][:, :2].repeat(1, kpt_preds.shape[-1] // 2)
        pos_kpt_preds = kpt_preds[pos_inds] * factors
        pos_kpt_targets = kpt_targets[pos_inds] * factors
        pos_areas = area_targets[pos_inds]
        pos_valid = kpt_weights[pos_inds][:,0::2]
        if len(pos_areas) == 0:
            loss_oks = pos_kpt_preds.sum() * 0
        else:
            assert (pos_areas > 0).all()
            # num_total_pos = sum((inds.numel() for inds in indices))
            loss_oks = self.loss_oks_ex(
                pos_kpt_preds,
                pos_kpt_targets,
                pos_valid,
                pos_areas,
                avg_factor=num_boxes)
        
        losses = {'loss_oks': loss_oks}
        
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_focal if self.focal_loss else self.loss_labels,
            'cardinality': self.loss_cardinality,
            # 'boxes': self.loss_boxes,
            # 'oks': self.loss_oks,
            'kpts': self.loss_kpts,
            'offset': self.loss_offset,
            'root': self.loss_root,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied,
                      see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the
        # output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses