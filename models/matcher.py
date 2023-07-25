# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from models.match_cost import KptL1Cost, OksCost

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best
    predictions, while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1,
                 focal_loss: bool = False, focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 kpt_cost: float = 1, oks_cost: float = 1, num_keypoints: int = 19):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates
                       in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the
                       matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_kpt = kpt_cost
        self.cost_oks = oks_cost
        self.kpt_cost = KptL1Cost(weight=kpt_cost)
        self.oks_cost = OksCost(num_keypoints=num_keypoints, weight=oks_cost)
        self.focal_loss = focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the
                                classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted
                               box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target
                     is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number
                           of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        out_prob = outputs["pred_logits"].reshape(outputs["pred_logits"].shape[0], -1, outputs["pred_logits"].shape[-1])
        batch_size, num_queries = out_prob.shape[:2]
        out_kpt = outputs["pred_kpts"].flatten(-2,-1)
        out_kpt = out_kpt.reshape(out_kpt.shape[0], -1, out_kpt.shape[-1])
        out_bbox = outputs["pred_bboxes"].reshape(outputs["pred_bboxes"].shape[0], -1, outputs["pred_bboxes"].shape[-1])

        # We flatten to compute the cost matrices in a batch
        #
        # [batch_size * num_queries, num_classes]
        if self.focal_loss:
            out_prob = out_prob.flatten(0, 1).sigmoid()
        else:
            out_prob = out_prob.flatten(0, 1).softmax(-1)

        # [batch_size * num_queries, 4]
        out_bbox = out_bbox.flatten(0, 1)
        out_kpt = out_kpt.flatten(0, 1)

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        tgt_kpt = torch.cat([v["keypoints"] for v in targets])
        tgt_area = torch.cat([v["area"] for v in targets])
        
        # # Separate bbox and kpts
        # out_bbox = out[..., -4:]
        # out_kpt = out[..., :-4]
        
        # Compute the classification cost.
        if self.focal_loss:
            neg_cost_class = (1 - self.focal_alpha) * (out_prob ** self.focal_gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = self.focal_alpha * ((1 - out_prob) ** self.focal_gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Contrary to the loss, we don't use the NLL, but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute factor
        img_h, img_w = targets[0]['orig_size']
        factor = tgt_kpt.new_tensor([img_w, img_h, 1.0]).unsqueeze(0)
        
        # keypoint regression L1 cost
        gt_kpt_reshape = tgt_kpt.reshape(tgt_kpt.shape[0], -1, 3)
        valid_kpt_flag = gt_kpt_reshape[..., -1]
        kpt_pred_tmp = out_kpt.clone().detach().reshape(out_kpt.shape[0], -1, 2)
        normalize_gt_keypoints = gt_kpt_reshape[..., :2]
        kpt_cost = self.kpt_cost(kpt_pred_tmp, normalize_gt_keypoints, valid_kpt_flag)
        
        # keypoint OKS cost
        kpt_pred_tmp = out_kpt.clone().detach().reshape(out_kpt.shape[0], -1, 2)
        kpt_pred_tmp = kpt_pred_tmp * factor[:, :2].unsqueeze(0)
        gt_kpt_reshape = gt_kpt_reshape * factor.unsqueeze(0)
        oks_cost = self.oks_cost(kpt_pred_tmp, gt_kpt_reshape[..., :2], valid_kpt_flag, tgt_area)
        
        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox),
            box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        cost_matrix = self.cost_bbox * cost_bbox \
            + self.cost_class * cost_class \
            + self.cost_giou * cost_giou \
            + self.cost_kpt * kpt_cost \
            + self.cost_oks * oks_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        # for i, target in enumerate(targets):
        #     if 'track_query_match_ids' not in target:
        #         continue

        #     prop_i = 0
        #     for j in range(cost_matrix.shape[1]):
        #         # if target['track_queries_fal_pos_mask'][j] or target['track_queries_placeholder_mask'][j]:
        #         if target['track_queries_fal_pos_mask'][j]:
        #             # false positive and palceholder track queries should not
        #             # be matched to any target
        #             cost_matrix[i, j] = np.inf
        #         elif target['track_queries_mask'][j]:
        #             track_query_id = target['track_query_match_ids'][prop_i]
        #             prop_i += 1

        #             cost_matrix[i, j] = np.inf
        #             cost_matrix[i, :, track_query_id + sum(sizes[:i])] = np.inf
        #             cost_matrix[i, j, track_query_id + sum(sizes[:i])] = -1

        indices = [linear_sum_assignment(c[i])
                   for i, c in enumerate(cost_matrix.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(
        cost_class=args.LOSS.set_cost_class,
        cost_bbox=args.LOSS.set_cost_bbox,
        cost_giou=args.LOSS.set_cost_giou,
        focal_loss=args.LOSS.focal_loss,
        focal_alpha=args.LOSS.focal_alpha,
        focal_gamma=args.LOSS.focal_gamma,
        num_keypoints=args.MODEL.DAT.num_keypoints,
        kpt_cost=args.LOSS.kpt_cost,
        oks_cost=args.LOSS.oks_cost,)
