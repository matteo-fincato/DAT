# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia 
# --------------------------------------------------------

from .dat import DAT
from .criterion import SetCriterion
from .matcher import build_matcher
from .dat import PostProcess

def build_model(config):

    model_type = config.MODEL.TYPE
    if model_type == 'dat':
        model = DAT(**config.MODEL.DAT)
        if config.LOSS.focal_loss:
            postprocessors = {'bbox': PostProcess()}
        else:
            postprocessors = {'bbox': PostProcess()}
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model, postprocessors

def build_criterion(config):

    weight_dict = {'loss_ce': config.LOSS.CE,
                #    'loss_bbox': config.LOSS.BBOX,
                   'loss_giou': config.LOSS.GIOU,
                #    'loss_oks': config.LOSS.OKS,
                   'loss_kpt': config.LOSS.KPT,
                   'loss_offset': config.LOSS.OFFSET,
                   'loss_root': config.LOSS.ROOT,}

    if config.LOSS.masks:
        weight_dict["loss_mask"] = config.LOSS.mask_loss_coef
        weight_dict["loss_dice"] = config.LOSS.dice_loss_coef

    # TODO this is a hack
    if config.MODEL.DAT.aux_loss:
        aux_weight_dict = {}
        for i in range(config.MODEL.DAT.def_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

        weight_dict.update(aux_weight_dict)

    # losses = ['labels', 'boxes', 'cardinality', 'oks', 'kpts', 'offset', 'root']
    losses = ['labels', 'cardinality', 'kpts', 'offset', 'root']
    if config.LOSS.masks:
        losses.append('masks')

    matcher = build_matcher(config)
    
    criterion = SetCriterion(
    num_classes=config.MODEL.DAT.num_classes + 1 if config.LOSS.focal_loss else config.MODEL.DAT.num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=config.LOSS.eos_coef,
    losses=losses,
    focal_loss=config.LOSS.focal_loss,
    focal_alpha=config.LOSS.focal_alpha,
    focal_gamma=config.LOSS.focal_gamma,
    num_keypoints=config.MODEL.DAT.num_keypoints,)
    
    return criterion