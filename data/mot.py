# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
MOT dataset with tracking training augmentations.
"""
import bisect
import copy
import csv
import os
import random
from pathlib import Path

import torch

from . import transforms as T
from .coco import CocoDetection, make_coco_transforms
from .coco import build as build_coco
from .crowdhuman import build_crowdhuman


class MOT(CocoDetection):

    def __init__(self, *args, **kwargs):
        super(MOT, self).__init__(*args, **kwargs)

    @property
    def sequences(self):
        return self.coco.dataset['sequences']

    # @property
    # def frame_range(self):
    #     if 'frame_range' in self.coco.dataset:
    #         return self.coco.dataset['frame_range']
    #     else:
    #         return {'start': 0, 'end': 1.0}

    def seq_length(self, idx):
        return self.coco.imgs[idx]['seq_length']

    def sample_weight(self, idx):
        return 1.0 / self.seq_length(idx)

    def __getitem__(self, idx):
        random_state = {
            'random': random.getstate(),
            'torch': torch.random.get_rng_state()}

        img, target = self._getitem_from_id(idx, random_state, random_jitter=False)

        return img, target

    def write_result_files(self, results, output_dir):
        """Write the detections in the format for the MOT17Det sumbission

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        """

        files = {}
        for image_id, res in results.items():
            img = self.coco.loadImgs(image_id)[0]
            file_name_without_ext = os.path.splitext(img['file_name'])[0]
            seq_name, frame = file_name_without_ext.split('_')
            frame = int(frame)

            outfile = os.path.join(output_dir, f"{seq_name}.txt")

            # check if out in keys and create empty list if not
            if outfile not in files.keys():
                files[outfile] = []

            for box, score in zip(res['boxes'], res['scores']):
                if score <= 0.7:
                    continue
                x1 = box[0].item()
                y1 = box[1].item()
                x2 = box[2].item()
                y2 = box[3].item()
                files[outfile].append(
                    [frame, -1, x1, y1, x2 - x1, y2 - y1, score.item(), -1, -1, -1])

        for k, v in files.items():
            with open(k, "w") as of:
                writer = csv.writer(of, delimiter=',')
                for d in v:
                    writer.writerow(d)


class WeightedConcatDataset(torch.utils.data.ConcatDataset):

    def sample_weight(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        if hasattr(self.datasets[dataset_idx], 'sample_weight'):
            return self.datasets[dataset_idx].sample_weight(sample_idx)
        else:
            return 1 / len(self.datasets[dataset_idx])


def build_mot(image_set, args):
    if image_set == 'train':
        root = Path(args.DATA.mot_path_train)
    elif image_set == 'val':
        root = Path(args.DATA.mot_path_val)
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided MOT17Det path {root} does not exist'

    split = getattr(args.DATA, f"{image_set}_split")
    split_json = getattr(args.DATA, f"{image_set}_split_json")

    img_folder = root / split
    ann_file = root / f"annotations/{split_json}.json"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.DATA.img_transform, args.DATA.overflow_boxes)

    dataset = MOT(
        img_folder, ann_file, transforms, norm_transforms,
        return_masks=args.DATA.masks,
        overflow_boxes=args.DATA.overflow_boxes,
        remove_no_obj_imgs=False,
        )

    return dataset


def build_mot_posetrack21(image_set, args):
    if image_set == 'train':
        root = Path(args.DATA.mot_path_train)
    elif image_set == 'val':
        root = Path(args.DATA.mot_path_val)
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided PoseTrack21 path {root} does not exist'

    split = getattr(args.DATA, f"{image_set}_split")
    split_json = getattr(args.DATA, f"{image_set}_split_json")

    img_folder = root
    ann_file = root / f"posetrack21_annotations/{split_json}_annotations.json"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.DATA.img_transform, args.DATA.overflow_boxes)

    dataset = MOT(
        img_folder, ann_file, transforms, norm_transforms,
        return_masks=args.DATA.masks,
        overflow_boxes=args.DATA.overflow_boxes,
        remove_no_obj_imgs=False,
        )

    return dataset


def build_mot_posetrack18(image_set, args):
    if image_set == 'train':
        root = Path(args.DATA.mot_path_train)
    elif image_set == 'val':
        root = Path(args.DATA.mot_path_val)
    else:
        ValueError(f'unknown {image_set}')

    assert root.exists(), f'provided PoseTrack21 path {root} does not exist'

    split = getattr(args.DATA, f"{image_set}_split")
    split_json = getattr(args.DATA, f"{image_set}_split_json")

    img_folder = root
    ann_file = root / f"posetrack18_annotations/{split_json}_annotations.json"

    transforms, norm_transforms = make_coco_transforms(
        image_set, args.DATA.img_transform, args.DATA.overflow_boxes)

    dataset = MOT(
        img_folder, ann_file, transforms, norm_transforms,
        return_masks=args.DATA.masks,
        overflow_boxes=args.DATA.overflow_boxes,
        remove_no_obj_imgs=False,
        )

    return dataset
 
 
def build_mot_crowdhuman(image_set, args):
    if image_set == 'train':
        args_crowdhuman = copy.deepcopy(args)
        args_crowdhuman.train_split = args.crowdhuman_train_split

        crowdhuman_dataset = build_crowdhuman('train', args_crowdhuman)

        if getattr(args, f"{image_set}_split") is None:
            return crowdhuman_dataset

    dataset = build_mot(image_set, args)

    if image_set == 'train':
        dataset = torch.utils.data.ConcatDataset(
            [dataset, crowdhuman_dataset])

    return dataset


def build_mot_coco_person(image_set, args):
    if image_set == 'train':
        args_coco_person = copy.deepcopy(args)
        args_coco_person.train_split = args.coco_person_train_split

        coco_person_dataset = build_coco('train', args_coco_person, 'person_keypoints')

        if getattr(args, f"{image_set}_split") is None:
            return coco_person_dataset

    dataset = build_mot(image_set, args)

    if image_set == 'train':
        dataset = torch.utils.data.ConcatDataset(
            [dataset, coco_person_dataset])

    return dataset
