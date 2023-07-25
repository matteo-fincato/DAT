from .build import build_loader
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO

def get_coco_api_from_dataset(dataset: Subset) -> COCO:
    """Return COCO class from PyTorch dataset for evaluation with COCO eval."""
    for _ in range(10):
        # if isinstance(dataset, CocoDetection):
        #     break
        if isinstance(dataset, Subset):
            dataset = dataset.dataset

    if not isinstance(dataset, CocoDetection):
        raise NotImplementedError

    return dataset.coco