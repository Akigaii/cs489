"""
dataset.py

Purpose:
- Load the breast ultrasound dataset from a dataframe.
- Build image paths using: image_dir / case_id / image.png
- Return (image_tensor, label_tensor, case_id).

CSV columns used:
- case_id
- label_num

Complete the TODOs in this file only.
"""

import os
from typing import Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class BreastCancerDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, image_dir: str, image_size: int = 224):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        
        # FINISHED: build a transform pipeline that:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD  = [0.229, 0.224, 0.225]
        transform_pipeline = transforms.Compose(
            transforms.resize(image_size, image_size),  # 1) resizes to (image_size, image_size)
            transforms.ToTensor(),                      # 2) converts to tensor
            transforms.Normalize(mean = IMAGENET_MEAN,  # 3) normalizes using ImageNet mean/std
                                 std  = IMAGENET_STD )   
        )        
        self.transform = transform_pipeline

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.dataframe.iloc[idx]
        case_id = row['case_id']
        image_path = os.path.join(self.image_dir, case_id, 'image.png')

        # FINISHED: load the image from image_path and convert it to RGB
        image = Image.open(image_path)
        image = image.convert('RGB')

        # FINISHED: apply self.transform to the image
        image = self.transform(image)

        # FINISEHD: convert row['label_num'] to a float tensor
        label = torch.tensor(row['label_num'])

        return image, label, case_id
