import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import get_args
from dataset import BreastCancerDataset
from model import build_model
from train import build_criterion, run_one_epoch
from main import make_loaders


BEST_MODEL = "M1_focal_w3.0"
BEST_MODEL_DIR = os.path.join(os.getcwd(), BEST_MODEL)
BEST_MODEL_SPLIT_SIZES = os.path.join(BEST_MODEL_DIR, "split_sizes.json")
print(BEST_MODEL_SPLIT_SIZES)

args = get_args()

with open(BEST_MODEL_SPLIT_SIZES, 'r', encoding = 'utf-8') as file:
    raw = json.load(file)

test_ids = raw["test"]

# TODO: Compare against dataset test_ids

#TODO: Turn into dataloader
test_ds = BreastCancerDataset(test_ids, args.image_dir, args.image_size)