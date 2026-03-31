"""
main.py

Purpose:
- Run one experiment for Assignment 3 using a single train/validation/test split.
- Save the best model based on validation AUROC.

Complete the TODOs in this file only.
"""

import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import get_args
from dataset import BreastCancerDataset
from model import build_model
from train import build_criterion, run_one_epoch


def set_seed(seed: int):
    
    # FINISHED: set the seeds for random, numpy, torch, and torch.cuda
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    


def prepare_dataframe(csv_path: str, positive_label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {'case_id', 'tumor_type'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f'CSV must contain columns: {required_cols}')

    # FINISHED: create a numeric column called label_num
    # positive_label should map to 1, all others to 0
    df['label_num'] = np.where(df['tumor_type'] == positive_label, 1, 0)
 
    return df


def split_dataframe(df, train_ratio, val_ratio, test_ratio, seed, stratified):
    
    # FINISHED (stratified still needs to be done):
    
    # 1) verify the ratios sum to 1.0
    if train_ratio + val_ratio + test_ratio != 1.0:
        raise ValueError("Ratios do not add up to 1.0.")
        
    # 2) split into train and temp
    df = df.sample(frac = 1, random_state = seed).reset_index(drop = True)
    X = df['case_id']
    y = df['label_num']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size = train_ratio, 
        random_state = seed
        # stratify = stratified
    )
    
    # 3) split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size = 1 - (val_ratio / (val_ratio + test_ratio)),
        random_state = seed
        # stratify=stratify_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    
    



def make_loaders(args, train_df, val_df, test_df):
    train_ds = BreastCancerDataset(train_df, args.image_dir, args.image_size)
    val_ds = BreastCancerDataset(val_df, args.image_dir, args.image_size)
    test_ds = BreastCancerDataset(test_df, args.image_dir, args.image_size)

    # TODO: create train, validation, and test dataloaders
    train_loader = None
    val_loader = None
    test_loader = None
    return train_loader, val_loader, test_loader


def main():
    args = get_args()

    # TODO: set seed and choose device

    # TODO: read the CSV and prepare label_num
    # TODO: split the dataframe
    # TODO: create the dataloaders

    # TODO: build model, criterion, and optimizer

    # TODO: create an experiment folder name
    # Suggested format: M1_bce or M2_focal_w3.0

    best_val_auroc = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # TODO: run one training epoch and one validation epoch

        # TODO: store epoch metrics in history

        # TODO: update best model based on validation AUROC
        # TODO: apply early stopping
        pass

    # TODO: load the best model state
    # TODO: evaluate on the test set
    # TODO: save best_model.pt, history.json, test_metrics.json, split_sizes.json


if __name__ == '__main__':
    main()
