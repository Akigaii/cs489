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
        test_size = val_ratio + test_ratio,
        random_state = seed,
        stratify = y if stratified else None
    )
    
    # 3) split temp into validation and test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size = test_ratio / (val_ratio + test_ratio),
        random_state = seed,
        stratify = y_temp if stratified else None
    )
        
    return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    
    



def make_loaders(args, train_df, val_df, test_df):
    train_ds = BreastCancerDataset(train_df, args.image_dir, args.image_size)
    val_ds = BreastCancerDataset(val_df, args.image_dir, args.image_size)
    test_ds = BreastCancerDataset(test_df, args.image_dir, args.image_size)

    # FINISHED: create train, validation, and test dataloaders
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
    val_loader = DataLoader(val_ds, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    test_loader = DataLoader(test_ds, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)
    return train_loader, val_loader, test_loader


def main():
    args = get_args()

    # FINISHED: set seed and choose device
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Running on {device}")

    # FINISHED: read the CSV and prepare label_num
    df = prepare_dataframe(args.csv_path, args.positive_label)
    
    # FINISHED: split the dataframe
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataframe(df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed, args.stratified_split)
    
    # FINISHED: create the dataloaders
    train_df = pd.DataFrame({'case_id': X_train, 'label_num': y_train})
    val_df   = pd.DataFrame({'case_id': X_val,   'label_num': y_val})
    test_df  = pd.DataFrame({'case_id': X_test,  'label_num': y_test})
    train_loader, val_loader, test_loader = make_loaders(args, train_df, val_df, test_df)

    # FINISHED: build model, criterion, and optimizer
    model = build_model(args.model_name).to(device)
    criterion = build_criterion(args.loss_name, args.minority_weight, args.majority_weight, args.gamma).to(device)
    optimizer = Adam(
        model.parameters(),
        lr = args.lr,
        weight_decay = args.weight_decay
    )
    
    # FINISHED: create an experiment folder name
    # Suggested format: M1_bce or M2_focal_w3.0
    exper_folder = f"{args.model_name}_{args.loss_name}"
    if args.loss_name == "wbce" or args.loss_name == "focal":
        exper_folder = exper_folder + f"_w{args.minority_weight}" 
    print(exper_folder)
    os.makedirs(exper_folder, exist_ok = True)
    
    best_val_auroc = -1.0
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # FINISHED: run one training epoch and one validation epoch
        train_mean_loss, train_metrics = run_one_epoch(model, train_loader, criterion, optimizer, device, True)
        val_mean_loss, val_metrics     = run_one_epoch(model, val_loader, criterion, optimizer, device, False)
        print(f"\nEpoch {epoch} finished.")
        print(f"    - train_mean_loss = {train_mean_loss}")
        print(f"    - train_auroc     = {train_metrics['auroc']}")
        print(f"    - val_mean_loss   = {val_mean_loss}")
        print(f"    - val_auroc       = {val_metrics['auroc']}")
        
        # FINISHED: store epoch metrics in history
        epoch_history = {}
        epoch_history["train_mean_loss"] = train_mean_loss
        epoch_history["train_metrics"]   = train_metrics
        epoch_history["val_mean_loss"]   = val_mean_loss
        epoch_history["val_metrics"]     = val_metrics
        history.append(epoch_history)
        

        # FINISHED: update best model based on validation AUROC
        current_auroc = val_metrics["auroc"]
        if current_auroc > best_val_auroc:
            print("\nNew best model found!")
            # print(f"Prev AUROC: {best_val_auroc},   New AUROC: {current_auroc}")
            torch.save(model.state_dict(), os.path.join(exper_folder, f"{exper_folder}_best_model.pt"))
            best_val_auroc = current_auroc
            patience_counter = 0
        else:
            patience_counter += 1
            
        # FINISHED: apply early stopping
        if patience_counter >= args.patience:
            break

    # FINISHED: load the best model state
    model.load_state_dict(torch.load(os.path.join(exper_folder, f"{exper_folder}_best_model.pt"), map_location="cpu"))
    
    # FINISHED: evaluate on the test set
    test_mean_loss, test_metrics = run_one_epoch(model, test_loader, criterion, optimizer, device, False)
    print(f"\nTest Results:")
    print(f" -   test_mean_loss = {test_mean_loss}")
    print(f" -   test_metrics   = {test_metrics}")
    
    # FINISHED: save best_model.pt, history.json, test_metrics.json, split_sizes.json
    torch.save(model.state_dict(), os.path.join(exper_folder, f"{exper_folder}_best_model.pt"))
    
    with open(os.path.join(exper_folder, "history.json"), "w") as f:
        json.dump(history, f, indent = 4)
        
    with open(os.path.join(exper_folder, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent = 4)
        
    split_sizes = {
        "train_ids":  train_df["case_id"].tolist(),
        "val_ids":    val_df["case_id"].tolist(),
        "test_ids":   test_df["case_id"].tolist(),
        "train_size": len(train_df),
        "val_size":   len(val_df),
        "test_size":  len(test_df)
        }
    with open(os.path.join(exper_folder, "split_sizes.json"), "w") as f:
        json.dump(split_sizes, f, indent = 4)
    

if __name__ == '__main__':
    main()

# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M1 --loss_name bce
# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M1 --loss_name wbce
# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M1 --loss_name focal

# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M2 --loss_name bce
# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M2 --loss_name wbce
# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M2 --loss_name focal

# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M1 --loss_name wbce --minority_weight 3.0
# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M1 --loss_name focal --minority_weight 3.0
# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M2 --loss_name wbce --minority_weight 3.0
# python main.py --csv_path C:\Users\aigaui\cs489\as3\Assignment2_Dataset\dataset.csv --image_dir C:\Users\aigaui\cs489\as3\Assignment2_Dataset\images --model_name M2 --loss_name focal --minority_weight 3.0