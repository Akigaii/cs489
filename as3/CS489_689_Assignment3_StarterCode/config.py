import argparse


def get_args():
    parser = argparse.ArgumentParser(description='CS 489/689 Assignment 3: CNN for breast cancer image classification')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to dataset CSV with case_id and tumor_type columns')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing BUS case folders')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for saved models and metrics')

    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=2, help='Dataloader workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--model_name', type=str, choices=['M1', 'M2'], required=True, help='Model architecture')
    parser.add_argument('--loss_name', type=str, choices=['bce', 'wbce', 'focal'], required=True, help='Loss function')
    parser.add_argument('--minority_weight', type=float, default=2.0, help='Minority class weight for WBCE/Focal')
    parser.add_argument('--majority_weight', type=float, default=1.0, help='Majority class weight for Focal')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma for focal loss')

    parser.add_argument('--train_ratio', type=float, default=0.70, help='Training split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Testing split ratio')
    parser.add_argument('--stratified_split', action='store_true', help='Use stratified train/val/test split')
    parser.add_argument('--positive_label', type=str, default='Malignant', help='Positive class name in tumor_type column')

    return parser.parse_args()
