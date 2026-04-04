# CS 489/689 Assignment 3 Starter Code

This package contains the starter code for Assignment 3.

## Dataset format
The CSV file must contain these columns:
- `case_id`
- `tumor_type`

Example:
```csv
case_id,tumor_type
BUS_ID_0000,Malignant
BUS_ID_0001,Benign
```

The image folder should follow this layout:
```text
images/
  BUS_ID_0000/
    image.png
  BUS_ID_0001/
    image.png
```

## Required experiments
- M1 + BCE
- M1 + WBCE
- M1 + Focal
- M2 + BCE
- M2 + WBCE
- M2 + Focal

For WBCE and Focal, use these weight settings:
- `(2.0, 1.0)`
- `(3.0, 1.0)`

For Focal, use:
- `gamma = 2.0`

## Example commands
```bash
python main.py --csv_path dataset.csv --image_dir images --model_name M1 --loss_name bce --stratified_split
python main.py --csv_path dataset.csv --image_dir images --model_name M2 --loss_name wbce --minority_weight 2.0 --majority_weight 1.0 --stratified_split
python main.py --csv_path dataset.csv --image_dir images --model_name M2 --loss_name focal --minority_weight 3.0 --majority_weight 1.0 --gamma 2.0 --stratified_split
```
