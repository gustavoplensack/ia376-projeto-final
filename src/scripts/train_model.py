'''
This file implements the training script
'''
from pathlib import Path

from src.data import OCRDataset

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


train_files = f'{PROJECT_ROOT}/data/train'
train_csv_path = f'{PROJECT_ROOT}/data/ocr_baseline.csv'

dataset = OCRDataset(train_files, train_csv_path)


print(dataset[0])
