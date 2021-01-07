'''
This file implements the training script
'''
import os
from pathlib import Path
from typing import Tuple

import pytorch_lightning as pl
import torch
from decouple import config
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.data import OCRDataset
from src.models import T5Module

# Get envvars
BATCH_SIZE = config('BATCH_SIZE', default=8, cast=int)
BATCH_SIZE = config('BATCH_SIZE', default=8, cast=int)

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Get data files
TRAIN_FILES = f'{PROJECT_ROOT}/data/train'
TEST_FILES = f'{PROJECT_ROOT}/data/test'
GCV_OCR_CSV = f'{PROJECT_ROOT}/data/ocr_baseline.csv'


def _make_datasets() -> Tuple[OCRDataset, OCRDataset]:
    '''
    Instantiate train and test datasets train and test data.

    Returns:

        - A tuple with train and test datasets.
    '''

    train_dataset = OCRDataset(TRAIN_FILES, GCV_OCR_CSV)
    test_dataset = OCRDataset(TEST_FILES, GCV_OCR_CSV)

    return train_dataset, test_dataset


def _make_dataloader(dataset: OCRDataset, is_train: bool = True) -> DataLoader:
    '''
    Instantiantes a dataloader class for a given dataset.
    NOTE: this class is just a wrapper for default torch implemetation.

    Args:

        - dataset instance of OCRDataset
        - is_train: parameter to indicate if it is a train instance, if is
            makes it shuffle the data.

    Returns:

        - A dataloader instance for the specified dataset.
    '''

    batch_size = BATCH_SIZE if is_train else 1

    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def main():
    '''
    Instantiates datasets, dataloaders and model.

    Then it starts training!
    '''
    seed_everything(42)

    print('Instantiating datasets ...')
    train_dataset, test_dataset = _make_datasets()

    train_dataloader = _make_dataloader(train_dataset, is_train=True)
    test_dataloader = _make_dataloader(test_dataset, is_train=False)

    model = T5Module(train_dataloader, test_dataloader, test_dataloader)

    checkpoint_path = './epoch=49.ckpt'
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    print(f'Files in {checkpoint_dir}: {os.listdir(checkpoint_dir)}')
    print(f'Saving checkpoints to {checkpoint_dir}')
    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_dir,
                                          # Keeps all checkpoints.
                                          save_top_k=-1,
                                          monitor='val_f1')

    resume_from_checkpoint = None
    if os.path.exists(checkpoint_path):
        print(f'Restoring checkpoint: {checkpoint_path}')
        resume_from_checkpoint = checkpoint_path
        print('Using checkpoint:', resume_from_checkpoint)

    trainer = pl.Trainer(gpus=1,
                         max_epochs=1,
                         check_val_every_n_epoch=1,
                         profiler=True,
                         checkpoint_callback=checkpoint_callback,
                         progress_bar_refresh_rate=50,
                         resume_from_checkpoint=resume_from_checkpoint)

    trainer.fit(model)
    return


# Calls the train script main function
main()
