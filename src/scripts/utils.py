'''
This file implements shared script utils
'''

import os
from pathlib import Path
from typing import List, Tuple, Union

import pytorch_lightning as pl
from decouple import config
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from src.data import OCRDataset

# Get envvars
BATCH_SIZE = config('BATCH_SIZE', default=8, cast=int)
CHECKPOINTS_PATH = config('CHECKPOINTS_PATH', default='.', cast=str)
EXPERIMENTS_SEED = config('EXPERIMENTS_SEED', default=42, cast=int)
T5_TYPE = config('T5_TYPE', default='t5-small', cast=str)
N_EPOCHS = config('N_EPOCHS', default=50, cast=int)


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


def _get_checkpoints_from_path(path: str, file_extension: str = '.ckpt',
                               sort: bool = True) -> Union[List[str], None]:
    '''
    Gets all the checkpoint files from a given path.

    Args:

        - path: path to checkpoint files.
        - file_extension: checkpoint files extension.
        - sort: if the results are sorted or not.

    Returns:

        - A list of the available checkpoints.
    '''
    # Get all files in the path
    files_in_checkpoint_path = os.listdir(f'{CHECKPOINTS_PATH}')

    # list all checkpoints
    checkpoints = [
        ckpt for ckpt in files_in_checkpoint_path if ckpt.endswith(
            file_extension)
    ]

    if len(checkpoints) == 0:
        print('WARNING: No checkpoint found!')
        return None

    return sorted(checkpoints) if sort else checkpoints


def _configure_trainer() -> pl.Trainer:
    '''
    Configures PL trainer from the .env settings.

    Returns:
        - A pl.Trainer instance configured with data.
    '''

    print(f'Checkpoints will be saved to {CHECKPOINTS_PATH}')
    checkpoint_callback = ModelCheckpoint(filepath=CHECKPOINTS_PATH,
                                          # Keeps all checkpoints.
                                          save_top_k=-1,
                                          monitor='val_f1')

    checkpoints = _get_checkpoints_from_path(CHECKPOINTS_PATH, sort=True)
    print(f'Available checkpoints in {CHECKPOINTS_PATH}: {checkpoints}')

    latest_checkpoint_path = None
    if checkpoints is not None:
        latest_checkpoint = checkpoints[-1]
        print(
            f'Using latest checkpoint {latest_checkpoint}'
            ' as a starting point')
        latest_checkpoint_path = f'{CHECKPOINTS_PATH}/{latest_checkpoint}'

    trainer = pl.Trainer(gpus=-1,
                         max_epochs=N_EPOCHS,
                         check_val_every_n_epoch=1,
                         profiler=True,
                         checkpoint_callback=checkpoint_callback,
                         progress_bar_refresh_rate=20,
                         resume_from_checkpoint=latest_checkpoint_path)

    return trainer
