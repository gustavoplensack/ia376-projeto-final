'''
This file implements utilites to handle data and datasets
'''
import glob
import os
import shutil
from typing import List, Tuple

from decouple import config

from src.errors import ExperimentMisconfigurationError

EXPERIMENT_DATASET_KEY = config(
    'EXPERIMENT_DATASET_KEY', default='baseline', cast=str)


def multiply_dataset_samples(path_to_files: str, key_pool: List[str]):
    '''

    Multiply datasets file to cause simple augmentation of samples.
    This function should be only used in a single key-experiments.


    Args:

        - path_to files: path to original files.
        - key_pool: list of keys to multiply files. Example:
        ['comapny','address']

    Raises:

        - ExperimentMisconfigurationError if EXPERIMENT_DATASET_KEY is not
        configured as 'single-key'

    '''

    if EXPERIMENT_DATASET_KEY != 'single-key':
        raise ExperimentMisconfigurationError(
            'To multiply dataset samples, experiment must be single-key '
            f'received experiment was: {EXPERIMENT_DATASET_KEY}.'
        )

    files_in_path = os.listdir(path_to_files)

    for file in files_in_path:
        if file.endswith('.txt') or file.endswith('.jpg'):

            # create a copy with file-key
            for key in key_pool:
                shutil.copyfile(f'{path_to_files}{file}',
                                f'{path_to_files}{file}-{key}')

            # Removes base file
            os.remove(f'{path_to_files}{file}')

    return


def load_text_jpeg_pairs(path_to_files: str) -> List[Tuple[str]]:
    '''
    Load text and jped samples from path_to_files

    Args:

        - path_to_files: The path to the files.

    Returns:

        - A list of tuples of [(path_to_image, path_to_text)].

    '''

    img_files = glob.glob(f'{path_to_files}/*.jpg*')
    txt_files = glob.glob(f'{path_to_files}/*.txt*')

    # sortind data, to assert that the data is organized
    img_files.sort()
    txt_files.sort()

    return [sample for sample in zip(img_files, txt_files)]
