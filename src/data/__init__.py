from decouple import config

from .baseline_dataset import BaselineDataset
from .single_key_dataset import SingleKeyDataset
from .utils import load_text_jpeg_pairs, multiply_dataset_samples

EXPERIMENT_DATASET_KEY = config(
    'EXPERIMENT_DATASET_KEY', default='baseline', cast=str)


DATASETS_DICT = {
    'baseline': BaselineDataset,
    'single-key': SingleKeyDataset
}

# OCR dataset is the class that is meant to be used to
OCRDataset = DATASETS_DICT[EXPERIMENT_DATASET_KEY]
