from pathlib import Path

from src.data import multiply_dataset_samples

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

KEY_POOL = ['address', 'company']

multiply_dataset_samples(f'{PROJECT_ROOT}/data/train/', KEY_POOL)
multiply_dataset_samples(f'{PROJECT_ROOT}/data/test/', KEY_POOL)
