from decouple import config
from pytorch_lightning.loggers import NeptuneLogger

# Get envvars
BATCH_SIZE = config('BATCH_SIZE', default=8, cast=int)
CHECKPOINTS_PATH = config('CHECKPOINTS_PATH', default='.', cast=str)
EXPERIMENTS_SEED = config('EXPERIMENTS_SEED', default=42, cast=int)
T5_TYPE = config('T5_TYPE', default='t5-small', cast=str)
SEQ_LEN = config('SEQ_LEN', default=128, cast=int)
LEARNING_RATE = config('LEARNING_RATE', default=3e-4, cast=float)
N_EPOCHS = config('N_EPOCHS', default=50, cast=int)

# Neptune confs
NEPTUNE_API_TOKEN = config(
    'NEPTUNE_API_TOKEN', default='', cast=str)
NEPTUNE_PROJECT = config(
    'NEPTUNE_PROJECT', default='gplensack/IA376-Final', cast=str)
NEPTUNE_EXPERIMENT_NAME = config(
    'NEPTUNE_EXPERIMENT_NAME', default='teste', cast=str)


# Usage of NeptuneLogger based on:
# https://towardsdatascience.com/how-to-keep-track-of-pytorch-lightning-experiments-with-neptune-af467ec05600
NEPTUNE_LOGGER = NeptuneLogger(
    api_key=NEPTUNE_API_TOKEN,
    project_name=NEPTUNE_PROJECT,
    close_after_fit=True,
    experiment_name=NEPTUNE_EXPERIMENT_NAME,
    params={
        'BATCH_SIZE': BATCH_SIZE,
        'CHECKPOINTS_PATH': CHECKPOINTS_PATH,
        'EXPERIMENTS_SEED': EXPERIMENTS_SEED,
        'T5_TYPE': T5_TYPE,
        'SEQ_LEN': SEQ_LEN,
        'LEARNING_RATE': LEARNING_RATE,
        'N_EPOCHS': N_EPOCHS,
    })
