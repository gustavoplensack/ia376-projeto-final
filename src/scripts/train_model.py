'''
This file implements the training script
'''

from decouple import config
from pytorch_lightning import seed_everything

from src.models import T5Module
from src.scripts.utils import (_configure_trainer, _make_dataloader,
                               _make_datasets)

# Get env vars
EXPERIMENTS_SEED = config('EXPERIMENTS_SEED', default=42, cast=int)
T5_TYPE = config('T5_TYPE', default='t5-small', cast=str)


def main():
    '''
    Instantiates datasets, dataloaders and model.

    Then it starts training!
    '''
    seed_everything(EXPERIMENTS_SEED)

    print('Instantiating datasets ...')
    train_dataset, test_dataset = _make_datasets()

    print('Instantiating dataloaders ...')
    train_dataloader = _make_dataloader(train_dataset, is_train=True)
    test_dataloader = _make_dataloader(test_dataset, is_train=False)

    print(f'Instantiating T5 module with a {T5_TYPE} ...')
    model = T5Module(train_dataloader, test_dataloader, test_dataloader)

    trainer = _configure_trainer()
    trainer.fit(model)
    return


# Calls the train script main function
main()
