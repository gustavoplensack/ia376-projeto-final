'''
Implements PL module for T5
'''
from random import choice

import pytorch_lightning as pl
from decouple import config
from numpy import average
from torch import optim, stack
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.log.nepune_logger import NEPTUNE_LOGGER
from src.metrics import compute_exact_match, compute_f1

# Experiment configuration
T5_TYPE = config('T5_TYPE', default='t5-small', cast=str)
SEQ_LEN = config('SEQ_LEN', default=128, cast=int)
LEARNING_RATE = config('LEARNING_RATE', default=3e-4, cast=float)

# Get a tokenizer instance
T5_TOK = T5Tokenizer.from_pretrained(T5_TYPE)


class T5Module(pl.LightningModule):
    '''
    Neural network built with an efficient-net for image feature extraction and
    a T5 for text generation.
    '''

    def __init__(self, train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 test_dataloader: DataLoader):
        super().__init__()

        self.t5 = \
            T5ForConditionalGeneration.from_pretrained(T5_TYPE,
                                                       return_dict=True,
                                                       output_attentions=True)

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def forward(self, x_tokens, x_mask, x_original,
                y_tokens, y_mask, y_original):

        if self.training:
            loss = self.t5.forward(
                input_ids=x_tokens,
                labels=y_tokens,
            )[0]

            return loss
        else:
            predicted_token_ids = self.t5.generate(
                x_tokens, max_length=SEQ_LEN
            )

            return predicted_token_ids

    def training_step(self, batch, batch_idx):

        x_tokens, x_mask, x_original, y_tokens, y_mask, y_original = batch
        loss = self(x_tokens, x_mask, x_original,
                    y_tokens, y_mask, y_original)

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        loss = stack([x['loss'] for x in outputs]).mean()

        self.log('train_loss', loss)

        return

    def validation_step(self, batch, batch_idx):
        x_tokens, x_mask, x_original, y_tokens, y_mask, y_original = batch

        preds = self(x_tokens, x_mask, x_original,
                     y_tokens, y_mask, y_original)

        decoded_preds = [T5_TOK.decode(
            pred, skip_special_tokens=True) for pred in preds]

        return {"pred": decoded_preds, "target": y_original}

    def validation_epoch_end(self, outputs):
        trues = sum([list(x['target']) for x in outputs], [])
        preds = sum([list(x['pred']) for x in outputs], [])

        # Select a random sample from the trues and preds
        true, pred = choice(list(zip(trues, preds)))

        NEPTUNE_LOGGER.experiment.log_text(
            'pred_vs_target',
            f"Epoch: {self.current_epoch} \n tgt: {true}\n prd: {pred}\n")

        em = average([compute_exact_match(g, r) for g, r in zip(preds, trues)])
        f1 = average([compute_f1(g, r) for g, r in zip(preds, trues)])

        # Logging metrics
        self.log("val_em", em, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)

        return

    def test_step(self, batch, batch_idx):
        x_tokens, x_mask, x_original, y_tokens, y_mask, y_original = batch

        preds = self(x_tokens, x_mask, x_original,
                     y_tokens, y_mask, y_original)

        decoded_preds = [T5_TOK.decode(
            pred, skip_special_tokens=True) for pred in preds]

        return {"pred": decoded_preds, "target": y_original}

    def test_epoch_end(self, outputs):
        trues = sum([list(x['target']) for x in outputs], [])
        preds = sum([list(x['pred']) for x in outputs], [])

        # Select a random sample from the trues and preds
        true, pred = choice(list(zip(trues, preds)))

        print(f"\n tgt: {true}\n prd: {pred}\n")

        em = average([compute_exact_match(g, r) for g, r in zip(preds, trues)])
        f1 = average([compute_f1(g, r) for g, r in zip(preds, trues)])

        # Logging metrics
        self.log("test_em", em, prog_bar=True)
        self.log("test_f1", f1, prog_bar=True)

        return

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=LEARNING_RATE)
