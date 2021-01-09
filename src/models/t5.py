'''
Implements PL module for T5
'''
from random import choice

import neptune
import pytorch_lightning as pl
from decouple import config
from numpy import average
from torch import optim, stack
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.metrics import compute_exact_match, compute_f1

# Experiment configuration
T5_TYPE = config('T5_TYPE', default='t5-small', cast=str)
SEQ_LEN = config('SEQ_LEN', default=128, cast=int)
LEARNING_RATE = config('LEARNING_RATE', default=3e-4, cast=float)

# Get a tokenizer instance
T5_TOK = T5Tokenizer.from_pretrained(T5_TYPE)


# Neptune confs
NEPTUNE_API_TOKEN = config(
    'NEPTUNE_API_TOKEN', default=neptune.ANONYMOUS_API_TOKEN, cast=str)
NEPTUNE_PROJECT = config(
    'NEPTUNE_PROJECT', default='gplensack/IA376-Final', cast=str)
NEPTUNE_EXPERIMENT_NAME = config(
    'NEPTUNE_EXPERIMENT_NAME', default='teste', cast=str)


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

        # Neptune confs
        neptune.init(NEPTUNE_PROJECT, NEPTUNE_API_TOKEN)
        neptune.create_experiment(name=NEPTUNE_EXPERIMENT_NAME)

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

        tqdm_dict = {"train_loss": loss}
        neptune.log_metric('train_loss', loss)

        return {"progress_bar": tqdm_dict}

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

        neptune.log_text('pred_vs_target',
                         f"Epoch: {self.current_epoch} \n tgt: {true}\n prd: {pred}\n")

        em = average([compute_exact_match(g, r) for g, r in zip(preds, trues)])
        f1 = average([compute_f1(g, r) for g, r in zip(preds, trues)])

        # Logging metrics
        neptune.log_metric('val_em', em)
        neptune.log_metric('val_f1', f1)
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
        neptune.log_metric('test_em', em)
        neptune.log_metric('test_f1', f1)
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
