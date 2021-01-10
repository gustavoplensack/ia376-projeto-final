'''
This file implements a Pytorch dataset class that handles data the
pre-processed data.
'''
import json

import pandas as pd
from decouple import config
from torch.utils.data import Dataset
from transformers import T5Tokenizer

from .utils import load_text_jpeg_pairs

T5_TYPE = config('T5_TYPE', default='t5-small', cast=str)
SEQ_LEN = config('SEQ_LEN', default=128, cast=int)


class SingleKeyDataset(Dataset):
    '''
    Pytorch's dataset abstraction to build images from text GCV OCR of the
    recipts alongside with the document annotations provided by the SROIE
    official dataset.

    This represents the single-key experiment, where only 'address' and
    'company' are evaluated at a single pred for each.

    Data is prefixed with 'address' or 'company' as input.

    Args:

        - path_to_jpg_txt_pairs: path to jpg/txt pairs
        - path_to_csv_file: defines the path to the ocrized sroie 2019 csv.
    '''

    def __init__(self, path_to_jpg_txt_pairs: str, path_to_csv_file: str):

        self._dataset = load_text_jpeg_pairs(path_to_jpg_txt_pairs)
        self._tokenizer = T5Tokenizer.from_pretrained(T5_TYPE)
        self.csv_file = pd.read_csv(path_to_csv_file, index_col=0)

    def index_csv_from_current_sample(self, file_name: str) -> str:
        '''
        Idexes the CSV file from the file name
        '''
        # cleans the 'slice/FILE.txt' to be only SLICE.
        cleaned_file_name = file_name.split('/')[-1]
        cleaned_file_name = cleaned_file_name.split('.')[0]

        ocrized_sample = \
            self.csv_file.loc[cleaned_file_name]['ocr_gvision_output']

        json_sample = json.loads(ocrized_sample).get('textAnnotations')[0]

        ocrized_text = json_sample.get('description')

        ocrized_text = ocrized_text.replace('\n', ' ')

        return ocrized_text

    def get_padded_input(self, text_file: str):
        '''
        Gets a input OCR and pads with the key.
        '''

        key = text_file.split('-')[-1]

        text = self.index_csv_from_current_sample(text_file)

        return f'{key}: {text}'

    def __len__(self) -> int:
        return len(self._dataset)

    def read_from_json(self, json_path: str):
        '''
        Reads the data from the JSON.
        '''
        data = json.load(open(json_path))

        info_key = json_path.split('-')[-1]

        sample = data.get(info_key, '')

        return f'{info_key}: {sample}'

    def __getitem__(self, idx: int):

        _, txt_file = self._dataset[idx]

        target = self.read_from_json(txt_file)
        source = self.get_padded_input(txt_file)

        source_tokenized = self._tokenizer.encode_plus(source,
                                                       padding='max_lenght',
                                                       max_length=SEQ_LEN,
                                                       return_tensors='pt')

        target_tokenized = self._tokenizer.encode_plus(target,
                                                       padding='max_lenght',
                                                       max_length=SEQ_LEN,
                                                       return_tensors='pt')

        source_token_ids = source_tokenized['input_ids'].squeeze()
        source_mask = source_tokenized['attention_mask'].squeeze()
        original_source = source

        target_token_ids = target_tokenized['input_ids'].squeeze()
        target_mask = target_tokenized['attention_mask'].squeeze()
        original_target = target

        return (source_token_ids, source_mask, original_source,
                target_token_ids, target_mask, original_target)
