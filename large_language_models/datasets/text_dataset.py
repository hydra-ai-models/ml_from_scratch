'''
    TextDataset is a PyTorch Dataset class which takes text data in a file or a string as input
    and returns at each index a block of text tokens as features and next word tokens as labels.
'''

from tokenizers.base_tokenizer import BaseTokenizer
from torch.utils.data import Dataset
import torch
from typing import Optional, Type
import os

class TextDataset(Dataset):
    '''
        A PyTorch Dataset class which takes text data in a file or a string as input
        and returns at each index a block of text tokens as features and next word tokens as labels.
    '''
    def __init__(self, max_block_size: int, tokenizer: Type[BaseTokenizer], split_name: str, train_fraction: float, filename: Optional[str] = '', document: Optional[str] = '') -> None:
        '''
            Initializer function.

            Args:
                1. max_block_size - Length of the longest block to use for language modeling.
                2. tokenizer: Tokenizer to use for converting text to tokens.
                3. split_name: Specifies if the split is train or validation. Possible values are ["train", "validation"].
                4. train_fraction: Fraction of characters in the dataset to be used for training. Characters from beginning
                    till this fraction will be used as the training dataset, and the rest is used as the validation dataset.
                5. filename: Optional name of filename containing data to be used in the dataset.
                6. document: Optional string containing data to be used in the dataset. One of filename or document should be specified.
        '''
        self._max_block_size = max_block_size
        self._tokenizer = tokenizer
        self.split_name = split_name
        self.train_fraction = train_fraction
        self._filename = filename
        self._document = document

        if not os.path.exists(filename):
            ValueError('Text dataset does not exist at filename {filename}. Please verify the filename and run again.')

        if not len(filename) and not len(document):
            ValueError('One of filename or document should be non empty. Please correct input to TextDataset() and try again.')

        # Read data from file or the input document string.
        if len(filename):
            with open(filename, 'r') as reader:
                text_data = reader.read()
        else:
            text_data = document

        # Retain data only in the input split (train or validation).
        if split_name == 'train':
            self._data = text_data[:int(len(text_data) * train_fraction)]
        else:
            self._data = text_data[int(len(text_data) * train_fraction):]

    def __len__(self) -> int:
        return len(self._data) - self._max_block_size

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        # To avoid tokenizing same tokens twice (in features and labels), tokenize the tokens needed for input and output in one go,
        # and then separate this tokenized block into input and output.
        tokenized_block = self._tokenizer.encode(self._data[idx: idx + self._max_block_size + 1])
        return {'features': torch.Tensor(tokenized_block[:-1]).long(), 'labels': torch.Tensor(tokenized_block[1:]).long()}
