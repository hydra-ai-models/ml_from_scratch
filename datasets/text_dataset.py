'''
    Datasets for Natural Language Processing (NLP) tasks like language modeling.
'''

from tokenizers.base_tokenizer import BaseTokenizer
from torch.utils.data import Dataset
from torch import Tensor
from typing import Optional, Type
import os

class TextDataset(Dataset):
    def __init__(self, max_block_size: int, tokenizer: Type[BaseTokenizer], filename: Optional[str] = '', document: Optional[str] = '') -> None:
        self._max_block_size = max_block_size
        self._tokenizer = tokenizer
        self._filename = filename
        self._document = document

        if not os.path.exists(filename):
            ValueError('Text dataset does not exist at filename {filename}. Please verify the filename and run again.')

        if not len(filename) and not len(document):
            ValueError('One of filename or document should be non empty. Please correct input to TextDataset() and try again.')

        if len(filename):
            with open(filename, 'r') as reader:
                self._data = reader.read()
        else:
            self._data = document

    def __len__(self) -> int:
        return len(self._data) - self._max_block_size

    def __getitem__(self, idx) -> dict[str, Tensor]:
        # To avoid tokenizing same tokens twice, tokenize the tokens needed for input and output in one go,
        # and then separate this tokenized block into input and output.
        tokenized_block = self._tokenizer.encode(self._data[idx: idx + self._max_block_size + 1])
        return {'features': Tensor(tokenized_block[:-1]), 'labels': Tensor(tokenized_block[1:])}

def read_dataset(filename: str, visualize: bool = False) -> str:
    ''' Read a text file and return the contents as a string.

        Args:
            filename - Path of the text file to be read.
            visualize - Whether to visualize the statistics of the file contents.

        Returns
            Content of the file as a string.
    '''
    with open(filename, 'r') as reader:
        data = reader.read()

    if visualize:
        print(f'Visualizing dataset at path {filename}.')
        print(f'First 100 characters:\n{data[0:100]}.')
        print(f'Length: {len(data)}.')
    return data
