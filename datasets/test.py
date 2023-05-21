'''
    Test the Datasets implemented in the datasets directory.

    Run using
        python -m datasets.test
'''

from datasets.text_dataset import TextDataset
from tokenizers.base_tokenizer import BaseTokenizer
from tokenizers.char_tokenizer import CharTokenizer
from torch import Tensor, equal
from typing import Type

import unittest

class TextDatasetTester(unittest.TestCase):
    ''' Test the TextDatasetTester class. '''
    def test_getitem(self):
        ''' Test the __getitem__ function in TextDatasetTester. '''
        document = 'Hello, how are you?'
        tokenizer = CharTokenizer(document)

        self.generate_items_and_test([0, 2], document, tokenizer)

    def compare_items(self, item1: dict[str, Tensor], item2: dict[str, Tensor]) -> bool:
        '''
            Compare two dictionaries of type dict[str, Tensor]. Returns True is they contain
            identical values, and False otherwise.
        '''
        item1_keys = sorted(item1.keys())
        item2_keys = sorted(item2.keys())

        # Terminate early if lengths of lists are different.
        if len(item1_keys) != len(item2_keys):
            return False

        for (item1_key, item2_key) in zip(item1_keys, item2_keys):
            if (item1_key != item2_key) or not (equal(item1[item1_key], item2[item2_key])):
                return False
        return True


    def generate_items_and_test(self, indices_to_test: list[int], document: str, tokenizer: Type[BaseTokenizer]):
        '''
            Verify if elements at indices_to_test from TextDataset matches the expected values.

            Args:
                indices_to_test: List of indices where the actual and expected values are to be compared.
                document: String containing characters to use for building the tokenizer vocabulary.
                tokenizer: Tokenizer to use for testing.

            Returns:
        '''
        max_block_size = 8
        dataset = TextDataset(max_block_size = max_block_size, tokenizer = tokenizer, split_name = 'train', train_fraction = 1, document = document)

        for index in indices_to_test:
            actual_item = dataset.__getitem__(index)
            expected_item = {
                'features': Tensor(tokenizer.encode(document[index : index + max_block_size])),
                'labels': Tensor(tokenizer.encode(document[index + 1 : index + max_block_size + 1]))
            }
            self.assertTrue(self.compare_items(expected_item, actual_item), f'{expected_item=} not matching {actual_item=}')



if __name__ == '__main__':
    unittest.main()
