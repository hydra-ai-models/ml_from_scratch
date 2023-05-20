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

class CharTokenizerTester(unittest.TestCase):
    def test_getitem(self):
        # Create tokenizer.
        document = 'Hello, how are you?'
        tokenizer = CharTokenizer(document)

        self.generate_items_and_test([0, 2], document, tokenizer)

    def compare_items(self, item1: dict[str, Tensor], item2: dict[str, Tensor]) -> bool:
        item1_keys = sorted(item1.keys())
        item2_keys = sorted(item2.keys())
        if len(item1_keys) != len(item2_keys):
            return False

        for (item1_key, item2_key) in zip(item1_keys, item2_keys):
            if (item1_key != item2_key) or not (equal(item1[item1_key], item2[item2_key])):
                return False
        return True


    def generate_items_and_test(self, indices_to_test: list[int], document: str, tokenizer: Type[BaseTokenizer]):
        max_block_size = 8
        dataset = TextDataset(max_block_size = max_block_size, tokenizer = tokenizer, document = document)

        for index in indices_to_test:
            actual_item = dataset.__getitem__(index)
            expected_item = {
                'features': Tensor(tokenizer.encode(document[index : index + max_block_size])),
                'labels': Tensor(tokenizer.encode(document[index + 1 : index + max_block_size + 1]))
            }
            self.assertTrue(self.compare_items(expected_item, actual_item), f'{expected_item=} not matching {actual_item=}')



if __name__ == '__main__':
    unittest.main()
