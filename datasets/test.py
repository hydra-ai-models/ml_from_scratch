'''
    Test the Datasets implemented in the datasets directory.

    Run using
        python -m datasets.test
'''

from datasets.text_dataset import TextDataset
from tokenizers.char_tokenizer import CharTokenizer
from torch import Tensor

import unittest

class CharTokenizerTester(unittest.TestCase):
    def test_getitem(self):
        # Create tokenizer.
        document = 'Hello, how are you?'
        tokenizer = CharTokenizer(document)

        self.test_at_indices([0, 2], document, tokenizer)

    def compare_items(self, item1: dict[str, Tensor], item2: dict[str, Tensor]) -> bool:
        item1_list = list(item1)
        item2_list = list(item2)
        if len(item1_list) != len(item2_list):
            return False

    def test_at_indices(self, indices_to_test, document, tokenizer):
        max_block_size = 8
        dataset = TextDataset(max_block_size = max_block_size, tokenizer = tokenizer, document = document)

        for index in indices_to_test:
            actual_item = dataset.__getitem__(index)
            expected_item = {
                'features': Tensor(tokenizer.encode(document[index : index + max_block_size])),
                'labels': Tensor(tokenizer.encode(document[index + 1 : index + max_block_size + 1]))
            }
            self.assertEqual(self.compare_items(expected_item, actual_item))



if __name__ == '__main__':
    unittest.main()
