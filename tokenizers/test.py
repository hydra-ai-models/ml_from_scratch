'''
    Test the tokenizers implemented in the tokenizer directory.

    Run using
        python -m tokenizers.test
'''

from tokenizers.char_tokenizer import CharTokenizer

import unittest

class CharTokenizerTester(unittest.TestCase):
    def test_encoding_and_decoding(self):
        document = 'Hello, how are you?'

        tokenizer = CharTokenizer(document)
        self.assertEqual(document, tokenizer.decode(tokenizer.encode(document)))

if __name__ == '__main__':
    unittest.main()
