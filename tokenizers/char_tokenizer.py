# Implementations of character tokenizer, which uses each character as a token.

from tokenizers.base_tokenizer import BaseTokenizer

class CharTokenizer(BaseTokenizer):
    ''' Concrete tokenizer class which does character level tokenization. Vocabulary for tokenization is created dynamically based on the document passed during initialization. Every unique character in this document becomes a new token. Currently unseen tokens during encoding are not handled, but this can be handled in the future simply by adding a single new token.'''

    def __init__(self, document: str)-> None:
        ''' Initializes CharTokenizer.

            Args:
                1. document - String of text to use to dynamically create the vocabulary for tokenization. Current implementation assumes that all tokens to be encoded and decoded in the future will be present in this document. This restriction will be removed in future implementations.

            Returns:
                No return values.
        '''
        # Since this is a character level tokenizer, vocabulary is obtained by finding unique characters in the document using the set(), and converting it to a list with the list(). Sorting is done just to ensure a repeatable order for debugging and cleanliness.
        self._vocabulary = sorted(list(set(document)))

        # Vocabulary length is cached for speed up.
        self._vocabulary_length = len(self._vocabulary)

        self._token_to_index_map = {token:index for (index, token) in enumerate(self._vocabulary)}
        self._index_to_token_map = {index: token for (index, token) in enumerate(self._vocabulary)}

    def encode(self, text: str) -> list[int]:
        ''' Takes a string of text as input, breaks it into tokens, and returns the list of token indices in the input. This method should be implemented by concrete subclasses of TokenizerBase.

            Args:
                1. text - String of text to be tokenized.

            Returns:
                List of tokens corresponding to the input text.
        '''
        return [self._token_to_index_map[token] for token in text]


    def decode(self, token_indices: list[int]) -> str:
        ''' Takes a list of token indices as input, converts the corresponding indices to token strings, and returns the concatenated string as output. This method should be implemented by concrete subclasses of TokenizerBase.

            Args:
                1. token_indices - List of token indices to be decoded.

            Returns:
                Text obtained by converting each token index in input to the corresponding token string, and concatenating them.
        '''
        return ''.join([self._index_to_token_map[index] for index in token_indices])

    def vocabulary(self) -> list[str]:
        ''' Returns a list of all the token strings in the vocabulary used for tokenization.

            Args:
                None

            Returns:
                List of token strings in the vocabulary used for tokenization.
        '''
        return self._vocabulary

    def vocabulary_length(self) -> int:
        ''' Returns the length of the vocabulary.

            Args:
                None

            Returns:
                Length of the vocabulary used for tokenization.
        '''
        return self._vocabulary_length
