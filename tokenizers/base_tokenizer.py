# Base Tokenizer class with a HuggingFace like API.

from abc import ABC, abstractmethod

class BaseTokenizer(ABC):
    ''' Abstract class providing a Hugging Face like tokenizer API. Concrete subclasses should implement the abstract methods in this class. '''

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ''' Abstract method which takes a string of text as input, breaks it into tokens, and returns the list of token indices in the input. This method should be implemented by concrete subclasses of this class.

            Args:
                1. text - String of text to be tokenized.

            Returns:
                List of tokens corresponding to the input text.
        '''
        pass

    @abstractmethod
    def decode(self, token_indices: list[str]) -> str:
        ''' Abstract method which takes a list of token indices as input, converts the corresponding indices to token strings, and returns the concatenated string as output. This method should be implemented by concrete subclasses of this class.

            Args:
                1. token_indices - List of token indices to be decoded.

            Returns:
                Text obtained by converting each token index in input to the corresponding token string, and concatenating them.
        '''
        pass

    @abstractmethod
    def vocabulary(self) -> list[str]:
        ''' Abstract method that returns a list of all the token strings in the vocabulary used for tokenization. This method should be implemented by concrete subclasses of this class.

            Args:
                None

            Returns:
                List of token strings in the vocabulary used for tokenization.
        '''
        pass

    @abstractmethod
    def vocabulary_length(self) -> int:
        ''' Abstract method that returns the length of the vocabulary. This method should be implemented by concrete subclasses of this class.

            Args:
                None

            Returns:
                Length of the vocabulary used for tokenization.
        '''
        pass
