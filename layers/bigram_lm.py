from torch.nn import Module
from torch.nn.functional import softmax
import torch

class BigramLanguageModel(Module):
    '''
        Simple Bigram language model which learns an embedding to predict the next token based on
        the current character.
    '''
    def __init__(self, vocabulary_size: int):
        super().__init__()
        # Embedding layer which predicts the logits of the next character as the embedding vector of the current character.
        self.embedding = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, x):
        return self.embedding(x)

    def generate(self, initial_tokens, num_tokens_to_generate: int):
        '''
            Generate tokens using the model.

            Args:
                1. initial_tokens - (batch_size, number_of_initial_tokens) tensor containing the initial tokens in the sequence.
                2. num_tokens_to_generate - Number of tokens to generate for each element in the batch.

            Returns:
                A (batch_size, number_of_initial_tokens + num_tokens_to_generate) tensor containing the initial tokens and the generated tokens for each element in the batch.
        '''
        generated_tokens = initial_tokens # generated_tokens has shape (batch_size, number_of_initial_tokens)
        for i in range(num_tokens_to_generate):
            # Take the last token for each element in the batch, and apply the model to generate the logits of the next token.
            logits = self(generated_tokens[:, -1])

            #next_token_logits = logits[:, -1, :] # (B, C)
            # Compute probability of next character by applying softmax for each element in the batch.
            next_token_probability = softmax(next_token_logits, dim = 1)

            # Generate next character based on the predicted probability distribution.
            next_token = torch.multinomial(next_token_probability, 1)

            generated_tokens = torch.cat((generated_tokens, next_token), dim = 1)
        return generated_tokens
