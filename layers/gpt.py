''' Class implementing the GPT language model from scratch.
'''

import torch, torch.nn as nn
from typing import Optional

from layers.transformer_decoder_block import TransformerDecoderBlock


class GPT(nn.Module):
    '''
        Layer corresponding to the Generative Pretrained Transformer (GPT) model.
    '''
    def __init__(self, num_decoder_blocks, vocabulary_size, embedding_dimension, num_heads, head_dimension, max_block_size):
        super().__init__()
        self.max_block_size = max_block_size
        self.token_embedding_layer = nn.Embedding(vocabulary_size, embedding_dimension)
        self.positional_encoding_layer = nn.Embedding(max_block_size, embedding_dimension)
        self.transformer_decoders = nn.ModuleList([TransformerDecoderBlock(True, embedding_dimension, num_heads, head_dimension) for block_index in range(num_decoder_blocks)])
        self.head_layer = nn.Linear(embedding_dimension, vocabulary_size)


    # Notation - B - batch size, T - block size, Re - embedding dimension.
    def forward(self, x, y: Optional[torch.Tensor] = None): # x - (B, T)
        (batch_size, current_block_size) = x.shape
        token_embeddings = self.token_embedding_layer(x) # (B, T, Re)
        positional_tensor = torch.arange(start = 0, end = current_block_size, dtype = torch.long)
        positional_embeddings = self.positional_encoding_layer(positional_tensor) # (T, Re)
        transformer_input = token_embeddings + positional_embeddings # (B, T, Re)
        transformer_output = transformer_input
        for decoder in self.transformer_decoders:
            transformer_output = decoder(transformer_output) # (B, T, Re)
        predictions = self.head_layer(transformer_output) # (B, T, vocabulary_size)

        if y != None:
            unrolled_predictions = predictions.view(batch_size * current_block_size, -1)
            unrolled_labels = y.view(batch_size * current_block_size)
            loss = nn.functional.cross_entropy(unrolled_predictions, unrolled_labels)
        else:
            loss = None

        return (predictions, loss)

    def generate(self, initial_tokens, num_tokens_to_generate):
        generated_tokens = initial_tokens
        for generated_token_index in range(num_tokens_to_generate):
            network_input = generated_tokens
            (_, current_block_size) = network_input.shape
            if current_block_size > self.max_block_size:
                network_input = network_input[:, current_block_size - self.max_block_size:]
            (network_output, _) = self(network_input) # (1, T, Vocab)
            token_probabilities = nn.functional.softmax(network_output[:, -1, :], dim=1) # (B, Vocab)
            current_generated_token = torch.multinomial(token_probabilities, 1)
            generated_tokens = torch.cat([generated_tokens, current_generated_token], dim = 1)
        return generated_tokens

class GPTWithLoRAFinetuning(GPT):
    '''
        Layer corresponding to the Generative Pretrained Transformer (GPT) model and
        supports LoRA finetuning (https://arxiv.org/abs/2106.09685).
    '''
    def __init__(self, num_decoder_blocks, vocabulary_size, embedding_dimension, num_heads, head_dimension, max_block_size, mode: str, lora_rank: int = 4):
        '''
            Initializer function.

            Args:
                1. num_decoder_blocks - See documentation of parent class (GPT).
                2. vocabulary_size - See documentation of parent class (GPT).
                3. embedding_dimension - See documentation of parent class (GPT).
                4. num_heads - See documentation of parent class (GPT).
                5. head_dimension - See documentation of parent class (GPT).
                6. max_block_size - See documentation of parent class (GPT).
                7. mode - Whether doing pretraining or finetuning. Supported values are ['pretraining', 'finetuning']
                8. lora_rank - Rank to be used for the LoRA matrices A and B. Default value is 4.
        '''
            super().init(num_decoder_blocks, vocabulary_size, embedding_dimension, num_heads, head_dimension, max_block_size)
            self.supported_mode_values: list[str] = ['pretraining', 'finetuning']
            assert mode in supported_mode_values, f'Unsupported value {mode}. Supported values are {self.supported_mode_values}'
            self.mode = mode

            if mode == 'finetuning':
                self.transformer_decoders = nn.ModuleList([TransformerDecoderBlockWithLoRAFinetuning(True, embedding_dimension, num_heads, head_dimension, mode, lora_rank) for block_index in range(num_decoder_blocks)])
