''' Class implementing the GPT language model from scratch.

    In the future. layers implemented in this file for multihead masked attention and
    multilayer perceptrons can be moved to their own files for clarity.
'''

import torch, torch.nn as nn
from typing import Optional

class MLP(nn.Module):
    '''
        Multilayer perceptron consisting of a linear layer, a non linear activation (ReLU) and a second linear layer.
    '''
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int):
        '''
            Args:
                1. input_dimension - Dimension of the input to the layer.
                2. hidden_dimension - Dimension of the output of the first linear layer.
                3. output_dimension - Dimension of the output of the MLP module.
        '''
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, output_dimension))

    def forward(self, x):
        return self.layer(x)

class MultiHeadMaskedAttention(nn.Module):
    '''
        Multihead attention layer which can perform causal or non causal attention.
    '''
    def __init__(self, mask: bool, num_heads: int, head_dimension: int, input_dimension: int):
        '''
            Args:
                1. mask - Whether to apply causal attention or not.
                2. num_heads - Number of heads to use when computing multihead attention.
                3. head_dimension - Dimension of the head embedding.
                4. input_dimension - dimension of the input embedding.
        '''
        super().__init__()
        self.mask: bool = mask
        self.num_heads = num_heads
        self.head_dimension = head_dimension
        self.Wq = nn.Linear(input_dimension, num_heads * head_dimension) # (Re, nh * Rh)
        self.Wk = nn.Linear(input_dimension, num_heads * head_dimension)
        self.Wv = nn.Linear(input_dimension, num_heads * head_dimension)
        self.head_merge_layer = nn.Linear(num_heads*head_dimension, input_dimension)

    def forward(self, queries, keys, values):
        # queries - (B, T, input_dimension)
        (batch_size, block_size, _) = queries.shape
        projected_queries = self.Wq(queries) # (B, T, nh*Rh)
        reshaped_queries = projected_queries.view(-1, block_size, self.num_heads, self.head_dimension) # (B, T, nh, Rh)
        transposed_queries = reshaped_queries.transpose(1, 2) # (B, nh, T, Rh)\

        projected_keys = self.Wk(keys) # (B, T, nh*Rh)
        reshaped_keys = projected_keys.view(-1, block_size, self.num_heads, self.head_dimension)
        transposed_keys = reshaped_keys.transpose(1, 2) # (B, nh, T, Rh)

        projected_values = self.Wv(values) # (B, T, nh*Rh)
        reshaped_values = projected_values.view(-1, block_size, self.num_heads, self.head_dimension)
        transposed_values = reshaped_values.transpose(1,2) # (B, nh, T, Rh)


        attention = transposed_queries @ (transposed_keys.transpose(2, 3)) # (B, nh, T, T)
        if self.mask:
            tril_mat = torch.tril(torch.ones(block_size, block_size))
            attention.masked_fill(tril_mat == 0, -torch.inf)
        attention = nn.functional.softmax(attention, dim = 2) / self.head_dimension**(0.5) # (B, nh, T, T)
        attention_output_multihead = attention @ transposed_values # (B, nh, T, Rh)

        # Combining multiple heads.
        transposed_attention_output_multihead = attention_output_multihead.transpose(1, 2) # (B, T, nh, Rh)
        rehaped_attention_output_multihead = transposed_attention_output_multihead.contiguous().view(-1, block_size, self.num_heads * self.head_dimension)
        attention_output = self.head_merge_layer(rehaped_attention_output_multihead)
        return attention_output


class TransformerDecoderBlock(nn.Module):
    '''
        Layer which applies a single Transformer Decoder block.
    '''
    def __init__(self, mask: bool, input_dimension: int, num_heads: int, head_dimension: int):
        super().__init__()
        self.attention_layer = MultiHeadMaskedAttention(mask, num_heads, head_dimension, input_dimension)
        self.mlp_layer = MLP(input_dimension, 2 * input_dimension, input_dimension)
        self.layer_norm = nn.LayerNorm(input_dimension)


    def forward(self, x):
        attention_output = self.attention_layer(x, x, x)
        normalized_output = self.layer_norm(attention_output)
        residual_output = x + normalized_output
        mlp_output = self.mlp_layer(residual_output)
        return mlp_output


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
