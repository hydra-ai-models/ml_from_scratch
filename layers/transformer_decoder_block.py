'''
    Class implementing a transformer decoder block.
'''

from torch import nn
from layers.mlp import MLP
from layers.masked_multihead_attention import MaskedMultiHeadAttention

class TransformerDecoderBlock(nn.Module):
    '''
        Layer which applies a single Transformer Decoder block.
    '''
    def __init__(self, mask: bool, input_dimension: int, num_heads: int, head_dimension: int):
        super().__init__()
        self.attention_layer = MaskedMultiHeadAttention(mask, num_heads, head_dimension, input_dimension)
        self.mlp_layer = MLP(input_dimension, 2 * input_dimension, input_dimension)
        self.layer_norm = nn.LayerNorm(input_dimension)


    def forward(self, x):
        attention_output = self.attention_layer(x, x, x)
        normalized_output = self.layer_norm(attention_output)
        residual_output = x + normalized_output
        mlp_output = self.mlp_layer(residual_output)
        return mlp_output
