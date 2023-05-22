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
    def __init__(self, mask: bool, input_dimension: int, num_heads: int, head_dimension: int, support_lora_finetuning: bool = False):
        super().__init__()
        self.attention_layer = MaskedMultiHeadAttention(mask, num_heads, head_dimension, input_dimension)
        self.mlp_layer = MLP(input_dimension, 2 * input_dimension, input_dimension)
        self.layer_norm = nn.LayerNorm(input_dimension)
        self.support_lora_finetuning = support_lora_finetuning


    def forward(self, x):
        attention_output = self.attention_layer(x, x, x)
        normalized_output = self.layer_norm(attention_output)
        residual_output = x + normalized_output
        mlp_output = self.mlp_layer(residual_output)
        return mlp_output

class TransformerDecoderBlockWithLoRAFinetuning(TransformerDecoderBlock):
    '''
        Layer which applies a single Transformer Decoder block and supported LoRA
        finetuning (https://arxiv.org/abs/2106.09685).
    '''
    def __init__(self, mask: bool, input_dimension: int, num_heads: int, head_dimension: int, mode: str, lora_rank: int = 4):
        '''
            Initializer.
                Args:
                    6. lora_rank - Rank to be used for the LoRA matrices A and B. Default value is 4.
        '''
        super().__init__(mask, input_dimension, num_heads, head_dimension)

        self.supported_mode_values: list[str] = ['pretraining', 'finetuning']
        assert mode in supported_mode_values, f'Unsupported value {mode}. Supported values are {self.supported_mode_values}'
        self.mode = mode

        if mode == 'finetuning':
            self.attention_layer = MaskedMultiHeadAttentionWithLoRAFinetuning(mask, num_heads, head_dimension, input_dimension, mode, lora_rank)


    def forward(self, x):
        attention_output = self.attention_layer(x, x, x)
        normalized_output = self.layer_norm(attention_output)
        residual_output = x + normalized_output
        mlp_output = self.mlp_layer(residual_output)
        return mlp_output
