'''
    Class implementing a transformer decoder block with LoRA finetuning.
'''

from torch import nn
from layers.mlp import MLP
from layers.masked_multihead_attention_with_lora_finetuning import MaskedMultiHeadAttentionWithLoRAFinetuning
from layers.transformer_decoder_block import TransformerDecoderBlock

class TransformerDecoderBlockWithLoRAFinetuning(TransformerDecoderBlock):
    '''
        Layer which applies a single Transformer Decoder block and supported LoRA
        finetuning (https://arxiv.org/abs/2106.09685).
    '''
    def __init__(self, mask: bool, input_dimension: int, num_heads: int, head_dimension: int, mode: str, lora_rank: int = 4):
        '''
            Initializer function.

            Args:
                1. mask - See documentation of parent class (TransformerDecoderBlock).
                2. input_dimension - See documentation of parent class (TransformerDecoderBlock).
                3. num_heads - See documentation of parent class (TransformerDecoderBlock).
                4. head_dimension - See documentation of parent class (TransformerDecoderBlock).
                5. mode - See documentation of parent class (TransformerDecoderBlock).
                6. lora_rank - See documentation of parent class (TransformerDecoderBlock).
        '''
        super().__init__(mask, input_dimension, num_heads, head_dimension)

        self.supported_mode_values: list[str] = ['pretraining', 'finetuning']
        assert mode in self.supported_mode_values, f'Unsupported value {mode}. Supported values are {self.supported_mode_values}'
        self.mode = mode

        if mode == 'finetuning':
            self.attention_layer = MaskedMultiHeadAttentionWithLoRAFinetuning(mask, num_heads, head_dimension, input_dimension, mode, lora_rank)
