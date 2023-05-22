''' Class implementing the GPT language model from scratch with LoRA finetuning.
'''

import os
import torch, torch.nn as nn
from typing import Optional

from layers.transformer_decoder_block_with_lora_finetuning import TransformerDecoderBlockWithLoRAFinetuning
from layers.gpt import GPT

class GPTWithLoRAFinetuning(GPT):
    '''
        Layer corresponding to the Generative Pretrained Transformer (GPT) model and
        supports LoRA finetuning (https://arxiv.org/abs/2106.09685).
    '''
    def __init__(self, num_decoder_blocks, vocabulary_size, embedding_dimension, num_heads, head_dimension, max_block_size, mode: str, pretrained_model_path: Optional[str], lora_rank: int = 4):
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
        super().__init__(num_decoder_blocks, vocabulary_size, embedding_dimension, num_heads, head_dimension, max_block_size)
        self.supported_mode_values: list[str] = ['pretraining', 'finetuning']
        assert mode in self.supported_mode_values, f'Unsupported value {mode}. Supported values are {self.supported_mode_values}'
        self.mode = mode

        if mode == 'finetuning':
            self.transformer_decoders = nn.ModuleList([TransformerDecoderBlockWithLoRAFinetuning(True, embedding_dimension, num_heads, head_dimension, mode, lora_rank) for block_index in range(num_decoder_blocks)])

            if not os.path.exists(pretrained_model_path):
                ValueError(f'Pretrained model does not exist at {pretrained_model_path}. Please update pretrained_model_path and run again.')
            # Note that strict is set to False, since model contains additional LoRA matrices, while pretrained model most likely wont
            # contain them.
            self.load_state_dict(torch.load(pretrained_model_path), strict = False)

            # Making all parameters except the LoRA ones non trainable. Note that this  implementation assumes
            # that LoRA layers have `lora` substring in the layer names.
            for name, param in self.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
                # Initialize the LoRA B matrix with zero values.
                if 'lora_B' in name:
                    torch.nn.init.zeros_(param)
