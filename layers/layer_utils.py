# Common utilities for layers.

import os

import torch.nn as nn
from typing import Optional

def num_parameters(model: nn.Module, output_path: Optional[str] = '') -> dict[str, int]:
    output_dict = {}
    total_parameters = 0
    total_trainable_parameters = 0
    for (name, parameter) in model.named_parameters():
        output_dict[f'layer_{name}'] = parameter.numel()
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            total_trainable_parameters += parameter.numel()
    output_dict['total_parameters'] = total_parameters
    output_dict['total_trainable_parameters'] = total_trainable_parameters

    if output_path:
        # Create directory of the output_path if it does not exist.
        dirname = os.path.dirname(output_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(output_path, 'w') as writer:
            writer.write(str(output_dict))
    return output_dict
