# Common utilities for layers.

import os
import yaml

import torch.nn as nn
from typing import Optional

def num_parameters(model: nn.Module, output_yaml_path: Optional[str] = '') -> dict[str, int]:
    output_dict = {}
    output_dict['all_parameters'] = {}
    output_dict['trainable_parameters'] = {}
    total_parameters = 0
    total_trainable_parameters = 0
    for (name, parameter) in model.named_parameters():
        output_dict['all_parameters'][name] = parameter.numel()
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            output_dict['trainable_parameters'][name] = parameter.numel()
            total_trainable_parameters += parameter.numel()
    output_dict['total_parameters'] = total_parameters
    output_dict['total_trainable_parameters'] = total_trainable_parameters

    if output_yaml_path:
        # Create directory of the output_path if it does not exist.
        dirname = os.path.dirname(output_yaml_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        writer = open(output_yaml_path, 'w')
        yaml.dump(output_dict, writer)
        writer.close()
    return output_dict
