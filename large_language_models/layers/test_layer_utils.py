'''
    Test the methods provided in layers.layer_utils.

    Run using
        python -m layers.test_layer_utils
'''


from layers import layer_utils
import torch.nn as nn
import unittest

class LayerUtilsTester(unittest.TestCase):
    def test_get_parameters(self):
        num_in_features = 6
        num_out_features = 5
        linear_layer = nn.Linear(num_in_features, num_out_features, bias = False)
        self.assertEqual(num_in_features * num_out_features, layer_utils.num_parameters(linear_layer)['total_trainable_parameters'])


if __name__ == '__main__':
    unittest.main()
