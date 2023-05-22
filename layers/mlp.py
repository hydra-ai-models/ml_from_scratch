''' Class implementing multilayer perceptron.
'''

from torch import nn

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
