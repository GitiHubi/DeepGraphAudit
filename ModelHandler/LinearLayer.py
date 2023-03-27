# import pytorch libraries
import torch
from torch import nn

# implement linear layer
class LinearLayer(nn.Module):

    # define class constructor
    def __init__(self, input_size, hidden_size, bias=True):

        # call super class constructor
        super(LinearLayer, self).__init__()

        # init linear layer
        self.linear = nn.Linear(input_size, hidden_size, bias=bias).double()

        # init weight parameters
        nn.init.xavier_uniform_(self.linear.weight)

        # case: bias enabled
        if bias:

            # init bias parameters
            nn.init.constant_(self.linear.bias, 0.0)

    # define forward pass
    def forward(self, x):

        # perform the dimensionality reduction
        x = self.linear(x)

        # return layer output
        return x