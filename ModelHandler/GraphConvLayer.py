# import pytorch libraries
import torch
from torch import nn

# implement graph convolutional layer
class GraphConvLayer(nn.Module):

    # define class constructor
    def __init__(self, input_dim, output_dim, bias=True):

        # call super class constructor
        super(GraphConvLayer, self).__init__()

        # init linear layer weight matrix of graph convolutional layer
        self.linear = nn.Linear(input_dim, output_dim, bias=bias).double()

        # init weight parameters
        nn.init.xavier_uniform_(self.linear.weight)

        # init bias parameters
        nn.init.constant_(self.linear.bias, 0.0)

    # define forward pass
    def forward(self, x, adj):

        # multiply adjacency matrix with features
        y = torch.matmul(adj, x)

        # perform the dimensionality reduction
        y = self.linear(y)

        # return layer output
        return y