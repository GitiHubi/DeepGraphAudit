# import python libraries
import math

# import pytorch libraries
import torch
from torch import nn
import torch.nn.init as init

# implement graph convolutional layer
class GraphConv(nn.Module):

    # define class constructor
    def __init__(self, input_dim, output_dim, bias=True):

        # call super class constructor
        super(GraphConv, self).__init__()

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

# define encoder class
class GNNEncoder(nn.Module):

    # define class constructor
    def __init__(self, hidden_size, bottleneck, bias):

        # call super class constructor
        super(GNNEncoder, self).__init__()

        # init graph convolutional layers
        self.layers = self.init_layers(hidden_size, bias=bias)

        # init layer leaky relu non linear activations
        self.activations = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # init VAE linear mu layer -> todo: fix double issue, think about keeping here
        self.linear_mu = nn.Linear(hidden_size[-1], hidden_size[-1], bias=True).double()

        # init VAE linear sigma layer -> todo: fix double issue, think about keeping here
        self.linear_sigma = nn.Linear(hidden_size[-1], hidden_size[-1], bias=True).double()

        # case: linear bottleneck
        if bottleneck == 'linear':

            self.bottleneck = nn.Identity()


        # case: leaky relu bottleneck
        elif bottleneck == 'lrelu':

            self.bottleneck = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: tanh bottleneck
        elif bottleneck == 'tanh':

            self.bottleneck = nn.Tanh()

    # init encoder layers
    def init_layers(self, layer_dimensions, bias):

        # init graph convolutional layers
        layers = []

        # iterate over layer dimensions
        for i in range(0, len(layer_dimensions)-1):

            # create init graph convolutional layer
            layer = GraphConv(input_dim=layer_dimensions[i], output_dim=layer_dimensions[i + 1], bias=bias)

            # register linear graph convolutional layer
            self.add_module('layer_' + str(i), layer)

            # collect graph convolutional layer
            layers.append(layer)

        # return graph convolutional layers
        return layers

    # define encoder forward pass
    def forward(self, x, adj):

        # iterate over distinct graph convolutional layers
        for i in range(0, len(self.layers)):

            # case: non-bottleneck layer
            if i < len(self.layers)-1:

                # run forward pass through layer
                x = self.activations(self.layers[i](x, adj))

            # case: bottleneck layer
            else:

                # run forward pass through layer
                x = self.bottleneck(self.layers[i](x, adj))

        # aggregate over all accounts into one-dimensional vector
        z = torch.sum(x, 1)

        # run VAE linear mu layer
        mu = self.linear_mu(z)

        # run VAE linear sigma layer
        sigma = self.linear_sigma(z)

        # return VAE mu and VAE sigma
        return z, mu, sigma