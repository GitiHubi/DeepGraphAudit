# import pytorch libraries
import torch
from torch import nn

# import project library
from ModelHandler.GraphConvLayer import GraphConvLayer

# define encoder class
class GNNDecoderDynamic(nn.Module):

    # define class constructor
    def __init__(self, hidden_size, bottleneck, bias):

        # call super class constructor
        super(GNNDecoderDynamic, self).__init__()

        # init graph convolutional layers
        self.layers = self.init_layers(hidden_size, bias=bias)

        # init layer leaky relu non linear activations
        self.activations = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: linear bottleneck
        if bottleneck == 'linear':

            self.bottleneck = nn.Identity()

        # case: leaky relu bottleneck
        elif bottleneck == 'lrelu':

            self.bottleneck = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: tanh bottleneck
        elif bottleneck == 'tanh':

            self.bottleneck = nn.Tanh()

        # case: sigmoid bottleneck
        elif bottleneck == 'sigmoid':

            self.bottleneck = nn.Sigmoid()

    # init encoder layers
    def init_layers(self, layer_dimensions, bias):

        # init graph convolutional layers
        layers = []

        # iterate over layer dimensions
        for i in range(0, len(layer_dimensions)-1):

            # create init graph convolutional layer
            layer = GraphConvLayer(input_dim=layer_dimensions[i], output_dim=layer_dimensions[i + 1], bias=bias)

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

        # return VAE mu and VAE sigma
        return x