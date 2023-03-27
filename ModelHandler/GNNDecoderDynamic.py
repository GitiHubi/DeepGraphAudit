# import pytorch libraries
from torch import nn

# import GNN Conv Layer
from ModelHandler.GraphConvLayer import GraphConvLayer

# define GNN decoder class
class GNNDecoderDynamic(nn.Module):

    # define class constructor
    def __init__(self, hidden_size, output, bias):

        # call super class constructor
        super(GNNDecoderDynamic, self).__init__()

        # init graph convolutional layers
        self.layers = self.init_layers(hidden_size, bias=bias)

        # init layer leaky relu non linear activations
        self.activations = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: linear output
        if output == 'linear':

            self.output = nn.Identity()

        # case: leaky relu output
        elif output == 'lrelu':

            self.output = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: tanh output
        elif output == 'tanh':

            self.output = nn.Tanh()

        # case: sigmoid output
        elif output == 'sigmoid':

            self.output = nn.Sigmoid()

    # init decoder layers
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

    # define decoder forward pass
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
                x = self.output(self.layers[i](x, adj))

        # return result
        return x