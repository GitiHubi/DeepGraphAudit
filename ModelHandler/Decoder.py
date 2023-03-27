# import pytorch libraries
from torch import nn

# import Linear Layer
from ModelHandler.LinearLayer import LinearLayer

# define decoder class
class Decoder(nn.Module):

    # define class constructor
    def __init__(self, hidden_size, output, bias):

        # call super class constructor
        super(Decoder, self).__init__()

        # init decoder linear layers
        self.layers = self.init_layers(hidden_size, bias=bias)

        # init intermediate decoder non-linearities
        self.activations = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: linear output
        if output == 'linear':

            # init linear output
            self.output = nn.Identity()

        # case: leaky relu output
        elif output == 'lrelu':

            # init leaky relu output
            self.output = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: tanh output
        elif output == 'tanh':

            # init tanh output
            self.output = nn.Tanh()

        # case: sigmoid output
        elif output == 'sigmoid':

            # init sigmoid output
            self.output = nn.Sigmoid()

    # init decoder layers
    def init_layers(self, layer_dimensions, bias):

        # init layers
        layers = []

        # iterate over layer dimensions
        for i in range(0, len(layer_dimensions) - 1):

            # create linear layer
            linearLayer = LinearLayer(layer_dimensions[i], layer_dimensions[i + 1], bias)

            # register linear layer
            self.add_module('layer_' + str(i), linearLayer)

            # collect linear layer
            layers.append(linearLayer)

        # return linear layers
        return layers

    # define decoder forward pass
    def forward(self, x):

        # iterate over distinct layers
        for i in range(0, len(self.layers)):

            # case: non-bottleneck layer
            if i < len(self.layers) - 1:

                # run forward pass through layer
                x = self.activations(self.layers[i](x))

            # case: bottleneck layer
            else:

                # run forward pass through layer
                x = self.output(self.layers[i](x))

        # return result
        return x
