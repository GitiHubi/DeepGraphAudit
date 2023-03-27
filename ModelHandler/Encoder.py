# import pytorch libraries
from torch import nn

# import Linear Layer
from ModelHandler.LinearLayer import LinearLayer

# define encoder class
class Encoder(nn.Module):

    # define class constructor
    def __init__(self, hidden_size, bottleneck, bias):

        # call super class constructor
        super(Encoder, self).__init__()

        # init decoder linear layers
        self.layers = self.init_layers(hidden_size, bias=True)

        # init intermediate decoder non-linearities
        self.activations = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: linear bottleneck
        if bottleneck == 'linear':

            # init linear bottleneck
            self.bottleneck = nn.Identity()

        # case: leaky relu bottleneck
        elif bottleneck == 'lrelu':

            # init leaky relu bottleneck
            self.bottleneck = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # case: tanh bottleneck
        elif bottleneck == 'tanh':

            # init tanh bottleneck
            self.bottleneck = nn.Tanh()

    # init encoder layers
    def init_layers(self, layer_dimensions, bias):

        # init layers
        layers = []

        # iterate over layer dimensions
        for i in range(0, len(layer_dimensions)-1):

            # create linear layer
            linearLayer = LinearLayer(layer_dimensions[i], layer_dimensions[i + 1], bias)

            # register linear layer
            self.add_module('layer_' + str(i), linearLayer)

            # collect linear layer
            layers.append(linearLayer)

        # return layers
        return layers

    # define encoder forward pass
    def forward(self, x):

        # iterate over distinct layers
        for i in range(0, len(self.layers)):

            # case: non-bottleneck layer
            if i < len(self.layers)-1:

                # run forward pass through layer
                x = self.activations(self.layers[i](x))

            # case: bottleneck layer
            else:

                # run forward pass through layer
                x = self.bottleneck(self.layers[i](x))

        # return result
        return x
