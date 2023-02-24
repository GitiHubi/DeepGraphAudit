# import pytorch libraries
from torch import nn

# define decoder class
class GNNDecoderStatic(nn.Module):

    # define class constructor
    def __init__(self, decoder_dim, bias):

        # call super class constructor
        super(GNNDecoderStatic, self).__init__()

        # init decoder architecture
        self.layers = self.init_layers(decoder_dim[:-2], bias=bias)
        self.reluLayer = nn.LeakyReLU(negative_slope=0.4, inplace=True)

        # init feature vector layer
        self.feat_linear = nn.Linear(decoder_dim[-3], decoder_dim[-2], bias=bias).double()

        # init fector layer weight parameters
        nn.init.xavier_uniform_(self.feat_linear.weight)

        # init adjacency matrix layer
        self.adj_linear = nn.Linear(decoder_dim[-3], decoder_dim[-1], bias=bias).double()

        # init adjacency matrix laye weight parameters
        nn.init.xavier_uniform_(self.adj_linear.weight)

        # init Sigmoid non-linearity
        self.sigmoid = nn.Sigmoid()

    # init decoder layers
    def init_layers(self, layer_dimensions, bias):

        # init layers
        layers = []

        # iterate over layer dimensions
        for i in range(0, len(layer_dimensions)-1):

            # create linear layer
            layer = self.LinearLayer(layer_dimensions[i], layer_dimensions[i + 1], bias).double()

            # collect linear layer
            layers.append(layer)

            # register linear layer
            self.add_module('linear_' + str(i), layer)

        # return layers
        return layers

    # init linear layer
    def LinearLayer(self, input_size, hidden_size, bias):

        # init linear layer
        linear = nn.Linear(input_size, hidden_size, bias=bias)

        # init linear layer parameters
        nn.init.xavier_uniform_(linear.weight)
        nn.init.constant_(linear.bias, 0.0)

        # return linear layer
        return linear

    # define decoder forward pass
    def forward(self, x):

        # iterate over distinct layers
        for i in range(0, len(self.layers)):

            # run forward pass through layer
            x = self.reluLayer(self.layers[i](x))

        # run final feature vector linear layer
        feat_matrix = self.sigmoid(self.feat_linear(x))

        # run final adjacency matrix linear layer
        adj_matrix = self.sigmoid(self.adj_linear(x))

        # return encoder output
        return feat_matrix, adj_matrix
