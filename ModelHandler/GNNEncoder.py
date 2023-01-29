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

        # init weight parameter
        nn.init.xavier_uniform_(self.linear.weight)

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
    def __init__(self, input_dim, hidden_dim, embed_dim):

        # call super class constructor
        super(GNNEncoder, self).__init__()

        # init first graph convolutional layer
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim, bias=True)

        # init second graph convolutional layer
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=embed_dim, bias=True)

        # init VAE linear mu layer -> todo: fix double issue
        self.linear_mu = nn.Linear(embed_dim, embed_dim, bias=True).double()

        # init VAE linear sigma layer -> todo: fix double issue
        self.linear_sigma = nn.Linear(embed_dim, embed_dim, bias=True).double()

        # init ReLU non-linearity
        self.relu = nn.ReLU()

    # define encoder forward pass
    def forward(self, x, adj):

        # run first graph convolutional layer
        x = self.conv1(x, adj)

        # run ReLU non-linearity
        x = self.relu(x)

        # run second graph convolutional layer
        x = self.conv2(x, adj)

        # run ReLU non-linearity
        x = self.relu(x)

        # aggregate over all accounts into one-dimensional vector
        z = torch.sum(x, 1)

        # run VAE linear mu layer
        mu = self.linear_mu(z)

        # run VAE linear sigma layer
        sigma = self.linear_sigma(z)

        # return VAE mu and VAE sigma
        return z, mu, sigma