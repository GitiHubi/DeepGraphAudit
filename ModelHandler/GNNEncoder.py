# import pytorch libraries
import torch
from torch import nn
import torch.nn.init as init

# implement graph convolutional layer
class GraphConv(nn.Module):

    # define class constructor
    def __init__(self, input_dim, output_dim):

        # call super class constructor
        super(GraphConv, self).__init__()

        # init weight matrix of graph convolutional layer
        self.weight = nn.Parameter(torch.DoubleTensor(input_dim, output_dim))

    # define forward pass
    def forward(self, x, adj):

        # multiply adjacency matrix with identity matrix
        y = torch.matmul(adj, x)

        # multiply output with weight matrix of graph convolutional layer
        # performs the dimensionality reduction
        y = torch.matmul(y, self.weight)

        # return layer output
        return y

# define encoder class
class GNNEncoder(nn.Module):

    # define class constructor
    def __init__(self, input_dim, hidden_dim, embed_dim):

        # call super class constructor
        super(GNNEncoder, self).__init__()

        # init first graph convolutional layer
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)

        # init second graph convolutional layer
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=embed_dim)

        # init VAE linear mu layer -> todo: fix double issue
        self.linear_mu = nn.Linear(embed_dim, embed_dim).double()

        # init VAE linear sigma layer -> todo: fix double issue
        self.linear_sigma = nn.Linear(embed_dim, embed_dim).double()

        # init ReLU non-linearity
        self.relu = nn.ReLU()

        # iterate over encoder modules
        for m in self.modules():

            # case: graph convolutional layer
            if isinstance(m, GraphConv):

                # init weights using xavier uniform initialisation
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

        # define encoder forward pass
    def forward(self, x, adj):

        # run first graph convolutional layer
        x = self.conv1(x, adj)

        # run ReLU non-linearity
        x = self.relu(x)

        # run second graph convolutional layer
        x = self.conv2(x, adj)

        # aggregate into one-dimensional vector
        # using this to aggregate node info to graph info
        z = torch.sum(x, 1)

        # run VAE linear mu layer
        mu = self.linear_mu(z)

        # run VAE linear sigma layer
        sigma = self.linear_sigma(z)

        # return VAE mu and VAE sigma
        return z, mu, sigma