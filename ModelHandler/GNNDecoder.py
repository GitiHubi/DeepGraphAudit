# import pytorch libraries
import torch
from torch import nn

# define decoder class
class GNNDecoder(nn.Module):

    # define class constructor
    def __init__(self, hidden_dim, embed_dim, output_dim):

        # call super class constructor
        super(GNNDecoder, self).__init__()

        # init first linear layer -> todo: fix double issue
        self.linear1 = nn.Linear(embed_dim, hidden_dim).double()

        # init second linear layer -> todo: fix double issue
        self.linear2 = nn.Linear(hidden_dim, output_dim).double()

        # init ReLU non-linearity
        self.relu = nn.ReLU()

    # define decoder forward pass
    def forward(self, x):

        # run first linear layer
        x = self.linear1(x)

        # run ReLU non-linearity
        x = self.relu(x)

        # run second linear layer
        x = self.linear2(x)

        # run final Sigmoid non-linearity
        x = torch.sigmoid(x)

        # return encoder output
        return x
