# import pytorch libraries
from torch.utils import data

# define accounting dataset
class AccountingGNNDatasetDynamic(data.Dataset):

    # define class constructor
    def __init__(self, adj_matrices: list, feat_matrices: list):

        # init adjacency matrix
        self.adj_matrices = adj_matrices

        # init node features
        self.feat_matrices = feat_matrices

    # define length function
    def __len__(self):

        # determine number of adjacency matrices
        no_adjacency_matrix = len(self.adj_matrices)

        # return number of adjacency matrices
        return no_adjacency_matrix

    # define get item function
    def __getitem__(self, idx):

        # determine batch adjacency matrix
        adj_matrix = self.adj_matrices[idx]

        # determine batch features
        features = self.feat_matrices[idx]

        # return adjacency matrix and features
        return adj_matrix, features