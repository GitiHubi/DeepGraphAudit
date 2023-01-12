import torch
import torch.nn as nn
import scipy.optimize

# import project libraries
import ModelHandler.GNNEncoder as GNNEncoder
import ModelHandler.GNNDecoder as GNNDecoder

# implement graph variational autoencoder neural network
class GNNAutoencoder(nn.Module):

    # define class constructor
    def __init__(self, input_dim, hidden_dim, embed_dim, output_dim, device='cpu'):

        # call super class constructor
        super(GNNAutoencoder, self).__init__()

        # set global input dimension
        self.input_dim = input_dim

        # init graph VAE encoder model
        self.encoder = GNNEncoder.GNNEncoder(input_dim, hidden_dim, embed_dim)

        # init graph VAE decoder model
        self.decoder = GNNDecoder.GNNDecoder(hidden_dim, embed_dim, output_dim)

        # init embedding dimension
        self.device = device

    # define graph VAE forward pass
    def forward(self, features, adj_matrices):

        # init recon features and adjacency matrices
        recon_features = torch.zeros_like(features)
        recon_adj_matrices = torch.zeros_like(adj_matrices)

        # run encoder forward pass
        z, mu, sigma = self.encoder(features, adj_matrices)

        # compute sigma sample
        sigma_sample = sigma.mul(0.5).exp_()

        # determine random sample of epsilon
        eps = torch.autograd.Variable(torch.randn(sigma.size())).to(self.device)

        # determine stochastic latent z sample
        z = eps * sigma_sample + mu

        # run decoder forward pass
        reconstructions = self.decoder(z)

        # iterate over reconstructions
        for i, reconstruction in enumerate(reconstructions):

            # reconstruct upper triangular matrix from output vector
            recon_adj_lower = self.reconstruct_adj_upper(output_vector=reconstruction, num_nodes=self.input_dim)

            # reconstruct full adjacency matrix from triangular matrix
            recon_adj_matrix = self.recover_full_adj_from_lower(recon_adj_lower)

            # determine reconstructed features
            recon_feature = torch.sum(recon_adj_matrix, axis=1)

            # collect reconstructed features and adjacency matrices
            recon_features[i, :, :] = recon_feature
            recon_adj_matrices[i, :, :] = recon_adj_matrix

        # return reconstructed features and adjacency matrix
        return z, mu, sigma, recon_features, recon_adj_matrices

    # reconstruct upper triangular matrix
    def reconstruct_adj_upper(self, output_vector, num_nodes):

        # init new max-node adjacency matrix
        adj_matrix = torch.zeros(num_nodes, num_nodes).double()

        # init new upper triangular boolean matrix
        adj_matrix_upper_triangular = torch.triu(torch.ones(num_nodes, num_nodes)) == 1

        # fill new max-node adjacency matrix
        adj_matrix[adj_matrix_upper_triangular] = output_vector

        # return adjacency matrix
        return adj_matrix

    # reconstruct full triangular matrix
    def recover_full_adj_from_lower(self, matrix):

        # determine matrix diagonal entries
        adj_diagonal = torch.diag(torch.diag(matrix, 0))

        # determine matrix transposed entries
        adj_transpose = torch.transpose(matrix, 0, 1)

        # determine full adjacency matrix
        adj_full = matrix + adj_transpose - adj_diagonal

        # return full adjacency matrix
        return adj_full
