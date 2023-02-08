import torch
import torch.nn as nn
import scipy.optimize

# import project libraries
import ModelHandler.GNNEncoder as GNNEncoder
import ModelHandler.GNNDecoder as GNNDecoder

# implement graph variational autoencoder neural network
class GNNAutoencoder(nn.Module):

    # define class constructor
    def __init__(self, statistics, feat_embed_dim, encoder_dim, bottleneck, decoder_dim, bias=True, device='cpu'):

        # call super class constructor
        super(GNNAutoencoder, self).__init__()

        # init dataset statistics
        self.statistics = statistics

        # init VAE feature embedding layers
        self.feat_embeddings = self.init_embedding_layers(feat_embed_dim)

        # init VAE feature embeddin non-linearity
        self.feat_sigmoid = nn.Sigmoid()

        # init graph VAE encoder model
        self.encoder = GNNEncoder.GNNEncoder(encoder_dim, bottleneck, bias)

        # init graph VAE decoder model
        self.decoder = GNNDecoder.GNNDecoder(decoder_dim, bias)

        # init embedding dimension
        self.device = device

    # define graph VAE forward pass
    def forward(self, feat_matrices, adj_matrices):

        # run encoder forward pass
        z, mu, sigma = self.encoder(feat_matrices, adj_matrices)

        # compute sigma sample
        # sigma_sample = sigma.mul(0.5).exp_()

        # determine random sample of epsilon
        # eps = torch.autograd.Variable(torch.randn(sigma.size())).to(self.device)

        # determine stochastic latent z sample
        # z = eps * sigma_sample + mu

        # run decoder forward pass
        rec_feat_matrices, rec_adj_matrices = self.decoder(z)

        # reformat reconstructed feature matrices
        rec_feat_matrices = rec_feat_matrices.reshape_as(feat_matrices)

        # reformat reconstructed adjacency matrices
        rec_adj_matrices = rec_adj_matrices.reshape_as(adj_matrices)

        # return reconstructed features and adjacency matrix
        return z, mu, sigma, rec_feat_matrices, rec_adj_matrices

    def init_embedding_layers(self, embedding_dim):

        # init embedding layers
        embedding_layers = []

        # iterate over je features
        for i, feature in enumerate(self.statistics['je_features']):

            # case: je header feature
            if feature in self.statistics['je_header_features']:

                # init categorical embedding layer
                feature_embedding_layer = self.init_feature_layer_categorical(i, feature, embedding_dim)

            # case: je segment categorial feature
            elif feature in self.statistics['je_segment_features_categorical']:

                # init categorical embedding layer
                feature_embedding_layer = self.init_feature_layer_categorical(i, feature, embedding_dim)

            # case: je segment numerical feature
            elif feature in self.statistics['je_segment_features_numerical']:

                # init numerical embedding layer
                feature_embedding_layer = self.init_feature_layer_numerical(i, feature, embedding_dim)

            # collect embedding layer
            embedding_layers.append(feature_embedding_layer)

        # return the embedding layers
        return embedding_layers

    def init_feature_layer_categorical(self, count, feature, embedding_dim):

        # determine attribute dimensionality
        attribute_dim = self.statistics['{}_dim'.format(feature)]

        # init categorical embedding layer
        feature_embedding_layer = nn.Embedding(attribute_dim, embedding_dim).double()

        # register categorical embedding layer
        self.add_module('embedding_cat_{}_{}'.format(str(count), str(feature)), feature_embedding_layer)

        # init embedding layer parameters
        nn.init.xavier_uniform_(feature_embedding_layer.weight)

        # return categorical feature embedding layers
        return feature_embedding_layer

    def init_feature_layer_numerical(self, count, feature, embedding_dim):

        # init numerical embedding layer
        feature_embedding_layer = nn.Linear(1, embedding_dim).double()

        # register numerical embedding layer
        self.add_module('embedding_num_{}_{}'.format(str(count), str(feature)), feature_embedding_layer)

        # init embedding layer parameters
        nn.init.xavier_uniform_(feature_embedding_layer.weight)

        # return numerical feature embedding layers
        return feature_embedding_layer

    def embedd_features_batch(self, feat_matrices):

        # iterate over je features
        for i, feature in enumerate(self.statistics['je_features']):

            # case: je header feature
            if feature in self.statistics['je_header_features']:

                # determine values of current feature
                feat_matrices_values = feat_matrices[:, :, i].type(torch.LongTensor).to(self.device)

            # case: je segment categorial feature
            elif feature in self.statistics['je_segment_features_categorical']:

                # determine values of current feature
                feat_matrices_values = feat_matrices[:, :, i].type(torch.LongTensor).to(self.device)

            # case: je segment numerical feature
            elif feature in self.statistics['je_segment_features_numerical']:

                # determine values of current feature
                feat_matrices_values = feat_matrices[:, :, i].unsqueeze(2).to(self.device)

            # determine linear embedding of current feature
            feat_matrices_embedding = self.feat_embeddings[i](feat_matrices_values)

            # determine non-linear embedding of current feature
            feat_matrices_embedding = self.feat_sigmoid(feat_matrices_embedding)

            # case: initial feature
            if i == 0:

                # determine embedded features
                feat_matrices_embeddings = feat_matrices_embedding

            # case: non-initial dimension
            else:

                # determine embedded features
                feat_matrices_embeddings = torch.cat((feat_matrices_embeddings, feat_matrices_embedding), dim=2)

        # return embedded features
        return feat_matrices_embeddings

    # reformat feat matrices
    def reconstruct_feat_matrices(self, feat_matrices, rec_feat_matrices, xdim, ydim):

        # init recon features and adjacency matrices
        rec_feat_matrices_formatted = torch.zeros_like(feat_matrices)

        # iterate over reconstructions
        for i, rec_feat_matrix in enumerate(rec_feat_matrices):

            # reconstruct upper triangular matrix from output vector
            rec_feat_lower = self.recover_upper(rec_feat_matrix, xdim, ydim)

            # reconstruct full adjacency matrix from triangular matrix
            rec_feat_matrix = self.recover_full(rec_feat_lower)

            # collect reconstructed adjacency matrices
            rec_feat_matrices_formatted[i, :, :] = rec_feat_matrix

        # return reformated adjacency matrix
        return rec_feat_matrices_formatted

    # reformat adjacency matrices
    def reconstruct_adj_matrices(self, adj_matrices, rec_adj_matrices, xdim, ydim):

        # init recon adjacency matrices
        rec_adj_matrices_formatted = torch.zeros_like(adj_matrices)

        # iterate over reconstructions
        for i, rec_adj_matrix in enumerate(rec_adj_matrices):

            # reconstruct upper triangular matrix from output vector
            rec_adj_lower = self.recover_upper(rec_adj_matrix, xdim, ydim)

            # reconstruct full adjacency matrix from triangular matrix
            rec_adj_matrix = self.recover_full(rec_adj_lower)

            # collect reconstructed adjacency matrices
            rec_adj_matrices_formatted[i, :, :] = rec_adj_matrix

        # return reformated adjacency matrix
        return rec_adj_matrices_formatted

    # reconstruct upper triangular matrix
    def recover_upper(self, output_vector, xdim, ydim):

        # init new max-node adjacency matrix
        adj_matrix = torch.zeros(xdim, ydim).double().to(self.device)

        # init new upper triangular boolean matrix
        adj_matrix_upper_triangular = torch.triu(torch.ones(xdim, ydim)) == 1

        # fill new max-node adjacency matrix
        adj_matrix[adj_matrix_upper_triangular] = output_vector

        # return adjacency matrix
        return adj_matrix

    # reconstruct full triangular matrix
    def recover_full(self, matrix):

        # determine matrix diagonal entries
        adj_diagonal = torch.diag(torch.diag(matrix, 0))

        # determine matrix transposed entries
        adj_transpose = torch.transpose(matrix, 0, 1)

        # determine full adjacency matrix
        adj_full = matrix + adj_transpose - adj_diagonal

        # return full adjacency matrix
        return adj_full
