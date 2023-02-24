import torch
import torch.nn as nn

# import project libraries
import ModelHandler.GNNEncoder as GNNEncoder
import ModelHandler.GNNDecoderDynamic as GNNDecoderDynamic

# GNNAutoencoderDynamic class
class GNNAutoencoderDynamic(nn.Module):

    # define class constructor
    def __init__(self, statistics, feat_embed_dim, encoder_dim, encoder_bottleneck, decoder_dim, decoder_bottleneck, bias=True, device='cpu'):

        # call super class constructor
        super(GNNAutoencoderDynamic, self).__init__()

        # init dataset statistics
        self.statistics = statistics

        # init VAE feature embedding layers
        self.feat_embeddings = self.init_embedding_layers(feat_embed_dim)

        # init VAE feature embeddin non-linearity
        self.adj_sigmoid = nn.Sigmoid()

        # init graph VAE encoder model
        self.encoder = GNNEncoder.GNNEncoder(encoder_dim, bottleneck=encoder_bottleneck, bias=bias)

        # init graph VAE decoder model
        self.decoder = GNNDecoderDynamic.GNNDecoderDynamic(decoder_dim, bottleneck=decoder_bottleneck, bias=bias)

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

        # scale up latent representation
        z_tilde = z.repeat(adj_matrices.shape[1], 1)

        # reconstruct adjacency matrix
        rec_adj_matrices = self.adj_sigmoid(self.dot_product_decode(z_tilde).unsqueeze(0))

        # run decoder forward pass
        rec_feat_matrices = self.decoder(z_tilde.unsqueeze(0), rec_adj_matrices)

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
            # feat_matrices_embedding = self.feat_sigmoid(feat_matrices_embedding)

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

    # compute dot product
    def dot_product_decode(self, z):

        # compute dot product
        dot_product = torch.matmul(z, z.t())

        # return dot product
        return dot_product
