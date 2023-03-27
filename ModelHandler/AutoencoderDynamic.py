# import pytorch libraries
import torch
import torch.nn as nn

# import project libraries
import ModelHandler.Encoder as Encoder
import ModelHandler.Decoder as Decoder

# AutoencoderDynamic class
class AutoencoderDynamic(nn.Module):

    # define class constructor
    def __init__(self, statistics, feat_embed_dim, encoder_dim, encoder_bottleneck, decoder_dim, decoder_output, bias=True, device='cpu'):

        # call super class constructor
        super(AutoencoderDynamic, self).__init__()

        # init dataset statistics
        self.statistics = statistics

        # init VAE feature embedding layers
        self.feat_embeddings = self.init_embedding_layers(feat_embed_dim)

        # init the encoder model
        self.encoder = Encoder.Encoder(hidden_size=encoder_dim, bottleneck=encoder_bottleneck, bias=bias)

        # init the decoder model
        self.decoder = Decoder.Decoder(hidden_size=decoder_dim, output=decoder_output, bias=bias)

        # init embedding dimension
        self.device = device

    # define autoencoder model forward pass
    def forward(self, input):

        # run encoder forward pass
        z = self.encoder(input)

        # run decoder forward pass
        output = self.decoder(z)

        # return latent representation and output
        return z, output

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
                feat_matrices_values = feat_matrices[:, i].type(torch.LongTensor).to(self.device)

            # case: je segment categorial feature
            elif feature in self.statistics['je_segment_features_categorical']:

                # determine values of current feature
                feat_matrices_values = feat_matrices[:, i].type(torch.LongTensor).to(self.device)

            # case: je segment numerical feature
            elif feature in self.statistics['je_segment_features_numerical']:

                # determine values of current feature
                feat_matrices_values = feat_matrices[:, i].unsqueeze(1).to(self.device)

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
                feat_matrices_embeddings = torch.cat((feat_matrices_embeddings, feat_matrices_embedding), dim=1)

        # return embedded features
        return feat_matrices_embeddings