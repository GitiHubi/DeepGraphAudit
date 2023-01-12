# import pytorch libraries
import torch.nn as nn

# import project libraries
import ModelHandler.Encoder as Encoder
import ModelHandler.Decoder as Decoder

# define the autoencoder class
class Autoencoder(nn.Module):

    # define class constructor
    def __init__(self, encoder_layer, bottleneck, decoder_layer):

        # call super class constructor
        super(Autoencoder, self).__init__()

        # init the encoder model
        self.encoder = Encoder.Encoder(hidden_size=encoder_layer, bottleneck=bottleneck)

        # init the decoder model
        self.decoder = Decoder.Decoder(hidden_size=decoder_layer)

    # define autoencoder model forward pass
    def forward(self, input):

        # run encoder forward pass
        z = self.encoder(input)

        # run decoder forward pass
        output = self.decoder(z)

        # return latent representation and output
        return z, output
