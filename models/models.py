# libraries
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, window_lengths, num_channels, layer_dims: list[int]):

        super(Autoencoder, self).__init__()
        self.window_lengths = window_lengths
        self.num_channels = num_channels
        self.layer_dims = layer_dims
        self.input_dim = sum(window_length * num_channel for window_length,
                             num_channel in zip(window_lengths, num_channels))
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_layers = []
        current_dim = self.input_dim

        for dim in self.layer_dims[:-1]:
            encoder_layers.append(nn.Linear(current_dim, dim))
            encoder_layers.append(nn.BatchNorm1d(dim))
            encoder_layers.append(nn.ReLU())

            current_dim = dim

        dim = self.layer_dims[-1]
        encoder_layers.append(nn.Linear(current_dim, dim))
        current_dim = dim
        return nn.Sequential(*encoder_layers)

    def build_decoder(self):
        decoder_layers = []
        current_dim = self.layer_dims[-1]

        for dim in reversed(self.layer_dims[:-1]):
            decoder_layers.append(nn.Linear(current_dim, dim))
            decoder_layers.append(nn.BatchNorm1d(dim))
            decoder_layers.append(nn.ReLU())
            current_dim = dim
        decoder_layers.append(nn.Linear(current_dim, self.input_dim))
        return nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded