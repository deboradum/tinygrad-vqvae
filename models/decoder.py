import tinygrad.nn as nn
from residual import ResidualStack


class Decoder():
    def __init__(self, in_dim, h_dim, res_h_dim, num_res_layers):
        self.inv_conv_layers = [
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, num_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=4, stride=2, padding=1),
        ]

    def __call__(self, x):
        for layer in self.inv_conv_layers[:-1]:
            x = layer(x)
        x = x.relu()
        x = self.inv_conv_layers[-1](x)

        return x
