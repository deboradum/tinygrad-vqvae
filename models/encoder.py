import tinygrad.nn as nn

from residual import ResidualStack


class Encoder():
    def __init__(self, in_dim, h_dim, res_h_dim, num_res_layers):
        self.conv_layers = [
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1),
        ]
        self.res = ResidualStack(h_dim, h_dim, res_h_dim, num_res_layers)

    def __call__(self, x):
        for layer in self.conv_layers[:-1]:
            x = layer(x).relu()
        x = self.conv_layers[-1](x)

        return self.res(x)
