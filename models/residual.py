import tinygrad.nn as nn


class ResidualLayer():
    def __init__(self, in_dim, out_dim, h_dim):
        self.conv1 = nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(h_dim, out_dim, kernel_size=1, stride=1, bias=True)

    def __call__(self, x):
        x_ = x.relu()
        x_ = self.conv1(x_)
        x_ = x_.relu()
        x_ = self.conv2(x_)

        return x + x_

class ResidualStack():
    def __init__(self, in_dim, out_dim, h_dim, num_layers):
        self.stack = [ResidualLayer(in_dim, out_dim, h_dim)] * num_layers

    def __call__(self, x):
        for layer in self.stack:
            x = layer(x)
        x = x.relu()

        return x
