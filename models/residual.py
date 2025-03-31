import tinygrad.nn as nn


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        self.conv1 = nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
        self.conv2 = nn.Conv2d(h_dim, out_dim, kernel_size=1, stride=1, bias=False),

    def __call__(self, x):
        return x + self.conv2(self.conv1(x.relu()).relu())

class ResidualStack(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim, num_layers):
        self.stack = [
            ResidualLayer(in_dim, out_dim, h_dim) * num_layers
        ]

    def __call__(self, x):
        for layer in self.stack:
            x = layer(x)
        x = x.relu()

        return x
