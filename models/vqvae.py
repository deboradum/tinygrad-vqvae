import mlx.nn as nn


class VQVAE(nn.Module):
    def __init__(self):
        return

    def __call__(self, x):
        # The model takes an input x, that is passed through an encoder producing output z_{e}(x)
        z_e = self.encoder(x)
        # The discrete latent variables z are then calculated by a nearest neighbour look-up using the shared embedding space e
        z = None
        e_k = None
        # The input to the decoder is the corresponding embedding vector e_{k}
        x_hat = self.decoder(e_k)

        return x_hat
