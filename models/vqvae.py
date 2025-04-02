import tinygrad.nn as nn

from tinygrad import Tensor

from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import Quantizer

class VQVAE():
    def __init__(self, encoder_h_dim, res_h_dim, num_res_layers, k, d, beta):
        self.encoder = Encoder(3, encoder_h_dim, res_h_dim, num_res_layers)
        self.pre_quantization_conv = nn.Conv2d(
            encoder_h_dim, d, kernel_size=1, stride=1
        )
        self.quantizer = Quantizer(k, d, beta)
        self.decoder = Decoder(d, encoder_h_dim, res_h_dim, num_res_layers)

    def __call__(self, x):
        # The model takes an input x, that is passed through an encoder producing output z_{e}(x)
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        # The discrete latent variables z are then calculated by a nearest neighbour look-up using the shared embedding space e
        loss, z_q, perplexity, _, _ = self.quantizer(z_e)
        # The input to the decoder is the corresponding embedding vector
        x_hat = self.decoder(z_q)

        return x_hat, loss, perplexity


if __name__ == "__main__":
    vqvae = VQVAE(128, 32, 2, 512, 64, 0.25)

    x = Tensor.rand(8, 3, 32, 32)
    print("In shape:", x.shape)
    x_hat, loss, perplexity = vqvae(x)
    print("Out shape:", x_hat.shape)
    print("Loss", loss.numpy())
