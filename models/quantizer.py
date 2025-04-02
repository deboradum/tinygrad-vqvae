from tinygrad import Tensor
import tinygrad.nn as nn

class Quantizer:
    def __init__(self, k, d, beta):
        self.k = k
        self.d = d
        self.beta = beta
        self.e = nn.Embedding(k, d)
        self.e.weight = Tensor.uniform(*self.e.weight.shape ,low=- 1.0 / self.k, high=1.0 / self.k)

    def __call__(self, z):
        # (b, c, w, h) > (b, w, h, c) & flatten
        z = z.permute(0, 2, 3, 1)
        z_flat = z.view(-1, self.d)

        # || z_{e}(x) - e_{j} ||_{2}
        dists = (
            (z_flat**2).sum(axis=1, keepdim=True)
            + (self.e.weight**2).sum(axis=1, keepdim=True)
            - 2 * z_flat.matmul(self.e.weight.T)
        )

        # The posterior categorical distribution q(z|x) probabilities are defined as one-hot as follows:
        # q(z=k|x) = { 1 for k = argmin_{j} || z_{e}(x) - e_{j} ||_{2}
        #            { 0 else
        closest_indices = dists.argmin(axis=1).unsqueeze(1)
        one_hot = Tensor.zeros(closest_indices.shape[0], self.k).scatter(1, closest_indices, 1)

        z_q = one_hot.matmul(self.e.weight).view(*z.shape)

        loss = ((z_q.detach() - z) ** 2).mean() + self.beta * ((z_q - z.detach()) ** 2).mean()

        z_q = z + (z_q - z).detach()

        e_mean = one_hot.mean(0)
        perplexity = (-(e_mean * (e_mean + 1e-10).log()).sum()).exp()

        z_q = z_q.permute(0, 3, 1, 2)

        return loss, z_q, perplexity, one_hot, closest_indices
