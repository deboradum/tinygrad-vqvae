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
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.d)

        # || z_{e}(x) - e_{j} ||_{2}
        dists = (
            (z_flat**2).sum(axis=1, keepdim=True)
            + (self.e.weight**2).sum(axis=1, keepdim=True).T
            - 2 * z_flat.matmul(self.e.weight.T)
        )

        # The posterior categorical distribution q(z|x) probabilities are defined as one-hot as follows:
        # q(z=k|x) = { 1 for k = argmin_{j} || z_{e}(x) - e_{j} ||_{2}
        #            { 0 else
        closest_indices = dists.argmin(axis=1)
        one_hot = closest_indices.one_hot(self.k)
        z_q = self.e(closest_indices).view(*z.shape)

        loss_term_1 = ((z_q.detach() - z) ** 2).mean()
        loss_term_2 = self.beta * ((z_q - z.detach()) ** 2).mean()

        e_mean = one_hot.mean(0)
        eps = 1e-10
        perplexity = (-(e_mean + eps) * (e_mean + eps).log()).sum().exp()

        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss_term_1, loss_term_2, z_q, perplexity, one_hot, closest_indices
