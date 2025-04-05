import os

import numpy as np
import tinygrad.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from tinygrad import Tensor
from tinygrad import TinyJit
from torch.utils.data import DataLoader

from models.vqvae import VQVAE


def get_dataloaders(batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    x_train_var = np.var(train_dataset.data / 255.0)

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, test_loader, x_train_var


def save_snapshot(net, batch, path="results/0/"):
    Tensor.training = False
    os.makedirs(path, exist_ok=True)

    x_hat, _, _, _, _ = net(batch)
    x_hat = x_hat.numpy()
    batch = batch.numpy()

    separator = 11  # Width of the separator
    sep_color = (0, 255, 0)

    for i in range(batch.shape[0]):
        orig = batch[i].transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
        recon = x_hat[i].transpose(1, 2, 0)

        orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
        recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

        h, w, c = orig.shape

        separator_img = np.full((h, separator, c), sep_color, dtype=np.uint8)

        combined = np.concatenate([orig, separator_img, recon], axis=1).astype(np.uint8)
        img = Image.fromarray(combined)

        img.save(os.path.join(path, f"{i}.png"))


def train(epochs, net, optimizer, train_loader, test_loader, x_train_var, log_every=50):
    def step(X):
        Tensor.training = True
        optimizer.zero_grad()

        x_hat, loss_term_1, loss_term_2, perplexity, closest_indices = net(X)

        # MSE loss
        recon_loss = ((x_hat - X) ** 2).mean() / x_train_var
        loss = recon_loss + loss_term_1 + loss_term_2

        loss.backward()
        optimizer.step()

        return loss, perplexity, recon_loss, closest_indices, loss_term_1, loss_term_2

    jit_step = TinyJit(step)

    # Train loop
    for epoch in range(epochs):
        X_test, _ = next(iter(test_loader))
        X_test = Tensor(X_test.numpy())
        save_snapshot(net, X_test, path=f"results/{epoch}")
        running_loss = 0.0
        for i, (X, _) in enumerate(train_loader):
            if X.shape[0] != 32:
                continue

            X = Tensor(X.numpy())
            loss, perplexity, recon_loss, closest_indices, loss_term_1, loss_term_2 = (
                jit_step(X)
            )
            if i % log_every == 0:
                print(
                    f"Epoch {epoch}, step {i} - loss: {loss.item():.5f}, recon_loss: {recon_loss.item():.5f}, perplexity: {perplexity.item():.5f}, closest_indices: {len(np.unique(closest_indices.numpy()))}, loss_term_1: {loss_term_1.item():.5f}, loss_term_2: {loss_term_2.item():.5f}"
                )
            running_loss += loss.item()

        print("Average loss:", running_loss / len(train_loader))


if __name__ == "__main__":
    net = VQVAE(128, 32, 2, 512, 64, 0.25)
    optimizer = nn.optim.Adam(nn.state.get_parameters(net), lr=1e-4)

    train_loader, test_loader, x_train_var = get_dataloaders(batch_size=32)
    train(10, net, optimizer, train_loader, test_loader, x_train_var)
