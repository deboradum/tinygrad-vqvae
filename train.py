import os
import torch

import numpy as np
import tinygrad.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from PIL import Image
from tinygrad import Tensor
from tinygrad import TinyJit
from tinygrad.helpers import tqdm
from torch.utils.data import DataLoader

from models.vqvae import VQVAE


def get_dataloaders(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    train_images = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])  # Shape: (50000, 3, 32, 32)
    x_train_var = train_images.var().item()

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

    x_hat, _, _ = net(batch)
    x_hat = x_hat.numpy()
    batch = batch.numpy()

    # print("Batch min/max:", batch.min(), batch.max(), batch.var(axis=(2,3)).mean(1))
    # print("x_hat min/max:", x_hat.min(), x_hat.max(), x_hat.var(axis=(2, 3)).mean(1))

    # print("\n")
    # print("-"*30)
    # print(x_hat[0])
    # print("-"*30)
    # # print(x_hat[0].grad)
    # # print("-"*30)
    # print("\n")

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


def train(epochs, net, optimizer, train_loader, test_loader, x_train_var):
    def step(X):
        Tensor.training = True
        optimizer.zero_grad()

        x_hat, emb_loss, perplexity = net(X)
        # x_hat = x_hat.clamp(min_=1e-6, max_=1 - 1e-6)

        # MSE loss
        # recon_loss = ((x_hat - X) ** 2).mean() / x_train_var
        recon_loss = ((x_hat - X) ** 2).mean()
        loss = recon_loss + emb_loss

        # print(emb_loss.numpy(), recon_loss.numpy())

        loss.backward()
        optimizer.step()

        return loss, perplexity
    jit_step = TinyJit(step)

    # Train loop
    for epoch in range(epochs):
        X_test, _ = next(iter(test_loader))
        X_test = Tensor(X_test.numpy())
        save_snapshot(net, X_test, path=f"results/{epoch}")
        running_loss = 0.0
        for X, _ in tqdm(train_loader):
            if X.shape[0] != 64:
                continue
            X = Tensor(X.numpy())
            loss, perplexity = jit_step(X)

            running_loss += loss.item()

        print("Average loss:", running_loss / len(train_loader))


if __name__ == "__main__":
    net = VQVAE(128, 32, 2, 256, 128, 0.05)
    # optimizer = nn.optim.Adam(nn.state.get_parameters(net), lr=0.00001)
    optimizer = nn.optim.SGD(nn.state.get_parameters(net), lr=0.000001)

    train_loader, test_loader, x_train_var = get_dataloaders(batch_size=64)
    train(10, net, optimizer, train_loader, test_loader, x_train_var)
