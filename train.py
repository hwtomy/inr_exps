
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utility import MNISTCoordinateDataset
from model import Siren
import matplotlib.pyplot as plt
from utility import gradient, laplace

import os


batch_size = 1
epochs = 5
device = torch.device('cuda:0' )

train_dataset = MNISTCoordinateDataset(root='./mnist_data', train=True, download=True, use_fourier=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = MNISTCoordinateDataset(root='./mnist_data', train=False, download=True, use_fourier=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = Siren(in_features=train_dataset.coords.shape[-1], hidden_features=256, hidden_layers=3,
              out_features=1, outermost_linear=True).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

step = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for coords, pixels in train_loader:
        coords = coords.to(device)
        pixels = pixels.to(device)

        preds, coords_out = model(coords)
        # loss = criterion(preds, pixels)
        loss = ((preds - pixels) ** 2).mean()

        # with torch.no_grad():
        #     img_grad = gradient(preds, coords_out)
        #     img_laplacian = laplace(preds, coords_out)

        #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        #     axes[0].imshow(preds.cpu().view(28, 28).detach().numpy(), cmap='gray')
        #     axes[1].imshow(img_grad.norm(dim=-1).cpu().view(28, 28).detach().numpy(), cmap='gray')
        #     axes[2].imshow(img_laplacian.cpu().view(28, 28).detach().numpy(), cmap='gray')
        #     plt.show()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

    print(f"Epoch {epoch+1}/{epochs} - Average Loss: {total_loss / len(train_loader):.6f}")


model.eval()
with torch.no_grad():
    total_loss = 0.0
    for coords, pixels in test_loader:
        coords = coords.to(device)
        pixels = pixels.to(device)

        preds, _ = model(coords)
        loss = criterion(preds, pixels)

        total_loss += loss.item()

    print(f"Test Loss: {total_loss / len(test_loader):.6f}")
