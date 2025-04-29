
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utility import MNISTCoordinateDataset, psnr, gradient, laplace,preimg
from model import Siren, RobustifiedINR
import matplotlib.pyplot as plt
from utility import gradient, laplace
from torchvision import transforms
from torchvision.datasets import ImageFolder



batch_size = 1
epochs = 5
device = torch.device('cuda:0' )

# train_dataset = MNISTCoordinateDataset(root='./mnist_data', train=True, download=True, use_fourier=False)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = MNISTCoordinateDataset(root='./mnist_data', train=False, download=True, use_fourier=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
])
# dataset = readimg(root_dir='./dataset', transform=transform)
dataset = preimg('./dataset/kodak')
loader  = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

model = RobustifiedINR(coord_dim=20, ff_out_features=40, hidden_features=64, output_dim=1, coordinate_scales=[1.0, 1.0]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

step = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for coords, pixels in loader:
        coords = coords.to(device)
        pixels = pixels.to(device)

        preds = model(coords)
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

    print(f"Epoch {epoch+1}/{epochs}Loss: {total_loss / len(loader)}")


model.eval()
with torch.no_grad():
    total_loss = 0.0
    for coords, pixels in loader:
        coords = coords.to(device)
        pixels = pixels.to(device)

        preds= model(coords)
        loss = psnr(preds, pixels)

        total_psnr += loss

    print(f"Test Loss: ", total_psnr / len(loader))
