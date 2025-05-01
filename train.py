
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utility import MNISTCoordinateDataset, psnr, gradient, laplace, recons, save_images, preimg
from model import Siren, RobustifiedINR
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
import lpips
from tqdm import tqdm



batch_size = 1
epochs = 1
device = torch.device('cuda:0' )

# train_dataset = MNISTCoordinateDataset(root='./mnist_data', train=True, download=True, use_fourier=False)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# test_dataset = MNISTCoordinateDataset(root='./mnist_data', train=False, download=True, use_fourier=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# transform = transforms.Compose([
#     transforms.Resize((28,28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,),(0.5,)),
# ])
# dataset = readimg(root_dir='./dataset', transform=transform)
dataset = preimg('./dataset/kodak')
loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

model = RobustifiedINR(coord_dim=2, ff_out_features=40, hidden_features=64, output_dim=3, coordinate_scales=[1.0, 1.0]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

step = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for coords, pixels, (W,H)in loader:
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

output_dir = './output/kodak'
model.eval()
with torch.no_grad():
    total_psnr = 0.0
    for idx, (coords, pixels, (W,H)) in enumerate(loader, start=1):
        coords = coords.to(device)
        pixels = pixels.to(device)

        preds= model(coords)
        loss = psnr(preds, pixels)
        # loss_lpips = lpips_fn(preds, pixels).mean()
        # B, N, C = preds.shape
        # H = W = int(N**0.5)
        # preds_img  = preds .view(B, H, W, C).permute(0,3,1,2)
        # target_img = pixels.view(B, H, W, C).permute(0,3,1,2)


        # ssim_map = ssim(preds_img, target_img,
        #             data_range=1.0,
        #             size_average=False)   # shape: (B,)
   
        # loss_ssim = (1.0 - ssim_map).mean()
        imgs = recons(preds, H, W)  # (B,3,H,W)

        save_images(imgs, output_dir, prefix = f"pred{idx:03d}")
        total_psnr += loss

    print(f"Test Loss: ", total_psnr / len(loader))
