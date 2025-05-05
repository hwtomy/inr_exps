import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utility import MNISTCoordinateDataset, psnr, gradient, laplace, recons, save_images, preimg
from model import Siren, RobustifiedINR
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
import wandb


def train(model, coords, pixels, optimizer, device, scheduler,epochs):
    model.train()
    coords = coords.to(device)
    pixels = pixels.to(device)

    for epoch in tqdm(range(epochs), desc="Training"):
        preds = model(coords)
        loss = ((preds - pixels) ** 2).mean()
        wandb.log({"loss": loss})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


def evaluate_one_image(model, coords, pixels, device, H, W, output_dir, idx):
    model.eval()
    coords = coords.to(device)
    pixels = pixels.to(device)

    with torch.no_grad():
        preds = model(coords)
        loss = psnr(preds, pixels)
        imgs = recons(preds, H, W)
        # imgs = recons(preds, H*, W)
        save_images(imgs, output_dir, prefix=f"pred{idx:03d}")

    return loss


def main():

    # Config
    batch_size = 1
    epochs = 20000
    output_dir = './output/kodak'
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")

    # Load dataset
    dataset = preimg('./dataset/kodak')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_losses = []

    for idx, (coords, pixels, (W, H)) in enumerate(loader, start=1):
        print(f"\nImage {idx}")
        wandb.init(
            project="image-fitting",
            name=f"image_{idx}",  
            config={
                "epochs": epochs
                },
            reinit=True
                   )

        model = RobustifiedINR(
            coord_dim=2,
            ff_out_features=256,
            hidden_features=256,
            output_dim=3,
            coordinate_scales=[1.0, 1.0]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=3e-3)
        # scheduler = CosineAnnealingLR(optimizer, T_max=20000, eta_min=0.0001)
        scheduler = LambdaLR(optimizer, lambda x: 0.1**min(x/epochs, 1))

        train(model, coords, pixels, optimizer, device, scheduler,epochs)


        psnr_score = evaluate_one_image(model, coords, pixels, device, H, W, output_dir, idx)
        print(f"PSNR loss: {psnr_score:.4f} dB")

        all_losses.append(psnr_score)

    avg_psnr = sum(all_losses) / len(all_losses)
    print(f"\nAverage PSNR : {avg_psnr:.4f} dB")


if __name__ == "__main__":
    main()