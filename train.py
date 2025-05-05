import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from utility import MNISTCoordinateDataset, psnr, gradient, laplace, recons, save_images, preimg
from model import INRGabor
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import os
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR,LambdaLR
import wandb


def train(model, coords, pixels, optimizer, device, scheduler,epochs,temp_dir, idx, H, W):
    model.train()
    coords = coords.to(device)
    pixels = pixels.to(device)

    for epoch in tqdm(range(epochs), desc="Training"):
        preds,activation = model(coords)
        i=len(activation)
        # print(i)
        # exit()
        if epoch % 500 == 0:
            for j in range(i):
                # print(activation[j].shape)
                # imgs = recons(activation[j], H, W)
                # save_images(imgs,temp_dir, prefix=f"pred{idx:02d}{epoch:05d}{j}")
                act = activation[j].squeeze(0).permute(1, 0).reshape(-1, H, W).cpu() 


                grid = make_grid(
                    act.unsqueeze(1),  # [C, 1, H, W]
                    nrow=8, normalize=True, pad_value=1
                    )

                pil_img = ToPILImage()(grid)
                save_path = os.path.join(temp_dir, f"grid_pred{idx:02d}_epoch{epoch:05d}_layer{j}.png")
                pil_img.save(save_path)
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
        preds,activations = model(coords)
        loss = psnr(preds, pixels)
        imgs = recons(preds, H, W)
        save_images(imgs, output_dir, prefix=f"pred{idx}")

    return loss


def main():

    # Config
    batch_size = 1
    epochs = 5000
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_dir = './output/kodak'
    temp_dir = './output/kodak/temp'

    # output_dir = os.path.join(output_dir, timestamp)
    # temp_dir = os.path.join(temp_dir, timestamp)
    os.makedirs(temp_dir, exist_ok=True)
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

        model = INRGabor(
                in_features=2,
                out_features=3, 
                hidden_features=256,
                hidden_layers=2,
                first_omega_0=10.0,
                hidden_omega_0=10.0,
                scale=10.0,
                pos_encode=False,
                sidelength=H
                ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=8e-3)
        # scheduler = CosineAnnealingLR(optimizer, T_max=2000, eta_min=0.00001)
        scheduler = LambdaLR(optimizer, lambda x: 0.1**min(x/epochs, 1))
        train(model, coords, pixels, optimizer, device, scheduler,epochs,temp_dir,idx,H,W)


        psnr_score = evaluate_one_image(model, coords, pixels, device, H, W, output_dir, idx)
        print(f"PSNR loss: {psnr_score:.4f} dB")

        all_losses.append(psnr_score)
        exit()

    avg_psnr = sum(all_losses) / len(all_losses)
    print(f"\nAverage PSNR : {avg_psnr:.4f} dB")


if __name__ == "__main__":
    main()