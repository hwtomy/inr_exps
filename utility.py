import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch
from torch import nn
import math
from torchvision import transforms
import os
import glob
from PIL import Image
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

def get_mgrid(resolution, dim=2):
    tensors = tuple(torch.linspace(-1, 1, steps=resolution) for _ in range(dim))
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid.reshape(-1, dim)

def fourier_feature_mapping(coords, num_freqs=10):

    freq_bands = 2.0 ** torch.arange(num_freqs).float() * np.pi
    mapped = [coords]
    for freq in freq_bands:
        for fn in [torch.sin, torch.cos]:
            mapped.append(fn(coords * freq))
    return torch.cat(mapped, dim=-1)

class MNISTCoordinateDataset(Dataset):
    def __init__(self, root='./mnist_data', train=True, download=True, use_fourier=True, num_freqs=10):
        self.dataset = datasets.MNIST(root=root, train=train, download=download, transform=transforms.ToTensor())
        self.resolution = 28
        self.use_fourier = use_fourier
        self.num_freqs = num_freqs

        self.coords = get_mgrid(self.resolution)

        # self.coords = fourier_feature_mapping(self.coords, self.num_freqs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        img = img.squeeze(0).reshape(-1, 1) 
        return self.coords.clone(), img 
    
def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



class FourierFeatureMap(nn.Module):    
    def __init__(self, in_features, out_features, coordinate_scales):
        super().__init__()

        self.num_freq = out_features // 2
        self.out_features = out_features
        self.coordinate_scales = nn.Parameter(torch.tensor(coordinate_scales).unsqueeze(dim=0))
        self.coordinate_scales.requires_grad = False
        self.linear = nn.Linear(in_features, self.num_freq, bias=False)
        self.init_weights()
        self.linear.weight.requires_grad = False
    
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.normal_(std=1, mean=0)

    def forward(self, input):
        return torch.cat((np.sqrt(2)*torch.sin(self.linear(self.coordinate_scales*input)), 
                          np.sqrt(2)*torch.cos(self.linear(self.coordinate_scales*input))), dim=-1)
    
def psnr(preds, targets, max_pixel_value=1.0):
    mse = F.mse_loss(preds, targets, reduction='mean')
    if mse == 0:
        return float('inf')
    return 20 * math.log10(max_pixel_value) - 10 * math.log10(mse.item())


class preimg(Dataset):
    def __init__(self, root_dir, resize=None):
        self.paths = glob.glob(os.path.join(root_dir, '*.png'))
        self.resize = resize
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.resize:
            img = img.resize(self.resize, Image.BILINEAR)

        W, H = img.size
        #grid
        xs = torch.linspace(0, 1, W)
        ys = torch.linspace(0, 1, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # (H*W, 2)

        pix = self.to_tensor(img)            
        pix = pix.permute(1,2,0).view(-1,3) * 2 - 1 

        return coords, pix, (W, H)
    

def recons(preds, W, H):

    B, N, C = preds.shape

    img = preds.view(B, W, H, C)

    img = img.permute(0, 3, 1, 2)
    img = (img + 1.0) / 2.0
    return img.clamp(0.0, 1.0)


def save_images(img_tensor, output_dir, prefix = "pred"):

    os.makedirs(output_dir, exist_ok=True)
    to_pil = transforms.ToPILImage()
    B = img_tensor.shape[0]
    for i in range(B):
        img = img_tensor[i].cpu()
        pil = to_pil(img)
        filename = f"{prefix}_{i:03d}.png"
        pil.save(os.path.join(output_dir, filename))