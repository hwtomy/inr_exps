import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

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
