import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
    

class LF(nn.Module):
    def __init__(self, num_filters=40, kernel_size=401, sample_rate=16000,
                 min_freq=50.0, max_freq=8000.0):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq

        center_freqs = np.geomspace(min_freq, max_freq, num=num_filters)
        bandwidths = 1.019 * 24.7 * (4.37 * center_freqs / 1000 + 1)

        self.register_parameter('center_freqs', nn.Parameter(
            torch.tensor(center_freqs, dtype=torch.float32)))  # f_c
        self.register_parameter('bandwidths', nn.Parameter(
            torch.tensor(bandwidths, dtype=torch.float32)))     # b

        t = torch.arange(0, kernel_size).float() / sample_rate
        self.register_buffer('t', t)  # time base

    def forward(self, x):

        filters = self.gf()
        x = F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2)
        return x

    def gf(self):

        t = self.t.unsqueeze(0) 
        fc = self.center_freqs.unsqueeze(1)  
        bw = self.bandwidths.unsqueeze(1)   

        n = 4.0
        envelope = (t ** (n - 1)) * torch.exp(-2 * np.pi * bw * t)


        carrier = torch.cos(2 * np.pi * fc * t)

        filters = envelope * carrier
        filters = filters / torch.norm(filters, dim=1, keepdim=True)  
        return filters.unsqueeze(1)  



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
    
    

class RobustifiedINR(nn.Module):
    def __init__(self, coord_dim, ff_out_features, hidden_features, output_dim, coordinate_scales):
        super().__init__()
        
   
        self.fourier_encoder = FourierFeatureMap(coord_dim, ff_out_features, coordinate_scales)


        self.linear1 = nn.Linear(ff_out_features, ff_out_features, bias=False)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(ff_out_features, ff_out_features, bias=False)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(ff_out_features, ff_out_features, bias=False)


        self.fc1 = nn.Linear(ff_out_features, hidden_features)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.relu4 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_features, hidden_features)
        self.relu5 = nn.ReLU()
        self.fc5 = nn.Linear(hidden_features, hidden_features)
        self.relu6 = nn.ReLU()
        self.fc6 = nn.Linear(hidden_features, output_dim)

    def forward(self, coords):

        ff_encoded = self.fourier_encoder(coords)

        # Adaptive Filtering
        mask = self.linear1(ff_encoded)
        mask = self.relu1(mask)
        mask = self.linear2(mask)
        mask = self.relu2(mask)
        mask = self.linear3(mask)
        
        filtered = mask * ff_encoded  
        
        # Final Prediction
        out = self.fc1(filtered)
        out = self.relu2(out)
        out = self.fc2(out)
        out = self.relu3(out)
        out = self.fc3(out)
        out = self.relu4(out)
        out = self.fc4(out)
        out = self.relu5(out)
        out = self.fc5(out)
        out = self.relu6(out)
        out = self.fc6(out)


        return out