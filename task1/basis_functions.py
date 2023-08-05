import torch
import numpy as np

# Define basis functions
def sine_basis(x):
    return torch.sin(x * np.pi)

def gaussian_basis(x):
    mu = torch.linspace(0, 1, 5)
    sigma = 0.1
    return torch.exp(-(x.unsqueeze(-1) - mu)**2 / (2 * sigma**2))

def polynomial_basis(x):
    return torch.stack([x[:, 0]**2, x[:, 1]**2, x[:, 0] * x[:, 1]], dim=-1)