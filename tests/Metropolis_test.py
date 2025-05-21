'''A test for the MetropolisHastings Sampler'''

import unittest
import torch
import numpy as np
from torch import nn, Tensor
from typing import Callable, Tuple
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Add the parent directory (where "src/" is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Now you can import from src
from Samplers import MetropolisHastingsOld
from Models import vLogHarmonicNet



def visualize_metropolis(net, sampler, sweeps: int = 1000, 
                        grid_size: int = 100, grid_range: float = 4.0 ,fermion: int = 0):
    """
    Visualize the Metropolis sampling results with a 3D histogram and heatmap.
    Shows the distribution of the first fermion's coordinates.
    The 3D histogram shows only the last 10% of sweeps.
    
    Args:
        net: Neural network model
        sampler: MetropolisHastings sampler instance
        sweeps (int): Number of sampling sweeps
        grid_size (int): Resolution of the probability density grid
        grid_range (float): Range of x and y axes (-grid_range to grid_range)
    """
    # Collect walker positions
    positions_history = []
    
    @torch.no_grad()
    def run_sweeps_and_record():
        for _ in range(sweeps):
            chains, _ = sampler(n_sweeps=1)
            positions = chains[:, fermion, :]  # Take only one fermion
            positions_history.append(positions)
    
    run_sweeps_and_record()
    
    # Convert positions history to array and flatten
    all_positions = np.concatenate(positions_history, axis=0)
    x, y = all_positions[:, 0], all_positions[:, 1]  # First fermion's x and y coordinates
    
    # Get only last 10% of positions for the 3D histogram
    last_tenth_idx = int(0.9 * len(positions_history))
    recent_positions = np.concatenate(positions_history[last_tenth_idx:], axis=0)
    recent_x, recent_y = recent_positions[:, 0], recent_positions[:, 1]
    
    # Create figure with 3D histogram and heatmap
    fig = plt.figure(figsize=(15, 6))
    
    # 3D histogram (using only last 10% of sweeps)
    ax1 = fig.add_subplot(121, projection='3d')
    hist, xedges, yedges = np.histogram2d(recent_x, recent_y, bins=40, 
                                         range=[[-grid_range, grid_range], 
                                               [-grid_range, grid_range]])
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0
    
    dx = dy = 0.5 * (xedges[1] - xedges[0])
    dz = hist.ravel()
    
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
    ax1.set_title("First Fermion Distribution (3D)\nLast 10% of sweeps")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("Count")
    
    # Heatmap with network probability density
    ax2 = fig.add_subplot(122)
    
    # Calculate wavefunction on grid (for first fermion)
    xx, yy = np.meshgrid(np.linspace(-grid_range, grid_range, grid_size), 
                         np.linspace(-grid_range, grid_range, grid_size))
    # Create grid with first fermion coordinates varying and others fixed at 0
    grid_base = torch.zeros((grid_size * grid_size, net.dof, net.dim), dtype=torch.float32)
    grid_coords = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    grid_base[:, fermion, :] = grid_coords  
    
    with torch.no_grad():
        _, logabs = net(grid_base)
    psi2 = torch.exp(2 * logabs).reshape(grid_size, grid_size).numpy()
    
    im = ax2.imshow(psi2, extent=[-grid_range, grid_range, -grid_range, grid_range], 
                    origin='lower', cmap='viridis')
    plt.colorbar(im, ax=ax2)
    ax2.set_title(r"$|\psi(x, y)|^2$ Network (First Fermion)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    
    # Add walker position scatter plot on top of heatmap (using all sweeps)
    #ax2.scatter(x, y, c='red', alpha=0.1, s=1)
    
    plt.tight_layout()
    plt.show()
'''We start by defining a simple Gaussian log-probability model for testing.'''

class GaussianLogProbNet(nn.Module):
    def __init__(self, dof: int, dim: int):
        super().__init__()
        self.dof = dof
        self.dim = dim
        self.pretrain = False
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        logabs = -0.5 * torch.sum(x ** 2, dim=(1, 2))
        sign = torch.ones_like(logabs)
        return sign, logabs

# Parameters
nwalkers = 400
dof = 1
dim = 2
sweeps = 1000  
net = GaussianLogProbNet(dof, dim)
sampler = MetropolisHastingsOld(net, dof, nwalkers, target_acceptance=0.5, dim=dim)




# The results of the Metropolis sampling are visualized using a 3D histogram and a heatmap.
#visualize_metropolis(net, sampler, sweeps=1000)

'''Now, we are going to have different deviations in each dimension.'''

class GaussianLogProbNet2(nn.Module):
    def __init__(self, dof: int, dim: int):
        super().__init__()
        self.dof = dof
        self.dim = dim
        self.pretrain = False
        self.dummy_param = nn.Parameter(torch.zeros(1))
        # Define standard deviations for each dimension
        self.std = torch.tensor([1.0, 5.0])

    def forward(self, x: torch.Tensor):
        # Scale each dimension by its corresponding standard deviation
        scaled_x = x / self.std.view(1, 1, -1)
        logabs = -0.5 * torch.sum(scaled_x ** 2, dim=(1, 2))
        sign = torch.ones_like(logabs)
        return sign, logabs
    
net2 = GaussianLogProbNet2(dof, dim)

sampler2 = MetropolisHastingsOld(net2, dof, nwalkers, target_acceptance=0.6, dim=dim)
#visualize_metropolis(net2, sampler2, sweeps=1000)

class GaussianLogProbNet3(nn.Module):
    def __init__(self, dof: int = 2, dim: int = 2):
        super().__init__()
        self.dof = dof  # 2 fermions
        self.dim = dim  # 2 dimensions
        self.pretrain = False
        self.dummy_param = nn.Parameter(torch.zeros(1))
        # Define standard deviations for each fermion and dimension
        # Reshape to [dof, dim] = [2, 2] for easier broadcasting
        self.std = torch.tensor([[1.0, 3.0],  # First fermion's x and y std
                               [5.0, 8.0]])    # Second fermion's x and y std

    def forward(self, x: torch.Tensor):
        # x shape: [batch_size, dof, dim]
        # Scale each dimension by its corresponding standard deviation
        # Reshape std to [1, dof, dim] for broadcasting
        scaled_x = x / self.std.view(1, self.dof, self.dim)
        logabs = -0.5 * torch.sum(scaled_x ** 2, dim=(1, 2))
        sign = torch.ones_like(logabs)
        return sign, logabs

# Create network and sampler with new parameters
net3 = GaussianLogProbNet3(dof=2, dim=2)
sampler3 = MetropolisHastingsOld(net3, dof=2, nwalkers=400, target_acceptance=0.6, dim=2)
visualize_metropolis(net3, sampler3, sweeps=1000, grid_range=15.0)  # Increased grid_range to see wider distributions
visualize_metropolis(net3, sampler3, sweeps=1000, grid_range=15.0,fermion=1)  # Increased grid_range to see wider distributions