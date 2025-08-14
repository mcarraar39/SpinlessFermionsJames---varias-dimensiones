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
import pandas as pd
from scipy.stats import entropy

# Add the parent directory (where "src/" is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# Now you can import from src
from Samplers import MetropolisHastingsOld
from Models import vLogHarmonicNet

n_sweeps=10000
n_thermal=9000
n_walkers=10000
A=2
dim=2
target_acceptance=0.55

def visualize_metropolis(net, sampler, sweeps: int = n_sweeps, 
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
    ax1.set_title("Walkers Distribution (3D)\nLast 10% of sweeps")
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
    ax2.set_title(r"$|\psi(x, y)|^2$ Network ")
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
# nwalkers = 400
# dof = 1
# dim = 2
# sweeps = 1000  
# net = GaussianLogProbNet(dof, dim)
# sampler = MetropolisHastingsOld(net, dof, nwalkers, target_acceptance=0.5, dim=dim)




# The results of the Metropolis sampling are visualized using a 3D histogram and a heatmap.
#visualize_metropolis(net, sampler, sweeps=1000)

'''Now, we are going to have different deviations in each dimension.'''

# class GaussianLogProbNet2(nn.Module):
#     def __init__(self, dof: int, dim: int):
#         super().__init__()
#         self.dof = dof
#         self.dim = dim
#         self.pretrain = False
#         self.dummy_param = nn.Parameter(torch.zeros(1))
#         # Define standard deviations for each dimension
#         self.std = torch.tensor([1.0, 5.0])

#     def forward(self, x: torch.Tensor):
#         # Scale each dimension by its corresponding standard deviation
#         scaled_x = x / self.std.view(1, 1, -1)
#         logabs = -0.5 * torch.sum(scaled_x ** 2, dim=(1, 2))
#         sign = torch.ones_like(logabs)
#         return sign, logabs
    
# net2 = GaussianLogProbNet2(dof, dim)

# sampler2 = MetropolisHastingsOld(net2, dof, nwalkers, target_acceptance=0.6, dim=dim)
#visualize_metropolis(net2, sampler2, sweeps=1000)

class GaussianLogProbNet3(nn.Module):
    def __init__(self, dof: int = A, dim: int = dim):
        super().__init__()
        self.dof = dof        # fermions
        self.dim = dim        # spatial dimensions
        self.pretrain = False # kept for compatibility

        # Dummy parameter necesario para evitar errores con .parameters()
        self.dummy_param = nn.Parameter(torch.zeros(1))

        # Medias fijas [dof, dim] = [2, 2]
        self.register_buffer("mu", torch.tensor([
            [0.0,  2.0],  # Media del primer fermión
            [-4.0,  2.0],  # Media del segundo fermión
        ]))

        # Desviaciones estándar fijas [dof, dim] = [2, 2]
        self.register_buffer("sigma", torch.tensor([
            [1.0,  3.0],  # Std del primer fermión
            [5.0,  8.0],  # Std del segundo fermión
        ]))

    def forward(self, x: torch.Tensor):
        # x.shape = [batch_size, dof, dim]
        scaled_x = (x - self.mu) / self.sigma  # broadcasting
        logabs = -0.5 * torch.sum(scaled_x ** 2, dim=(1, 2))  # [batch_size]
        sign = torch.ones_like(logabs)
        return sign, logabs

# Create network and sampler with new parameters
net3 = GaussianLogProbNet3(dof=A, dim=dim)
sampler3 = MetropolisHastingsOld(net3, dof=A, nwalkers=n_walkers, target_acceptance=target_acceptance, dim=dim)
visualize_metropolis(net3, sampler3, sweeps=n_sweeps, grid_range=15.0)  # Increased grid_range to see wider distributions
visualize_metropolis(net3, sampler3, sweeps=n_sweeps, grid_range=15.0,fermion=1)  # Increased grid_range to see wider distributions


def compute_empirical_stats(sampler, sweeps: int = n_sweeps, thermalization: int = n_thermal):
    """
    Computes empirical mean, std, absolute error, and Monte Carlo uncertainty for each coordinate.

    Args:
        sampler: Instance of MetropolisHastingsOld
        sweeps (int): Total number of sweeps
        thermalization (int): Number of initial sweeps to discard
    """
    assert thermalization < sweeps, "Thermalization sweeps must be less than total sweeps."
    positions_history = []

    with torch.no_grad():
        for _ in range(sweeps):
            chains, _ = sampler(n_sweeps=1)
            positions_history.append(chains.cpu())

    # Discard burn-in
    positions_history = positions_history[thermalization:]
    all_positions = torch.cat(positions_history, dim=0)  # [samples, dof, dim]
    N = all_positions.shape[0]  # Total number of post-burn-in samples

    # Statistics
    mean = all_positions.mean(dim=0)        # [dof, dim]
    std = all_positions.std(dim=0, unbiased=True)          # [dof, dim]
    se = std / np.sqrt(N)                   # Standard error (uncertainty of the mean)
    std_se = std / torch.sqrt(torch.tensor(2. * (N - 1)))  # Standard error of the std

    # Reference values from network
    ref_mu = sampler.network.mu.cpu()
    ref_sigma = sampler.network.sigma.cpu()

    print(f"{'Ferm':>4s} {'Coord':>5s} {'Mean':>10s} {'RefMu':>10s} {'AbsErr':>10s} {'±SE':>10s} {'Std':>10s} {'RefStd':>10s} {'Std±SE':>10s}")
    print("-" * 95)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            m = mean[i, j].item()
            s = std[i, j].item()
            sem = se[i, j].item()
            ssem = std_se[i, j].item()
            ref_m = ref_mu[i, j].item()
            ref_s = ref_sigma[i, j].item()
            print(f"{i:>4d} {j:>5d} {m:10.4f} {ref_m:10.4f} {abs(m - ref_m):10.4f} {sem:10.4f} {s:10.4f} {ref_s:10.4f} {ssem:10.4f}")
            
compute_empirical_stats(sampler3, sweeps=n_sweeps, thermalization = n_thermal)

target_acceptance_list = np.arange(0.15, 0.95, 0.05)
results = []

for targ in target_acceptance_list:
    print(f"\nEvaluating for target_acceptance = {targ}")
    sampler = MetropolisHastingsOld(
        network=GaussianLogProbNet3(dof=A, dim=dim),
        dof=A,
        nwalkers=n_walkers,
        target_acceptance=targ,
        dim=dim,
    )

    x, _ = sampler(n_sweeps=n_sweeps)

    mu_ref = sampler.network.mu.cpu()
    std_ref = sampler.network.sigma.cpu()
    x_cpu = x.cpu()
    nwalkers = x.shape[0]

    for a in range(2):  # Fermions
        for d in range(2):  # Coordinates
            values = x_cpu[:, a, d]
            mean = values.mean().item()
            std = values.std(unbiased=True).item()
            se = std / nwalkers**0.5
            std_se = std / (2 * (nwalkers - 1))**0.5
            mu_ = mu_ref[a, d].item()
            std_ = std_ref[a, d].item()
            results.append({
                "AcceptRate": sampler.acceptance_rate,
                "Target": targ,
                "Ferm": a,
                "Coord": d,
                "Mean": mean,
                "RefMu": mu_,
                "AbsErr": abs(mean - mu_),
                "±SE": se,
                "Std": std,
                "RefStd": std_,
                "Std±SE": std_se
            })

df = pd.DataFrame(results)
pd.set_option('display.precision', 4)

# Agrupar y mostrar por fermión y coordenada
grouped = df.groupby(["Ferm", "Coord"])
for (ferm, coord), subdf in grouped:
    print(f"\n==== Fermión {ferm} – Coordenada {coord} ====")
    print(subdf[["Target", "AcceptRate", "Mean", "RefMu", "AbsErr", "±SE", "Std", "RefStd", "Std±SE"]].to_string(index=False))

def compute_kl_divergence_marginals(samples, reference_means, reference_stds, bins=100):
    """
    Computes the marginal KL divergence between empirical samples and reference Gaussians.

    Args:
        samples (Tensor): [nwalkers, dof, dim]
        reference_means (Tensor): [dof, dim]
        reference_stds (Tensor): [dof, dim]
        bins (int): Number of histogram bins for density estimation

    Returns:
        pd.DataFrame: KL divergence for each fermion and coordinate
    """
    samples = samples.cpu().numpy()
    ref_mu = reference_means.cpu().numpy()
    ref_sigma = reference_stds.cpu().numpy()

    dof, dim = ref_mu.shape
    results = []

    for a in range(dof):
        for d in range(dim):
            x = samples[:, a, d]
            mu, sigma = ref_mu[a, d], ref_sigma[a, d]

            hist, bin_edges = np.histogram(x, bins=bins, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            ref_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((bin_centers - mu) / sigma) ** 2)

            hist += 1e-12
            ref_pdf += 1e-12
            kl = entropy(hist, ref_pdf)

            results.append({
                "Fermion": a,
                "Coord": d,
                "KL(P||Q)": kl
            })

    return pd.DataFrame(results)
samples, _ = sampler3(n_sweeps=n_sweeps)
df_kl = compute_kl_divergence_marginals(samples, sampler3.network.mu, sampler3.network.sigma)
print(df_kl)