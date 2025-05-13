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
from Samplers import MetropolisHastings
from Models import vLogHarmonicNet

annimation=True
# # Dummy harmonic oscillator network
# class HarmonicOscillatorNet(nn.Module):
#     def __init__(self, pretrain=False):
#         super().__init__()
#         self.pretrain = pretrain  # needed for compatibility with your sampler
#         self.linear = nn.Linear(1, 1)  # still unused
#     def forward(self, x: torch.Tensor):
#         # x: [nwalkers, A, D]
#         log_psi = -0.5 * torch.sum(x**2, dim=(-1, -2))  # shape: [nwalkers]
#         sign = torch.ones_like(log_psi)  # wavefunction is positive
#         return sign, log_psi

# # Parameters
# nwalkers = 4096
# A = 6         # number of particles
# D = 2        # dimensions
# num_hidden = 64
# num_layers = 3
# num_dets = 1
# func = nn.Tanh() 
# pretrain=False



# net = vLogHarmonicNet(num_input=A,
#                       num_hidden=num_hidden,
#                       num_layers=num_layers,
#                       num_dets=num_dets,
#                       func=func,
#                       pretrain=pretrain,
#                       Dim=D)


# sampler = MetropolisHastings(
#     network=net,
#     dof=A,
#     nwalkers=nwalkers,
#     target_acceptance=0.6,
#     dim=D
# )

# # Run the Metropolis sampler
# samples, log_probs = sampler(n_sweeps=2000)

# print("samples shape:", samples.shape)       # [nwalkers, A, D]
# print("log_probs shape:", log_probs.shape)   # [nwalkers]

# # Visualize results (assuming A=1, D=1)
# positions = samples[:, 0, 0].cpu().numpy()

# plt.hist(positions, bins=100, density=True, alpha=0.7, label='Sampled')
# x = torch.linspace(-4, 4, 1000)
# plt.plot(x, torch.exp(-x**2).numpy(), label='True PDF ∝ exp(-x²)')
# plt.title("Metropolis Samples for 1D Harmonic Oscillator")
# plt.xlabel("x")
# plt.ylabel("Probability Density")
# plt.legend()
# plt.grid()
# plt.show()


if annimation==False:
    class GaussianLogProbNet(nn.Module):
        def __init__(self, dof: int, dim: int):
            super().__init__()
            self.dof = dof
            self.dim = dim
            self.pretrain = False
            self.dummy_param = nn.Parameter(torch.zeros(1))  # Para evitar StopIteration

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            logabs = -0.5 * torch.sum(x ** 2, dim=(1, 2))
            sign = torch.ones_like(logabs)
            return sign, logabs


    class TestMetropolisSampler(unittest.TestCase):
        def test_gaussian_sampling(self):
            torch.manual_seed(0)

            nwalkers = 5000
            dof = 1
            dim = 2
            sweeps = 5000

            net = GaussianLogProbNet(dof=dof, dim=dim)
            sampler = MetropolisHastings(network=net,
                                        dof=dof,
                                        nwalkers=nwalkers,
                                        target_acceptance=0.5,
                                        dim=dim)
            #sampler.reset_walkers()
            samples, _ = sampler(n_sweeps=sweeps)

            samples = samples.squeeze(1).cpu().numpy()

            mean = np.mean(samples, axis=0)
            var = np.var(samples, axis=0)
            print(f"Mean: {mean}, Var: {var}")

            for i in range(dim):
                self.assertAlmostEqual(mean[i], 0.0, delta=0.1)
                self.assertAlmostEqual(var[i], 1.0, delta=0.3)


    if __name__ == '__main__':
        # Ejecutar test (comentar si solo querés ver plots)
        unittest.main(exit=False)

        # ===============================
        # Visualización interactiva
        # ===============================
        torch.manual_seed(0)

        dof = 1
        dim = 2
        nwalkers = 5000
        sweeps = 10000

        net = GaussianLogProbNet(dof=dof, dim=dim)
        sampler = MetropolisHastings(network=net,
                                    dof=dof,
                                    nwalkers=nwalkers,
                                    target_acceptance=0.5,
                                    dim=dim)
        #sampler.reset_walkers()
        samples, _ = sampler(n_sweeps=sweeps)

        samples = samples.squeeze(1).cpu().numpy()

        # Histograma 2D de walkers
        x, y = samples[:, 0], samples[:, 1]

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist2d(x, y, bins=60, range=[[-3, 3], [-3, 3]], cmap='plasma', density=True)
        axs[0].set_title("Distribución de walkers")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")

        # Malla de la función de onda
        grid_size = 100
        xx, yy = np.meshgrid(np.linspace(-3, 3, grid_size), np.linspace(-3, 3, grid_size))
        grid = torch.tensor(np.stack([xx, yy], axis=-1), dtype=torch.float32).reshape(-1, 1, 2)
        with torch.no_grad():
            _, logabs = net(grid)
        psi2 = torch.exp(2 * logabs).reshape(grid_size, grid_size).numpy()

        axs[1].imshow(psi2, extent=[-3, 3, -3, 3], origin='lower', cmap='viridis')
        axs[1].set_title(r"$|\psi(x, y)|^2$ de la red")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")

        plt.tight_layout()
        plt.show()

if annimation==True:
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


    # Parámetros de simulación
    nwalkers = 400
    dof = 1
    dim = 2
    sweeps = 1000  # frames de la animación

    net = GaussianLogProbNet(dof, dim)
    sampler = MetropolisHastings(net, dof, nwalkers, target_acceptance=0.5, dim=dim)
    sampler.reset_walkers()

    # Guardar posiciones en cada sweep
    positions_history = []

    @torch.no_grad()
    def run_sweeps_and_record():
        for _ in range(sweeps):
            chains, _ = sampler(n_sweeps=1)
            positions = chains.squeeze(1).cpu().numpy()  # [nwalkers, 2]
            positions_history.append(positions)

    run_sweeps_and_record()

    # ============================
    # Animación con matplotlib
    # ============================

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter([], [], s=10, c='blue', alpha=0.6)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title("Evolución de los walkers")

    def update(frame):
        pos = positions_history[frame]
        sc.set_offsets(pos)
        ax.set_title(f"Sweep {frame+1}")
        return sc,

    ani = FuncAnimation(fig, update, frames=sweeps, interval=100, blit=True,repeat=False)
    plt.show()