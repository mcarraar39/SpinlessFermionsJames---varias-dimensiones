'''This script runs a post-pretraining energy stability check  of the Monte Carlo Sampler.
This will evaluate the local energy over several sweeps and plot its distribution.
It is intended to prove if the MC Sampler is stable after pretraining.'''

import torch
import matplotlib.pyplot as plt
import numpy as np
from src.utils import clip

def run_energy_stability_check(net, sampler, calc_elocal, device, sweeps=10000):
    """
    Performs a stability analysis of the local energy after neural network pretraining.
    
    This function evaluates how the local energy evolves over multiple Monte Carlo sweeps
    and visualizes the results through two plots:
    1. The evolution of mean energy and its standard deviation across sweeps
    2. The distribution of local energies across all walkers and sweeps
    
    The analysis helps determine if the Monte Carlo sampling has reached equilibrium
    and if the energy estimates are stable.

    Args:
        net (nn.Module): Neural network model that defines the wavefunction
                        Should have pretrain=False
        sampler (MetropolisHastings): Monte Carlo sampler instance configured
                                     with the desired number of walkers
        calc_elocal (Callable): Function that computes local energy
                               Returns (eloc, kinetic, potential, interaction)
        device (torch.device): Device to run computations on (CPU/GPU)
        sweeps (int, optional): Number of Monte Carlo sweeps to perform. 
                               Defaults to 1000.

    Returns:
        None. Displays plots and prints statistics to console:
        - Progress updates every 50 sweeps
        - Final energy mean, standard deviation, and standard error
        - Two visualization plots showing energy evolution and distribution

    Side Effects:
        - Sets network to eval mode during computation
        - Restores network to train mode after completion
        - Creates and displays matplotlib figures
        
    Note:
        The local energy is computed for each walker in each sweep,
        allowing analysis of both the ensemble average and its fluctuations.
    """
    print("\n▶ Running energy stability check after pretraining...")

    net.eval()
    sampler.network = net
    
    # Track energies per sweep
    sweep_energies = []
    sweep_means = []
    sweep_stds = []
    
    for sweep in range(sweeps):
        with torch.no_grad():  # Only disable gradients for sampling
            x, _ = sampler(n_sweeps=1)
        
        # Calculate energies for this sweep
        eloc, _kin, _potential, _inter = calc_elocal(x)  # Tensor [nwalkers]
        energies = eloc.detach().cpu().numpy()
        
        # Store results
        sweep_energies.append(energies)
        sweep_means.append(np.mean(energies))
        sweep_stds.append(np.std(energies))
        
        if sweep % 1000 == 0:  # Progress update every 50 sweeps
            print(f"Sweep {sweep}: Mean E = {sweep_means[-1]:.6f} ± {sweep_stds[-1]:.6f}")

    # Convert to numpy arrays
    sweep_means = np.array(sweep_means)
    sweep_stds = np.array(sweep_stds)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
    
    # Plot 1: Energy evolution
    sweep_mins = [np.min(energies) for energies in sweep_energies]
    sweep_maxs = [np.max(energies) for energies in sweep_energies]

    ax1.plot(sweep_means, 'b-', label='Mean Energy', alpha=0.7)
    ax1.plot(sweep_mins, 'r--', label='Min Energy', alpha=0.5)
    ax1.plot(sweep_maxs, 'g--', label='Max Energy', alpha=0.5)
    ax1.fill_between(np.arange(sweeps), 
                     sweep_means - sweep_stds, 
                     sweep_means + sweep_stds, 
                     color='b', alpha=0.2, label='±1σ')
    ax1.set_xlabel('Sweep')
    ax1.set_ylabel('Local Energy')
    ax1.set_title('Energy Evolution During Sampling')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Final distribution
    all_energies = np.concatenate(sweep_energies)
    final_mean = np.mean(all_energies)
    final_std = np.std(all_energies)
    
    ax2.hist(all_energies, bins=80, density=True, color='purple', alpha=0.7)
    ax2.axvline(final_mean, color='k', linestyle='--', 
                label=f'Mean = {final_mean:.4f} ± {final_std:.4f}')
    ax2.set_xlabel(r'$E_{\mathrm{loc}}$')
    ax2.set_ylabel('Density')
    ax2.set_title('Overall Energy Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print final statistics
    print(f"\nFinal energy statistics after {sweeps} sweeps:")
    print(f"Mean     = {final_mean:.6f}")
    print(f"Std      = {final_std:.6f}")
    print(f"Std. Err = {final_std/np.sqrt(len(all_energies)):.6f}")
    print(f"Maximum = {np.max(all_energies):.6f}")
    print(f"Minimum = {np.min(all_energies):.6f}")
    net.train()  # Restore training mode
    print("Energy stability check completed.\n")
    