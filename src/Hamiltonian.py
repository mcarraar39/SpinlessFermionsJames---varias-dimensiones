import torch
from torch import nn, Tensor
from typing import Tuple

import numpy as np
class HarmonicOscillatorWithInteractionD(nn.Module):
    """Computes the local energy for spinless fermions in a 1D harmonic trap with Gaussian interactions.

    The input is of shape (nwalkers, nparticles, ndimensions)
    
    The Hamiltonian has three terms:
    1. Kinetic energy: -1/2 ∇²
    2. Harmonic potential: 1/2 x²
    3. Gaussian interaction between particles: V₀/(√(2π)σ₀) exp(-(xᵢ-xⱼ)²/(2σ₀²))
    
    Args:
        net (nn.Module): Neural network representing the many-body wavefunction
        V0 (float): Strength of the Gaussian interaction
        sigma0 (float): Width of the Gaussian interaction
        nchunks (int): Number of chunks for batch processing
    """
    def __init__(self, net: nn.Module, V0: float, sigma0: float, nchunks: int, dimensions: int) -> None:
        super(HarmonicOscillatorWithInteractionD, self).__init__()
        self.net = net
        self.V0 = V0  # Interaction strength
        self.sigma0 = sigma0  # Interaction range
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))  # Normalization constant
        self.dimensions = dimensions # Number of dimensions
    def kinetic(self, x: Tensor) -> Tensor:
        nwalkers, nfermions, ndim = x.shape

        # Flatten spatial dimensions: [nwalkers, A * D]
        x_flat = x.view(nwalkers, -1)

        # Create separate leaf tensors
        xis = [xi.requires_grad_() for xi in x_flat.t()]
        x_reconstructed = torch.stack(xis, dim=1)  # [nwalkers, A * D]
        x_reshaped = x_reconstructed.view_as(x)    # [nwalkers, A, D]

        # Forward through the network
        _, ys = self.net(x_reshaped)

        ones = torch.ones_like(ys)

        # First derivatives: [nwalkers, A * D]
        dy_dxs_flat, = torch.autograd.grad(ys, x_reconstructed, ones, retain_graph=True, create_graph=True)

        # Second derivatives (Laplacian): match per-xi logic
        lay_ys = sum(
            torch.autograd.grad(
                dy_dxi, xi, torch.ones_like(dy_dxi), retain_graph=True, create_graph=False
            )[0]
            for xi, dy_dxi in zip(xis, dy_dxs_flat.t())
        )

        # Kinetic energy: -1/2 (∇² log|ψ| + |∇ log|ψ||²)
        ek_local = -0.5 * (lay_ys + dy_dxs_flat.pow(2).sum(dim=1))
        return ek_local



    def potential(self, x: Tensor) -> Tensor:
        """Computes harmonic oscillator potential: V = 1/2 Σᵢ xᵢ²"""
        return 0.5 * x.pow(2).view(x.size(0), -1).sum(dim=1)
        #return 0.5 * x.pow(2).sum(dim=(-1, -2))

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        """Computes pairwise Gaussian interactions between particles:
        V_int = V₀/(√(2π)σ₀) Σᵢ<ⱼ exp(-(xᵢ-xⱼ)²/(2σ₀²))
        """
        pairwise_sqdist = (x.unsqueeze(-2) - x.unsqueeze(-1)).pow(2).sum(dim=-1)  # shape: [nwalkers, A, A]
        interactions = torch.exp(-pairwise_sqdist / (2 * self.sigma0**2)).triu(diagonal=1)
        return self.gauss_const * interactions.sum(dim=(-2, -1))
    def forward(self, x: Tensor) -> Tensor:
        """Computes total local energy E_L = -1/2 ∇²ψ/ψ + V + V_int"""
        _kin = self.kinetic(x)
        _pot = self.potential(x)
        _int = self.gaussian_interaction(x)

        _eloc = _kin+_pot + _int
        return _eloc, _kin, _pot, _int
    
class HarmonicOscillatorWithInteractionDold2(nn.Module):
    """Computes the local energy for spinless fermions in a 1D harmonic trap with Gaussian interactions.

    The input is of shape (nwalkers, nparticles, ndimensions)
    
    The Hamiltonian has three terms:
    1. Kinetic energy: -1/2 ∇²
    2. Harmonic potential: 1/2 x²
    3. Gaussian interaction between particles: V₀/(√(2π)σ₀) exp(-(xᵢ-xⱼ)²/(2σ₀²))
    
    Args:
        net (nn.Module): Neural network representing the many-body wavefunction
        V0 (float): Strength of the Gaussian interaction
        sigma0 (float): Width of the Gaussian interaction
        nchunks (int): Number of chunks for batch processing
    """
    def __init__(self, net: nn.Module, V0: float, sigma0: float, nchunks: int, dimensions: int) -> None:
        super(HarmonicOscillatorWithInteractionDold2, self).__init__()
        self.net = net
        self.V0 = V0  # Interaction strength
        self.sigma0 = sigma0  # Interaction range
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))  # Normalization constant
        self.dimensions = dimensions # Number of dimensions

    def kinetic(self, x: Tensor) -> Tensor:
        """Computes kinetic energy: -1/2 ∇²ψ/ψ using automatic differentiation.
    
        Args:
            x (Tensor): Input tensor of shape [nwalkers, nfermions, ndim]
        Returns:
            Tensor: Kinetic energy per walker
        """
        x = x.requires_grad_()  # Enable gradient tracking for the whole tensor
        
        # Get log|ψ| from network using full tensor
        _, ys = self.net(x)
        #print(f"ys gen shape: {ys.shape}")
        ones = torch.ones_like(ys)
        
        # Initialize accumulators
        laplacian = torch.zeros_like(ys)
        grad_sq_sum = torch.zeros_like(ys)
        
        # Calculate derivatives for each dimension
        for dim in range(x.shape[-1]):
            # First derivatives for this dimension
            grad_dim = torch.autograd.grad(ys, x, ones, retain_graph=True, create_graph=True)[0][..., dim]
            grad_sq_sum += (grad_dim**2).sum(dim=1)  # Sum over particles
            
            # Second derivatives (Laplacian terms)
            for particle in range(x.shape[1]):
                second_deriv = torch.autograd.grad(
                    grad_dim[:, particle], 
                    x, 
                    torch.ones_like(grad_dim[:, particle]),
                    retain_graph=True
                )[0][..., dim]
                laplacian += second_deriv[:, particle]
        #print(f"laplacian gen:{laplacian}")
        # Compute kinetic energy: -1/2(∇²log|ψ| + (∇log|ψ|)²)
        kinetic_energy = -0.5 * (laplacian + grad_sq_sum) #
        
        return kinetic_energy

    def potential(self, x: Tensor) -> Tensor:
        """Computes harmonic oscillator potential: V = 1/2 Σᵢ xᵢ²"""
        return 0.5 * x.pow(2).sum(dim=(-1, -2))

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        """Computes pairwise Gaussian interactions between particles:
        V_int = V₀/(√(2π)σ₀) Σᵢ<ⱼ exp(-(xᵢ-xⱼ)²/(2σ₀²))
        """
        pairwise_sqdist = (x.unsqueeze(-2) - x.unsqueeze(-1)).pow(2).sum(dim=-1)  # shape: [nwalkers, A, A]
        interactions = torch.exp(-pairwise_sqdist / (2 * self.sigma0**2)).triu(diagonal=1)
        return self.gauss_const * interactions.sum(dim=(-2, -1))
    def forward(self, x: Tensor) -> Tensor:
        """Computes total local energy E_L = -1/2 ∇²ψ/ψ + V + V_int"""
        _kin = self.kinetic(x)
        _pot = self.potential(x)
        _int = self.gaussian_interaction(x)

        _eloc = _kin#+_pot + _int
        return _eloc

class HarmonicOscillatorWithInteractionDold(nn.Module):
    """Computes the local energy for spinless fermions in a 1D harmonic trap with Gaussian interactions.

    The input is of shape (nwalkers, nparticles, ndimensions)
    
    The Hamiltonian has three terms:
    1. Kinetic energy: -1/2 ∇²
    2. Harmonic potential: 1/2 x²
    3. Gaussian interaction between particles: V₀/(√(2π)σ₀) exp(-(xᵢ-xⱼ)²/(2σ₀²))
    
    Args:
        net (nn.Module): Neural network representing the many-body wavefunction
        V0 (float): Strength of the Gaussian interaction
        sigma0 (float): Width of the Gaussian interaction
        nchunks (int): Number of chunks for batch processing
    """
    def __init__(self, net: nn.Module, V0: float, sigma0: float, nchunks: int, dimensions: int) -> None:
        super(HarmonicOscillatorWithInteractionDold, self).__init__()
        self.net = net
        self.V0 = V0  # Interaction strength
        self.sigma0 = sigma0  # Interaction range
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))  # Normalization constant
        self.dimensions = dimensions # Number of dimensions

    def kinetic(self, x: Tensor) -> Tensor:
        """Computes kinetic energy: -1/2 ∇²ψ/ψ using automatic differentiation.
        
        Uses the neural network to compute derivatives of log|ψ| w.r.t. particle positions.
        Returns kinetic energy per MCMC walker.
        """
        # Prepare inputs for gradient calculation
        x = x.requires_grad_()  # Keep tracking
        nwalkers = x.shape[0]

        # Get log|ψ| from the network
        _, log_psi = self.net(x)
        ones = torch.ones_like(log_psi)

        # Gradient: d log|ψ| / dx
        grad = torch.autograd.grad(log_psi, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]  # shape: [nwalkers, A, D]
        grad_flat = grad.view(nwalkers, -1)  # [nwalkers, A*D] if needed

        # Laplacian: sum of second derivatives
        laplacian = torch.zeros_like(log_psi)
        for i in range(grad_flat.shape[1]):
            grad_i = grad_flat[:, i]
            second_deriv = torch.autograd.grad(grad_i, x, grad_outputs=ones, retain_graph=True)[0]
            laplacian += second_deriv.view(nwalkers, -1)[:, i]

        grad_sq = grad_flat.pow(2).sum(dim=1)
        kinetic_energy = -0.5 * (laplacian + grad_sq)
        return kinetic_energy

    def potential(self, x: Tensor) -> Tensor:
        """Computes harmonic oscillator potential: V = 1/2 Σᵢ xᵢ²"""
        return 0.5 * x.pow(2).sum(dim=(-1, -2))

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        """Computes pairwise Gaussian interactions between particles:
        V_int = V₀/(√(2π)σ₀) Σᵢ<ⱼ exp(-(xᵢ-xⱼ)²/(2σ₀²))
        """
        pairwise_sqdist = (x.unsqueeze(-2) - x.unsqueeze(-1)).pow(2).sum(dim=-1)  # shape: [nwalkers, A, A]
        interactions = torch.exp(-pairwise_sqdist / (2 * self.sigma0**2)).triu(diagonal=1)
        return self.gauss_const * interactions.sum(dim=(-2, -1))
    def forward(self, x: Tensor) -> Tensor:
        """Computes total local energy E_L = -1/2 ∇²ψ/ψ + V + V_int"""
        _kin = self.kinetic(x)
        _pot = self.potential(x)
        _int = self.gaussian_interaction(x)

        _eloc = _pot + _int
        return _eloc


class HarmonicOscillatorWithInteraction1D(nn.Module):
    """Computes the local energy for spinless fermions in a 1D harmonic trap with Gaussian interactions.

    The input is of shape (nwalkers, nparticles)
    
    The Hamiltonian has three terms:
    1. Kinetic energy: -1/2 ∇²
    2. Harmonic potential: 1/2 x²
    3. Gaussian interaction between particles: V₀/(√(2π)σ₀) exp(-(xᵢ-xⱼ)²/(2σ₀²))
    
    Args:
        net (nn.Module): Neural network representing the many-body wavefunction
        V0 (float): Strength of the Gaussian interaction
        sigma0 (float): Width of the Gaussian interaction
        nchunks (int): Number of chunks for batch processing
    """
    def __init__(self, net: nn.Module, V0: float, sigma0: float, nchunks: int) -> None:
        super(HarmonicOscillatorWithInteraction1D, self).__init__()
        self.net = net
        self.V0 = V0  # Interaction strength
        self.sigma0 = sigma0  # Interaction range
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))  # Normalization constant

    def kinetic(self, x: Tensor) -> Tensor:
        """Computes kinetic energy: -1/2 ∇²ψ/ψ using automatic differentiation.
        
        Uses the neural network to compute derivatives of log|ψ| w.r.t. particle positions.
        Returns kinetic energy per MCMC walker.
        """
        # Prepare inputs for gradient calculation
        xis = [xi.requires_grad_() for xi in x.flatten(start_dim=1).t()]
        xs_flat = torch.stack(xis, dim=1)

        # Get log|ψ| from network
        _, ys = self.net(xs_flat.view_as(x))
        ones = torch.ones_like(ys)

        # Calculate first derivatives ∇log|ψ|
        (dy_dxs, ) = torch.autograd.grad(ys, xs_flat, ones, retain_graph=True, create_graph=True)

        # Calculate Laplacian ∇²log|ψ|
        lay_ys = sum(torch.autograd.grad(dy_dxi, xi, ones, retain_graph=True, create_graph=False)[0] \
                    for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis)))))

        # Kinetic energy: -1/2(∇²log|ψ| + (∇log|ψ|)²)
        ek_local_per_walker = -0.5 * (lay_ys + dy_dxs.pow(2).sum(-1))
        return ek_local_per_walker

    def potential(self, x: Tensor) -> Tensor:
        """Computes harmonic oscillator potential: V = 1/2 Σᵢ xᵢ²"""
        return 0.5*(x.pow(2).sum(-1))

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        """Computes pairwise Gaussian interactions between particles:
        V_int = V₀/(√(2π)σ₀) Σᵢ<ⱼ exp(-(xᵢ-xⱼ)²/(2σ₀²))
        """
        return self.gauss_const * ( torch.exp(-(x.unsqueeze(-2) - x.unsqueeze(-1))**2/(2*self.sigma0**2)).triu(diagonal=1).sum(dim=(-2,-1)) )

    def forward(self, x: Tensor) -> Tensor:
        """Computes total local energy E_L = -1/2 ∇²ψ/ψ + V + V_int"""
        _kin = self.kinetic(x)
        _pot = self.potential(x)
        _int = self.gaussian_interaction(x)

        _eloc = _kin + _pot + _int
        return _eloc

class GaussianInteraction1D(nn.Module):
    """Computes the local energy for spinless fermions with Gaussian interactions in 1D.
    
    The Hamiltonian has two terms:
    1. Kinetic energy: -1/2 ∇²
    2. Gaussian interaction between particles: V₀/(√(2π)σ₀) exp(-(xᵢ-xⱼ)²/(2σ₀²))
    
    Unlike HarmonicOscillatorWithInteraction1D, this system has no external potential,
    representing fermions in free space with interactions.
    
    Args:
        net (nn.Module): Neural network representing the many-body wavefunction
        V0 (float): Strength of the Gaussian interaction
        sigma0 (float): Width of the Gaussian interaction
    """
    def __init__(self, net: nn.Module, V0: float, sigma0: float) -> None:
        super(GaussianInteraction1D, self).__init__()
        self.net = net
        self.V0 = V0  # Interaction strength
        self.sigma0 = sigma0  # Interaction range
        self.gauss_const = (self.V0/(np.sqrt(2*np.pi)*self.sigma0))  # Normalization constant

    def kinetic(self, x: Tensor) -> Tensor:
        """Computes kinetic energy: -1/2 ∇²ψ/ψ using automatic differentiation.
        
        Uses the neural network to compute derivatives of log|ψ| w.r.t. particle positions.
        Returns kinetic energy per MCMC walker.
        """
        # Prepare inputs for gradient calculation
        xis = [xi.requires_grad_() for xi in x.flatten(start_dim=1).t()]
        xs_flat = torch.stack(xis, dim=1)

        # Get log|ψ| from network
        _, ys = self.net(xs_flat.view_as(x))
        ones = torch.ones_like(ys)

        # Calculate first derivatives ∇log|ψ|
        (dy_dxs, ) = torch.autograd.grad(ys, xs_flat, ones, retain_graph=True, create_graph=True)

        # Calculate Laplacian ∇²log|ψ|
        lay_ys = sum(torch.autograd.grad(dy_dxi, xi, ones, retain_graph=True, create_graph=False)[0] \
                    for xi, dy_dxi in zip(xis, (dy_dxs[..., i] for i in range(len(xis)))))

        # Kinetic energy: -1/2(∇²log|ψ| + (∇log|ψ|)²)
        ek_local_per_walker = -0.5 * (lay_ys + dy_dxs.pow(2).sum(-1))
        return ek_local_per_walker

    def gaussian_interaction(self, x: Tensor) -> Tensor:
        """Computes pairwise Gaussian interactions between particles:
        V_int = V₀/(√(2π)σ₀) Σᵢ<ⱼ exp(-(xᵢ-xⱼ)²/(2σ₀²))
        
        The triu operation ensures we only count each interaction pair once.
        """
        return self.gauss_const * (torch.exp(-(x.unsqueeze(-2) - x.unsqueeze(-1))**2/(2*self.sigma0**2))
                                  .triu(diagonal=1).sum(dim=(-2,-1)))

    def forward(self, x: Tensor) -> Tensor:
        """Computes total local energy E_L = -1/2 ∇²ψ/ψ + V_int
        
        Unlike the harmonic oscillator case, there's no external potential term.
        """
        _kin = self.kinetic(x)
        _int = self.gaussian_interaction(x)

        _eloc = _kin + _int
        return _eloc