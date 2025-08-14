import torch
import torch.nn as nn

import numpy as np
from scipy.special import factorial

from typing import Tuple
from torch import Tensor

import torch
import torch.nn as nn
import numpy as np
from itertools import product
from math import factorial
from typing import Tuple

class HermitePolynomialMatrixND(nn.Module):
    def __init__(self, num_particles: int, Dim: int) -> None:
        """
        Generalized Hermite orbital matrix for D-dimensional harmonic oscillator.

        :param num_particles: Number of single-particle orbitals
        :param Dim: Dimensionality of the space
        """
        super(HermitePolynomialMatrixND, self).__init__()
        self.num_particles = num_particles
        self.Dim = Dim

        # Generate multi-dimensional quantum numbers (combinations of 1D quantum numbers)
        self.quantum_numbers = self._generate_multi_indices()

    def _generate_multi_indices_2(self) -> list[tuple[int]]:
        """
        Devuelve `num_particles` multi-índices cubriendo de forma
        equilibrada los niveles degenerados.
        """
        indices = []
        N = 0
        rng = np.random.default_rng(0)   # fija semilla si quieres reproduc.
        while len(indices) < self.num_particles:
            # todas las tuplas con suma == N
            layer = [t for t in product(range(N+1), repeat=self.Dim)
                    if sum(t) == N]
            rng.shuffle(layer)           # baraja para no sesgar orientación
            indices.extend(layer)
            N += 1

        return indices[:self.num_particles]

    def _generate_multi_indices(self) -> list[Tuple[int]]:
        """
        Generate the list of D-dimensional quantum number combinations for num_particles orbitals.
        Returns a list of tuples (n1, n2, ..., nD).
        """
        max_order = self.num_particles + self.Dim - 1
        all_combinations = list(product(range(max_order), repeat=self.Dim))
        # Sort combinations by total quantum number (sum of elements)
        all_combinations.sort(key=lambda x: sum(x))

        # Return the first `num_particles` combinations
        return all_combinations[:self.num_particles]

    def hermite(self, n: int, x: torch.Tensor) -> torch.Tensor:
        return torch.special.hermite_polynomial_h(x, n)

    def orbital_nd(self, n: Tuple[int], x: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-dimensional orbital: product of 1D orbitals across D dimensions.

        :param n: A tuple of length D with quantum numbers for each dimension
        :param x: Input tensor of shape [nwalkers, A, D]
        :return: Tensor of shape [nwalkers, A] (orbital value per particle)
        """
        assert len(n) == x.shape[-1] == self.Dim
        orbital = torch.ones_like(x[..., 0])
        for d, nd in enumerate(n):
            Hn_xd = self.hermite(nd, x[..., d])
            env = torch.exp(-0.5 * x[..., d] ** 2)
            norm = ((2 ** nd) * factorial(nd) * np.sqrt(np.pi)) ** -0.5
            orbital *= Hn_xd * norm * env
        return orbital  # shape: [nwalkers, A]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the matrix of orbitals for each walker and particle.
        Output shape: [nwalkers, 1, num_particles]
        """
        orbitals = [self.orbital_nd(n, x) for n in self.quantum_numbers]
        stacked = torch.stack(orbitals, dim=-1)  # [nwalkers, A, num_particles]
        return stacked.unsqueeze(1)  # [nwalkers, 1, A, num_particles]


class HermitePolynomialMatrix(nn.Module):

    def __init__(self, num_particles: int) -> None:
        r"""Constructor of class
        :param nfermions: The number of fermions in the exact solution
        :type nfermions: int

        :param device:
        :type device: device container
        """
        super(HermitePolynomialMatrix, self).__init__()
        self.num_particles = num_particles

    def log_factorial(self, n: int) -> float:
      return np.sum(np.log(np.arange(1,n+1,1)))

    def hermite(self, n: int, x: Tensor) -> Tensor:
        return torch.special.hermite_polynomial_h(x, n)
        
    def orbital(self, n: int, x: Tensor) -> Tensor:
        r"""Method class to calculate the n-th single particle orbital of
            the groundstate of the non-interacting Harmonic Oscillator.
            Tensors will be passed to the cpu to compute the Hermite Polynomials
            and subsequently passed back to `device`

        :param n: The order of the Hermite polynomial
        :type n: int

        :param x: The many-body positions
        :type x: class: `torch.Tensor`

        :return out: The values of single particle orbitals of the n-th order
                     for current many-body positions `x`.
        :type out: class: `torch.Tensor`
        """
        Hn_x = self.hermite(n, x)
        
        env = torch.exp(-0.5*x**2)
        norm = ( (2**n)*factorial(n)*(np.sqrt(np.pi)) )**(-0.5)

        orbitaln_x = Hn_x * norm * env
        return orbitaln_x

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        return torch.stack([ self.orbital(n=n, x=x) for n in range(self.num_particles) ], dim=-1).unsqueeze(1) #unsqueeze to state only one matrix at index, 1.
        