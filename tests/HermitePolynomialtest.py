import torch
import unittest
import importlib.util
import sys
import os
import numpy as np
from itertools import product
from math import factorial, pi, sqrt
from typing import List, Tuple

def load_class_from_path(path, class_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)



# ───────────────────────── utilidades ──────────────────────────
def _generate_multi_indices(nfermions: int, dim: int) -> List[Tuple[int]]:
    """Devuelve las primeras tuplas (n1,…,n_dim) ordenadas por n1+…+n_dim."""
    combos, N = [], 0
    while len(combos) < nfermions:
        combos.extend([t for t in product(range(N + 1), repeat=dim)
                       if sum(t) == N])
        N += 1
    return combos[:nfermions]


# ────────────────────── función principal ──────────────────────
def analytical_hermite_nd(x: torch.Tensor,
                          nfermions: int,
                          dim: int) -> torch.Tensor:
    """
    Matriz de Slater analítica para fermiones no interactuantes
    en un oscilador armónico D-dimensional.

    Args
    ----
    x         : [nwalkers, nfermions, dim]
    nfermions : nº de partículas / orbitales
    dim       : dimensionalidad espacial

    Returns
    -------
    [nwalkers, 1, nfermions, nfermions]
    """
    assert x.shape[1:] == (nfermions, dim)

    nwalkers = x.shape[0]
    device, dtype = x.device, x.dtype
    qnums = _generate_multi_indices(nfermions, dim)          # lista de tuplas

    orbitals = []                                            # [nwalkers, nfermions] por cada φ_j
    for n_tuple in qnums:
        phi = torch.ones_like(x[..., 0], dtype=dtype, device=device) #[nwalkers,nfermions]
        #print(f"The shape of x is {x.shape}")
        #print(f"The shape of phi is {phi.shape}")
        for d, nd in enumerate(n_tuple):
            # print(f"n_tuple is {n_tuple}")
            # print(f"d is {d}")
            # print(f"nd is {nd}")
            H_nd  = torch.special.hermite_polynomial_h(x[..., d], nd)
            norm  = 1.0 / sqrt((2.0 ** nd) * factorial(nd) * sqrt(pi))
            phi  *= norm * torch.exp(-0.5 * x[..., d] ** 2) * H_nd
        orbitals.append(phi)
    # print(f"The shape of orbitals is {len(orbitals)}")
    # print(f"The orbitals are {orbitals}")
    slater = torch.stack(orbitals, dim=-1)     # [nwalkers, nfermions, nfermions]
    # print(f"The shape of slater is {slater.shape}")
    # print(f"The slater is {slater}")
    return slater.unsqueeze(1)                 # [nwalkers, 1, nfermions, nfermions]

class TestHermitePolynomials(unittest.TestCase):
    def setUp(self):
        # Load HermiteND class
        path_nd = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Pretraining.py"
        self.HermiteND = load_class_from_path(path_nd, "HermitePolynomialMatrixND", "pretrainND")
        
        # Test parameters
        self.nwalkers = 436
        self.nfermions = 6
        self.dim = 4
        
        # Initialize model
        self.model_nd = self.HermiteND(num_particles=self.nfermions, Dim=self.dim)
        
        # Create test input
        torch.manual_seed(0)  # For reproducibility
        self.x = torch.rand(self.nwalkers, self.nfermions, self.dim)
        #self.x = torch.tensor([[1.,0.],[0.,1.]]).reshape(1,2,2)

    def test_hermite_polynomial_values(self):
        # Get numerical result
        with torch.no_grad():
            numerical = self.model_nd(self.x)
        
        # Get analytical result
        analytical = analytical_hermite_nd(self.x, self.nfermions, self.dim)
        
        # Compare shapes
        self.assertEqual(numerical.shape, analytical.shape, 
                        "Shape mismatch between numerical and analytical results")
        
        # Compare values
        max_diff = (numerical - analytical).abs().max().item()
        self.assertLess(max_diff, 1e-6, 
                       f"Maximum absolute difference {max_diff} exceeds tolerance")
        
        # Optional: Print detailed comparison
        print(f"Shapes: numerical {numerical.shape}, analytical {analytical.shape}")
        print(f"Max absolute difference: {max_diff}")
        #print(f"X is {self.x}")
        #print(f"Analitical values: {analytical}")
        #print(f"1 partido de raiz de pi es: {1/sqrt(pi)}")
        
        if max_diff >= 1e-6:
            diff = (numerical - analytical).abs()
            idx = (diff > 1e-6).nonzero(as_tuple=True)
            print("\nFirst few mismatches (index and values):")
            for i in range(min(5, len(idx[0]))):
                print(f"At {tuple(dim[i].item() for dim in idx)}: "
                      f"numerical={numerical[idx[0][i], idx[1][i], idx[2][i], idx[3][i]].item()} "
                      f"vs analytical={analytical[idx[0][i], idx[1][i], idx[2][i], idx[3][i]].item()}")

if __name__ == '__main__':
    unittest.main()


