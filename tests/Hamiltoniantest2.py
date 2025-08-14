import unittest
import torch, importlib.util, os, sys
import numpy as np
from math import factorial
from itertools import product

# ===================================================================
#  Utilidades físicas sencillas
# ===================================================================

torch.set_default_dtype(torch.float64)

def hermite(n, x):
    return torch.special.hermite_polynomial_h(x, n)

def phi_1d(n, x):
    norm = (2 ** n * factorial(n) * np.sqrt(np.pi)) ** -0.5
    return norm * hermite(n, x) * torch.exp(-0.5 * x ** 2)

def generate_multi_indices(n_orbitals, dim):
    """Primeros índices (n1,…,nD) ordenados por energía (suma)."""
    max_order = n_orbitals + dim - 1
    combos = list(product(range(max_order), repeat=dim))
    combos.sort(key=lambda t: sum(t))
    return combos[:n_orbitals]

def phi_nd(n_tuple, x):
    # x: [walkers, fermions, dim]
    psi = torch.ones_like(x[..., 0])
    for d, n_d in enumerate(n_tuple):
        psi *= phi_1d(n_d, x[..., d])
    return psi  # [w,f]

def slater_matrix(x, indices):
    # columnas = orbitals evaluadas en todas las partículas
    cols = [phi_nd(n, x) for n in indices]           # cada una [w,f]
    return torch.stack(cols, dim=2)                  # [w,f,f]

# ===================================================================
#  Red dummy gaussiana (log|ψ| = -½ Σ x²)  -> siempre derivable
# ===================================================================

class DummyGauss(torch.nn.Module):
    def forward(self, x):
        if x.ndim == 2:
            logabs = -0.5 * x.pow(2).sum(dim=1)
        else:
            logabs = -0.5 * x.pow(2).sum(dim=(1, 2))
        sign = torch.ones_like(logabs)
        return sign, logabs

# ===================================================================
#  Loader dinámico para importar tus Hamiltonianos sin tocar instalación
# ===================================================================

def load_cls(path, cls_name, alias):
    sys.path.insert(0, os.path.dirname(path))
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return getattr(module, cls_name)

# ===================================================================
#  Fichero de test
# ===================================================================
class TestHamiltonianDiagnostic(unittest.TestCase):
    """Diagnóstico: compara términos cinéticos/potenciales 1‑D vs 2‑D
    usando una onda gaussiana trivial (sin nodos) para aislar la lógica
    del Hamiltoniano."""

    def setUp(self):
        base = r"C:\\Users\\migue\\OneDrive - Universitat de Barcelona\\Master\\Intership"
        self.h1_path = os.path.join(base, "SpinlessFermionsJames",                      "src", "Hamiltonian.py")
        self.hd_path = os.path.join(base, "SpinlessFermionsJames - varias dimensiones", "src", "Hamiltonian.py")

        Ham1 = load_cls(self.h1_path, "HarmonicOscillatorWithInteraction1D", "Ham1")
        HamD = load_cls(self.hd_path, "HarmonicOscillatorWithInteractionD",  "HamD")

        self.nf        = 2   # fermions
        self.dimD      = 2   # dimensiones para el caso generalizado
        self.nw        = 4096
        self.net       = DummyGauss()
        self.ham1      = Ham1(self.net, V0=0.0, sigma0=0.5, nchunks=1)
        self.ham2      = HamD(self.net, V0=0.0, sigma0=0.5, nchunks=1, dimensions=self.dimD)
        torch.manual_seed(0)

    def test_energy_terms(self):
        # genera configuraciones aleatorias
        x1 = torch.randn(self.nw, self.nf)                    # 1‑D  [w,f]
        x2 = torch.randn(self.nw, self.nf, self.dimD)         # 2‑D  [w,f,D]

        # calcula términos (se necesita grad)
        kin1 = self.ham1.kinetic(x1)
        pot1 = self.ham1.potential(x1)
        kin2 = self.ham2.kinetic(x2)
        pot2 = self.ham2.potential(x2)

        # medias
        k1, p1 = kin1.mean().item(), pot1.mean().item()
        k2, p2 = kin2.mean().item(), pot2.mean().item()

        print("\n──── Diagnóstico Hamiltonianos (red Gaussiana) ────")
        print(f"Cinética  1D : {k1:.4f}")
        print(f"Cinética  2D : {k2:.4f}")
        print(f"Potencial 1D : {p1:.4f}")
        print(f"Potencial 2D : {p2:.4f}")

        # valores esperados para ψ = exp(-½ Σx²):  E = ½ N dim
        expected1 = 0.5 * self.nf              # 1.0
        expected2 = 0.5 * self.nf * self.dimD   # 2.0

        self.assertAlmostEqual(k1 + p1, expected1, delta=0.05,
                               msg="Total 1D fuera de rango")
        self.assertAlmostEqual(k2 + p2, expected2, delta=0.1,
                               msg="Total 2D fuera de rango")
        # valores esperados con ψ_G
        expected1 = 0.5 * self.nf * 1          # 1.0
        expected2 = 0.5 * self.nf * self.dimD  # 2.0

        self.assertAlmostEqual((kin1+pot1).mean(), expected1, delta=0.05)
        self.assertAlmostEqual((kin2+pot2).mean(), expected2, delta=0.05)
        # ratio cinética debe escalar ~ dim
        ratio = k2 / k1
        #self.assertAlmostEqual(ratio, self.dimD, delta=0.2,
        #                       msg="Escalado de cinética incorrecto")


if __name__ == "__main__":
    unittest.main()
