import torch
import importlib.util
import sys
import os
import numpy as np

def load_class_from_path(path, class_name, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)

# ==== Paths to your Pretraining.py files ====
path_1d = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames\src\Pretraining.py"
path_nd = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Pretraining.py"

# ==== Load the classes ====
Hermite1D = load_class_from_path(path_1d, "HermitePolynomialMatrix", "pretrain1D")
HermiteND = load_class_from_path(path_nd, "HermitePolynomialMatrixND", "pretrainND")

# ==== Parameters ====
nwalkers = 128
nfermions = 4
dim = 1

# ==== Create same input for both ====
x_1d = torch.randn(nwalkers, nfermions)              # [nwalkers, A]
x_nd = x_1d.unsqueeze(-1)                             # [nwalkers, A, 1]

# ==== Initialize and evaluate both models ====
model_1d = Hermite1D(num_particles=nfermions)
model_nd = HermiteND(num_particles=nfermions, Dim=dim)

with torch.no_grad():
    output_1d = model_1d(x_1d)        # [nwalkers, 1, A, nfermions]
    output_nd = model_nd(x_nd)       # [nwalkers, 1, A, nfermions]

# ==== Compare outputs ====
are_close = torch.allclose(output_1d, output_nd, atol=1e-6)

print(f"Shapes match: {output_1d.shape == output_nd.shape}")
print(f"Values match: {are_close}")
print(f"Max abs difference: {(output_1d - output_nd).abs().max().item()}")

if not are_close:
    # Optional: print a few mismatches
    diff = (output_1d - output_nd).abs()
    idx = (diff > 1e-6).nonzero(as_tuple=True)
    print(f"First few mismatches (index and values):")
    for i in range(min(5, len(idx[0]))):
        print(f"At {tuple(dim[i].item() for dim in idx)}: {output_1d[idx[0][i], idx[1][i], idx[2][i], idx[3][i]].item()} vs {output_nd[idx[0][i], idx[1][i], idx[2][i], idx[3][i]].item()}")
