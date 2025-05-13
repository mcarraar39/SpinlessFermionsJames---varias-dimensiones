import unittest
import torch
import torch.nn as nn
import importlib.util
import sys
import os

torch.set_default_dtype(torch.float64)  # Set default dtype to float64
def load_class_with_deps(file_path: str, class_name: str, module_alias: str):
    """Load models.py and dependencies in isolation."""
    src_dir = os.path.dirname(file_path)
    original_sys_modules = sys.modules.copy()
    sys.path.insert(0, src_dir)
    try:
        spec = importlib.util.spec_from_file_location(module_alias, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_alias] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)
    finally:
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        sys.path.pop(0)


class TestHamiltonianComparison(unittest.TestCase):

    def setUp(self):
        # 🔧 Configuration section – change easily here
        # ---- Configuration ----
        self.seed = 42
        self.nwalkers = 1064
        self.nfermions = 4
        self.dim = 1
        self.nsteps = 100

        self.target_acceptance = 0.5
        self.num_hidden = 8
        self.num_layers = 2
        self.num_dets = 2
        self.func = nn.Tanh()
        self.pretrain = False
        # Paths to both Hamiltonian.py files (not Models.py anymore)
        
        self.model_orig_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames\src\Models.py"
        self.hamil_orig_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames\src\Hamiltonian.py"

        self.model_gen_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Models.py"
        self.hamil_gen_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Hamiltonian.py"


    def test_hamiltonian_comparison(self):
        # Create random input tensor (nwalkers, nparticles, ndimensions)
        torch.manual_seed(self.seed)
        x = torch.randn(self.nwalkers, self.nfermions, self.dim)

        # Load models from the respective paths
        # Load and initialize original
        OriginalNet = load_class_with_deps(self.model_orig_path, "vLogHarmonicNet", "model_orig")
        OrigHamil = load_class_with_deps(self.hamil_orig_path, "HarmonicOscillatorWithInteraction1D", "hamil_orig")

        model_orig = OriginalNet(self.nfermions, self.num_hidden, self.num_layers,
                                self.num_dets, self.func, self.pretrain)
        torch.manual_seed(self.seed)
        hamil_orig = OrigHamil(net=model_orig, V0=0, sigma0=0.5, nchunks=1)
        
        # Load and initialize generalized
        GeneralizedNet = load_class_with_deps(self.model_gen_path, "vLogHarmonicNet", "model_gen")
        GenHamil = load_class_with_deps(self.hamil_gen_path, "HarmonicOscillatorWithInteractionD", "hamil_orig")

        model_gen = GeneralizedNet(self.nfermions, self.num_hidden, self.num_layers,
                                self.num_dets, self.func, self.pretrain,self.dim)
        torch.manual_seed(self.seed)
        hamil_gen = GenHamil(net=model_gen, V0=0, sigma0=0.5, nchunks=1, dimensions=self.dim)

        # Ensure both models start with the same weights
        model_gen.load_state_dict(model_orig.state_dict())

        torch.manual_seed(self.seed)
        

        # Flatten x to match the network input size for the 1D model
        x_flat = x.squeeze(-1) if x.shape[-1] == 1 else ValueError(f"self.dim must be 1, but got {self.dim}") # Flatten for the 1D model (if needed)
        #print(f"Flattened input shape: {x_flat.shape}")
        # Compute local energy for both Hamiltonians
        # Compute energies for both Hamiltonians (which return the total energy)
        energy_1D = hamil_orig(x_flat)  # Total energy for 1D model
        energy_D = hamil_gen(x)    # Total energy for generalized model

        # Print the results
        print(f"Energy for 1D model: {energy_1D}")
        print(f"Energy for generalized model: {energy_D}")

        # Compare energies exactly (without tolerance)
        if isinstance(energy_1D, torch.Tensor) and isinstance(energy_D, torch.Tensor):
            energy_match = energy_1D == energy_D  # Exact equality comparison
            print(f"Energies match (exact): {energy_match}")
        else:
            print("Error: The energies are not tensors or could not be compared.")

        # Find the indices where the energies do not match
        mismatch_indices = torch.nonzero(~energy_match).squeeze()

        if mismatch_indices.numel() > 0:
            print(f"There is a mismatch in {len(mismatch_indices)}/{self.nwalkers} energies.")
            print(f"The greatest error is {torch.max(torch.abs(energy_1D - energy_D))}.")
            print("The following indices have mismatched energies:")
            print(mismatch_indices)
            # Optionally, print the actual mismatched energy values
            for idx in mismatch_indices:
                print(f"Mismatch at index {idx.item()}: 1D energy = {energy_1D[idx].item()}, Generalized energy = {energy_D[idx].item()}")
        else:
            print("All energies match.")

        # Check if all energy values match
        if energy_match.all():
            print("Test passed. The energies are exactly equal.")
        else:
            print("Test failed. Some energy values do not match.")



# Run the test
if __name__ == "__main__":
    unittest.main()
