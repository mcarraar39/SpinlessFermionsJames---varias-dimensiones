import unittest
import torch
import torch.nn as nn
import importlib.util
import sys
import os


def load_module_with_dependencies(model_path: str, class_name: str, module_alias: str):
    """Load models.py and dependencies in isolation."""
    src_dir = os.path.dirname(model_path)
    original_sys_modules = sys.modules.copy()
    sys.path.insert(0, src_dir)

    try:
        spec = importlib.util.spec_from_file_location(module_alias, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {model_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_alias] = mod
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)
    finally:
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        sys.path.pop(0)


def check_equivariance_per_walker(sign_orig, sign_swapped, logabs_orig, logabs_swapped, atol=1e-5):
    """Check antisymmetry per walker and return count and masks."""
    logabs_ok = torch.isclose(logabs_orig, logabs_swapped, atol=atol)
    sign_ok = torch.isclose(sign_orig, -sign_swapped, atol=atol)
    total = sign_orig.shape[0]
    passed = (logabs_ok & sign_ok).sum().item()
    return passed, total, logabs_ok, sign_ok


class TestFermionSwapEquivariance(unittest.TestCase):

    def setUp(self):
        # 🔧 Configuration section – change easily here
        self.nwalkers = 1064
        self.num_input = 5                 # Number of fermions
        self.swap_indices = [0, 1]         # Fermion indices to swap (e.g. swap 0 and 2)
        self.dim = 1                       # Spatial dimension

        # Neural network architecture
        self.num_hidden = 8
        self.num_layers = 2
        self.num_dets = 1
        self.func = nn.Tanh()
        self.pretrain = False

        # Paths to both models.py
        self.original_model_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames\src\Models.py"
        self.generalized_model_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Models.py"

    def test_fermion_swap_equivariance(self):
        # Create input tensors
        input_gen = torch.randn(self.nwalkers, self.num_input, self.dim)
        input_orig = input_gen.squeeze(-1)  # Shape for original model

        # Create index permutation for swapping
        idx = torch.arange(self.num_input)
        idx[self.swap_indices] = idx[self.swap_indices[::-1]]  # Swap selected indices

        # Apply swap
        input_gen_swapped = input_gen[:, idx, :]
        input_orig_swapped = input_orig[:, idx]

        # Load models
        OriginalNet = load_module_with_dependencies(
            self.original_model_path, "vLogHarmonicNet", "models_original"
        )
        GeneralizedNet = load_module_with_dependencies(
            self.generalized_model_path, "vLogHarmonicNet", "models_generalized"
        )

        # Initialize models
        torch.manual_seed(42)
        original_model = OriginalNet(self.num_input, self.num_hidden, self.num_layers,
                                     self.num_dets, self.func, self.pretrain)

        torch.manual_seed(42)
        generalized_model = GeneralizedNet(self.num_input, self.num_hidden, self.num_layers,
                                           self.num_dets, self.func, self.pretrain, self.dim)

        # Run models
        #print('Original input:', input_orig)
        #print('Original swapped input:', input_orig_swapped)
        #print('Generalized input:', input_gen)
        #print('Generalized swapped input:', input_gen_swapped)

        sign_orig, logabs_orig = original_model(input_orig)
        sign_orig_swapped, logabs_orig_swapped = original_model(input_orig_swapped)
        print('Original output:', sign_orig, logabs_orig)
        print('Original swapped output:', sign_orig_swapped, logabs_orig_swapped)

        sign_gen, logabs_gen = generalized_model(input_gen)
        sign_gen_swapped, logabs_gen_swapped = generalized_model(input_gen_swapped)
        print('Generalized output:', sign_gen, logabs_gen)
        print('Generalized swapped output:', sign_gen_swapped, logabs_gen_swapped)

        # Check equivariance
        passed_orig, total, _, _ = check_equivariance_per_walker(
            sign_orig, sign_orig_swapped, logabs_orig, logabs_orig_swapped
        )
        passed_gen, _, _, _ = check_equivariance_per_walker(
            sign_gen, sign_gen_swapped, logabs_gen, logabs_gen_swapped
        )

        print(f"\nOriginal model:    {passed_orig}/{total} walkers passed")
        print(f"Generalized model: {passed_gen}/{total} walkers passed")

        # Optional: strict enforcement
        self.assertEqual(passed_orig, total, "Original model: not all walkers passed")
        self.assertEqual(passed_gen, total, "Generalized model: not all walkers passed")


if __name__ == "__main__":
    unittest.main()
