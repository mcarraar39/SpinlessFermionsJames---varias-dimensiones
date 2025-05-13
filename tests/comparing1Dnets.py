import unittest
import torch
import torch.nn as nn
import importlib.util
import sys
import os


def load_module_with_dependencies(model_path: str, class_name: str, module_alias: str):
    """
    Dynamically load `Models.py` and its entire dependency tree (e.g., Layers, Functions, etc.)
    from a specific `src` folder, and isolate it from other versions.
    """
    src_dir = os.path.dirname(model_path)

    # Create a fake isolated module namespace
    original_sys_modules = sys.modules.copy()
    sys.path.insert(0, src_dir)

    try:
        spec = importlib.util.spec_from_file_location(module_alias, model_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {model_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_alias] = mod  # allow internal relative imports
        spec.loader.exec_module(mod)
        return getattr(mod, class_name)

    finally:
        # Clean up: restore sys.modules and sys.path to original state
        sys.modules.clear()
        sys.modules.update(original_sys_modules)
        sys.path.pop(0)



class TestCompareVLogHarmonicNet1D(unittest.TestCase):

    def setUp(self):
        self.num_input = 4
        self.num_hidden = 8
        self.num_layers = 2
        self.num_dets = 1
        self.func = nn.ReLU()
        self.pretrain = False
        self.dim = 1
        self.input_tensor = torch.randn(self.num_input, self.dim)

        self.original_model_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames\src\Models.py"
        self.generalized_model_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Models.py"

    def test_compare_outputs(self):
        # Load vLogHarmonicNet class from each models.py
        OriginalNet = load_module_with_dependencies(
            self.original_model_path, "vLogHarmonicNet", "models_original"
        )

        GeneralizedNet = load_module_with_dependencies(
            self.generalized_model_path, "vLogHarmonicNet", "models_generalized"
        )

        # Initialize models
        nwalkers = 5
        nfermions = self.num_input
        dim = self.dim

        # Input for generalized model: [nwalkers, nfermions, dim]
        input_gen = torch.randn(nwalkers, nfermions, dim)

        # Input for original model: [nwalkers, nfermions]
        input_orig = input_gen.squeeze(-1) if dim == 1 else None  # only valid for dim=1

        torch.manual_seed(42)
        original_model = OriginalNet(self.num_input, self.num_hidden, self.num_layers,
                                     self.num_dets, self.func, self.pretrain)

        torch.manual_seed(42)
        generalized_model = GeneralizedNet(self.num_input, self.num_hidden, self.num_layers,
                                           self.num_dets, self.func, self.pretrain, self.dim)

        # Run both models
        #print('The input of the original model is', input_orig)
        #print('The input of the generalized model is', input_gen)

        print('Input shape for original model:', input_orig.shape if input_orig is not None else "N/A")
        print('Input shape for generalized model:', input_gen.shape)


        print('Loading original model')
        sign_orig, logabs_orig = original_model(input_orig)
        print('Loading generalized model')
        sign_gen, logabs_gen = generalized_model(input_gen)

        

        print('Original model output:', sign_orig, logabs_orig)
        print('Generalized model output:', sign_gen, logabs_gen)
        # Compare outputs
        self.assertTrue(torch.allclose(sign_orig, sign_gen, atol=1e-5), "Sign outputs are not close")
        self.assertTrue(torch.allclose(logabs_orig, logabs_gen, atol=1e-5), "Logabs outputs are not close")

if __name__ == "__main__":
    unittest.main()
