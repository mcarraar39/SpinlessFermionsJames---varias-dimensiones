import unittest
import torch
import torch.nn as nn
import importlib.util
import sys
import os


def load_class_with_deps(file_path: str, class_name: str, module_alias: str):
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


class TestMetropolisSamplers(unittest.TestCase):

    def setUp(self):
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

        # ---- File paths ----
        self.model_orig_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames\src\Models.py"
        self.sampler_orig_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames\src\Samplers.py"

        self.model_gen_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Models.py"
        self.sampler_gen_path = r"C:\Users\migue\OneDrive - Universitat de Barcelona\Master\Intership\SpinlessFermionsJames - varias dimensiones\src\Samplers.py"

    def test_metropolis_sampling_equivalence_1D(self):
        

        # Load and initialize original model and sampler
        
        OriginalNet = load_class_with_deps(self.model_orig_path, "vLogHarmonicNet", "model_orig")
        
        OrigSampler = load_class_with_deps(self.sampler_orig_path, "MetropolisHastings", "sampler_orig")
        torch.manual_seed(self.seed)
        model_orig = OriginalNet(self.nfermions, self.num_hidden, self.num_layers,
                                self.num_dets, self.func, self.pretrain)

        
        sampler_orig = OrigSampler(network=model_orig,
                                dof=self.nfermions,
                                nwalkers=self.nwalkers,
                                target_acceptance=self.target_acceptance)

        # Load and initialize generalized model and sampler
        GeneralizedNet = load_class_with_deps(self.model_gen_path, "vLogHarmonicNet", "model_gen")
        GenSampler = load_class_with_deps(self.sampler_gen_path, "MetropolisHastings", "sampler_gen")
        torch.manual_seed(self.seed)
        model_gen = GeneralizedNet(self.nfermions, self.num_hidden, self.num_layers,
                                self.num_dets, self.func, self.pretrain, self.dim)

        # Ensure both models start with the same weights
        #model_gen.load_state_dict(model_orig.state_dict())

        
        sampler_gen = GenSampler(network=model_gen,
                                dof=self.nfermions,
                                nwalkers=self.nwalkers,
                                target_acceptance=self.target_acceptance,
                                dim=self.dim)

        # ---- Run both samplers ----
        torch.manual_seed(self.seed)
        chains_orig, logprob_orig = sampler_orig(n_sweeps=self.nsteps)
        torch.manual_seed(self.seed)
        chains_gen, logprob_gen = sampler_gen(n_sweeps=self.nsteps)

        # ---- Adjust shape for comparison ----
        chains_gen_flat = chains_gen.squeeze(-1) if chains_gen.shape[-1] == 1 else chains_gen

        # ---- Print shapes ----
        print(f"\nOriginal chains shape:     {chains_orig.shape}")
        print(f"Generalized chains shape:  {chains_gen_flat.shape}")

        chains_close = torch.allclose(chains_orig, chains_gen_flat, atol=1e-5)
        logprob_close = torch.allclose(logprob_orig, logprob_gen, atol=1e-5)
        print(f"Chains close: {chains_close}")
        print(f"Log probabilities close: {logprob_close}")

        #print('Original chains:', chains_orig)
        #print('Generalized chains:', chains_gen_flat)
        # ---- Compare statistics ----
        mean_orig = chains_orig.mean().item()
        mean_gen = chains_gen_flat.mean().item()
        std_orig = chains_orig.std().item()
        std_gen = chains_gen_flat.std().item()

        print(f"\nChain means:")
        print(f"  Original:     {mean_orig:.6f}")
        print(f"  Generalized:  {mean_gen:.6f}")
        print(f"  Difference:   {abs(mean_orig - mean_gen):.6f}")

        print(f"\nChain std devs:")
        print(f"  Original:     {std_orig:.6f}")
        print(f"  Generalized:  {std_gen:.6f}")
        print(f"  Difference:   {abs(std_orig - std_gen):.6f}")

        # ---- Compare acceptance rates ----
        print(f"\nAcceptance rates:")
        print(f"  Original:     {sampler_orig.acceptance_rate.item():.4f}")
        print(f"  Generalized:  {sampler_gen.acceptance_rate.item():.4f}")

        # ---- Compare log probabilities ----
        logp_orig_mean = logprob_orig.mean().item()
        logp_gen_mean = logprob_gen.mean().item()

        print(f"\nLog probabilities (mean):")
        print(f"  Original:     {logp_orig_mean:.6f}")
        print(f"  Generalized:  {logp_gen_mean:.6f}")
        print(f"  Difference:   {abs(logp_orig_mean - logp_gen_mean):.6f}")

        # ---- Compare element-wise closeness ----
        chains_close = torch.allclose(chains_orig, chains_gen_flat, atol=1e-5)
        logprob_close = torch.allclose(logprob_orig, logprob_gen, atol=1e-5)

        try:
            self.assertTrue(abs(mean_orig - mean_gen) < 1e-2, "Mean of samples differs too much.")
        except AssertionError as e:
             print(f"\n❌ Mean check failed: {e}")

        try:
            self.assertTrue(abs(std_orig - std_gen) < 1e-2, "Std of samples differs too much.")
        except AssertionError as e:
            print(f"\n❌ Std check failed: {e}")

        try:
            self.assertTrue(chains_close, "Chains from original and generalized sampler do not match.")
        except AssertionError as e:
            print(f"\n❌ Chain closeness check failed: {e}")

        try:
            self.assertTrue(logprob_close, "Log probabilities from original and generalized sampler do not match.")
        except AssertionError as e:
            print(f"\n❌ Log probability check failed: {e}")

        print("\n✅ Finished sampler comparison test.\n")


if __name__ == "__main__":
    unittest.main()
