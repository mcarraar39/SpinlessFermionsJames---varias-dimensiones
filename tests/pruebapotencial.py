import unittest
import torch


# ---------- Función a testear ----------
def potential(x: torch.Tensor) -> torch.Tensor:
    """V(x) = ½ |x|²  (suma sobre partículas y dimensiones)"""
    return  0.5 * x.pow(2).view(x.size(0), -1).sum(dim=1)  # -> [nwalkers]


class TestHarmonicPotential(unittest.TestCase):

    nwalkers, nfermions = 128, 2

    # ---------- util ----------
    @staticmethod
    def is_vector(t: torch.Tensor, length: int):
        return t.dim() == 1 and t.numel() == length

    # ---------- tests ----------
    def test_shape_1D(self):
        x = torch.randn(self.nwalkers, self.nfermions)          # [B, A]
        v = potential(x)
        self.assertTrue(self.is_vector(v, self.nwalkers))

    def test_shape_2D(self):
        x = torch.randn(self.nwalkers, self.nfermions, 2)       # [B, A, D]
        v = potential(x)
        self.assertTrue(self.is_vector(v, self.nwalkers))

    def test_mean_value(self):
        # valor exacto esperado  = D·A/2
        for D in (1, 2, 3):
            x = torch.randn(self.nwalkers, self.nfermions, D) if D > 1 \
                else torch.randn(self.nwalkers, self.nfermions)
            v = potential(x)
            expected = D * self.nfermions / 2
            self.assertAlmostEqual(v.mean().item(), expected, delta=expected*0.5)


if __name__ == "__main__":
    unittest.main()
