import math
import unittest

from mc import bs_price, mc_price

CASES = [
    (100.0, 105.0, 0.03, 0.2, 1.0),
    (100.0, 80.0, 0.01, 0.35, 0.5),
    (50.0, 60.0, 0.05, 0.15, 2.0),
]


class TestClosedForm(unittest.TestCase):
    def test_put_call_parity(self):
        for S, K, r, sigma, T in CASES:
            c = bs_price(S, K, r, sigma, T, "call")
            p = bs_price(S, K, r, sigma, T, "put")
            self.assertAlmostEqual(c - p, S - K * math.exp(-r * T), places=10)

    def test_known_value(self):
        # S = K = 100, r = 0, sigma = 0.2, T = 1: 100 * (2 N(0.1) - 1)
        self.assertAlmostEqual(bs_price(100, 100, 0.0, 0.2, 1.0), 7.965567455405804, places=10)


class TestMonteCarlo(unittest.TestCase):
    def test_every_method_within_four_standard_errors(self):
        for S, K, r, sigma, T in CASES:
            for kind in ("call", "put"):
                exact = bs_price(S, K, r, sigma, T, kind)
                for method in ("plain", "antithetic", "control"):
                    p, se = mc_price(S, K, r, sigma, T, n=40_000, kind=kind, method=method, seed=11)
                    self.assertLess(abs(p - exact), 4 * se + 1e-12, (S, K, kind, method, p, exact, se))

    def test_variance_reduction_helps_for_calls(self):
        for S, K, r, sigma, T in CASES:
            _, plain = mc_price(S, K, r, sigma, T, n=40_000, method="plain", seed=3)
            _, anti = mc_price(S, K, r, sigma, T, n=40_000, method="antithetic", seed=3)
            _, ctrl = mc_price(S, K, r, sigma, T, n=40_000, method="control", seed=3)
            self.assertLess(anti, plain)
            self.assertLess(ctrl, plain)

    def test_same_seed_same_answer(self):
        a = mc_price(100, 100, 0.02, 0.25, 1.0, n=2_000, seed=5)
        b = mc_price(100, 100, 0.02, 0.25, 1.0, n=2_000, seed=5)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
