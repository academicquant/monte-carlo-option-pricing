"""Monte Carlo prices for European options under Black-Scholes, with plain,
antithetic and control-variate estimators, checked against the closed form."""
import math
import random


def norm_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_price(S, K, r, sigma, T, kind="call"):
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "call":
        return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)


def _payoff(x, K, kind):
    return max(x - K, 0.0) if kind == "call" else max(K - x, 0.0)


def _mean_and_error(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def mc_price(S, K, r, sigma, T, n=100_000, kind="call", method="plain", seed=0):
    """Return (price, standard error) from n draws of the terminal price.

    method="plain"       one payoff per normal draw
    method="antithetic"  n/2 pairs (z, -z), each pair averaged into one sample
    method="control"     discounted terminal price as control, its mean is S
    """
    rng = random.Random(seed)
    drift = (r - 0.5 * sigma * sigma) * T
    vol = sigma * math.sqrt(T)
    disc = math.exp(-r * T)

    if method == "antithetic":
        samples = []
        for _ in range(n // 2):
            z = rng.gauss(0.0, 1.0)
            up = _payoff(S * math.exp(drift + vol * z), K, kind)
            down = _payoff(S * math.exp(drift - vol * z), K, kind)
            samples.append(disc * 0.5 * (up + down))
        return _mean_and_error(samples)

    ys, xs = [], []
    for _ in range(n):
        ST = S * math.exp(drift + vol * rng.gauss(0.0, 1.0))
        ys.append(disc * _payoff(ST, K, kind))
        xs.append(disc * ST)
    if method == "plain":
        return _mean_and_error(ys)
    if method != "control":
        raise ValueError("method must be plain, antithetic or control")
    my, mx = sum(ys) / n, sum(xs) / n
    cov = sum((y - my) * (x - mx) for y, x in zip(ys, xs)) / (n - 1)
    varx = sum((x - mx) ** 2 for x in xs) / (n - 1)
    beta = cov / varx
    return _mean_and_error([y - beta * (x - S) for y, x in zip(ys, xs)])


if __name__ == "__main__":
    S, K, r, sigma, T = 100.0, 105.0, 0.03, 0.2, 1.0
    print(f"closed form call {bs_price(S, K, r, sigma, T):.4f}")
    for m in ("plain", "antithetic", "control"):
        p, se = mc_price(S, K, r, sigma, T, n=200_000, method=m, seed=7)
        print(f"{m:<11} {p:.4f}  standard error {se:.4f}")
