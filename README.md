# Monte Carlo option pricing with variance reduction, checked against Black-Scholes

Pricing a European option by Monte Carlo comes up a lot in quant interviews, and I've found the follow-up question is nearly always about the noise in your estimate and how you'd probably bring it down without just running more paths. I wrote this to show you the two answers I'd hope to hear from a candidate (antithetic variates and a control variate), and I've checked every estimate here against the Black-Scholes closed form so that you can see roughly how close each one gets.

```python
from mc import bs_price, mc_price

bs_price(100, 105, 0.03, 0.2, 1.0)                                   # 7.1281
mc_price(100, 105, 0.03, 0.2, 1.0, n=200_000, method="plain")        # (price, standard error)
mc_price(100, 105, 0.03, 0.2, 1.0, n=200_000, method="antithetic")
mc_price(100, 105, 0.03, 0.2, 1.0, n=200_000, method="control")
```

When I run `python3 mc.py` it uses 200,000 normal draws with seed 7, and for a one-year call with spot 100 and strike 105 (at a 3% rate and 20% volatility) you should get the numbers in the table below.

| Method | Price | Standard error |
|---|---|---|
| Closed form | 7.1281 | |
| Plain | 7.1609 | 0.0281 |
| Antithetic | 7.1216 | 0.0230 |
| Control variate | 7.1268 | 0.0136 |

With antithetic variates you pair each draw z with its mirror image -z, so a path that finishes high is matched by one that finishes low, and I'd expect the average of each pair to fluctuate rather less than two independent paths would.

The control variate is the discounted stock price at expiry, and you already know its expectation exactly because it's simply the spot price of 100. You can therefore subtract beta times the amount by which your simulated average misses it (with beta estimated from the same paths), which for this particular call roughly halves the standard error.

Since the error shrinks with the square root of the number of paths, I think you'd probably need about four times as many plain paths to achieve the same improvement. Both methods also use exactly the same number of normal draws as the plain estimate, which is why I'd consider the table a reasonably fair comparison.

## Tests

Running `python3 -m unittest test_mc` checks put-call parity and one known value of the closed form first. After that it checks that each method lands within four standard errors of Black-Scholes (for calls and for puts, on three different sets of inputs), that both variance reduction methods beat the plain standard error on calls, and that the same seed gives you the same answer twice. You'll only need the Python standard library to run it.

## More practice

If you're preparing for interviews, I've put stochastic calculus and pricing questions with worked answers at https://academicquant.co.uk/quant-questions-stochastic-calculus?utm_source=github&utm_medium=post&utm_campaign=github-monte-carlo-options

MIT licence.
