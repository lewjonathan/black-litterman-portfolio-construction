# Black–Litterman Portfolio Construction

This project explores portfolio construction in a small multi-asset setting using constrained optimization and Black–Litterman expected return updates.

## Results Summary

Out-of-sample evaluation (mid-2024 to end-2025) highlights several key observations:

- Black–Litterman with equilibrium priors alone largely reproduces benchmark-like exposures and does not generate excess returns
- Minimum variance allocation delivered the strongest risk-adjusted performance (Sharpe ~1.49 vs ~0.83 for SPY), highlighting the impact of risk organization
- Introducing a small number of structured macro/factor views meaningfully shifted portfolio outcomes
- Scaling view conviction increased total return (up to ~48% vs ~32% for SPY), while maintaining comparable or lower volatility (~13–14% vs ~17%)

## Interpretation

- Black–Litterman functions as a portfolio construction lens rather than an alpha-generating model
- Priors act as a stabilizing anchor, preventing extreme allocations but not contributing to excess return
- Performance improvements are driven by the specification and scaling of views
- Portfolio construction choices (constraints, risk targeting, scaling) materially influence outcomes

## Caveats

- Results are based on a small asset universe and limited number of views
- View specification is partially data-informed and may not generalize across regimes
- No claim of persistent alpha — results are conditional on the chosen inputs and period

## What is included

- Minimum variance allocation
- Black–Litterman baseline using equilibrium priors
- Placeholder structure for relative views
- Mean-variance allocations across different risk-aversion settings
- Out-of-sample comparison against a benchmark

## Research focus

The goal of this project is to compare how different portfolio construction methods behave under the same constrained allocation framework.

The implementation emphasizes:

- constrained allocation
- covariance shrinkage
- prior-based expected returns
- out-of-sample evaluation
- comparison across portfolio construction approaches

## Notes

This repository is intentionally limited to the portfolio construction layer.

More advanced view construction and attribution extensions are omitted.

## File

- `portfolio_construction_bl_github.py` — core research pipeline

Note: This repository contains a public redacted version. Exact implementation details, private parameters, diagnostic plots, and full outputs are omitted. The focus is on research workflow, validation structure, and interpretation.
