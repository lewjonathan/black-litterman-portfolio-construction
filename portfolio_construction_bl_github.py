"""
Black-Litterman Portfolio Construction Project

Public Template Version

Universe:
    6 assets, intentionally redacted in this public version.

Models:
    1) Minimum Variance
    2) Black-Litterman baseline using equilibrium priors
    3) Black-Litterman with placeholder view structure
    4) Mean-variance allocations across different risk-aversion profiles

Notes:
    - This file is a public research template, not a production trading engine.
    - Asset universe, active views, detailed attribution, and final portfolio outputs
      are intentionally omitted from the public version.
    - Replace ASSET_1 ... ASSET_6 with a private/research universe before running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

try:
    import yfinance as yf
except ImportError as exc:
    raise ImportError("Please install yfinance: pip install yfinance") from exc

try:
    from pandas_datareader import data as web
except ImportError:
    web = None


# =========================
# Configuration
# =========================

@dataclass
class PortfolioConfig:
    """
    Public template configuration.

    The asset universe is intentionally redacted. Replace placeholder assets with
    a private/research universe before running the pipeline.
    """

    tickers: Tuple[str, ...] = (
        "ASSET_1",
        "ASSET_2",
        "ASSET_3",
        "ASSET_4",
        "ASSET_5",
        "ASSET_6",
    )

    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    train_ratio: float = 0.8
    rebalance_freq: str = "M"
    covariance_method: str = "ledoit_wolf"  # "sample" or "ledoit_wolf"

    max_weight: float = 0.35
    min_weight: float = 0.0
    risk_aversion_market: float = 2.5
    tau: float = 0.05
    annual_trading_days: int = 252
    default_rf_annual: float = 0.03
    trustee_risk_aversion: float = 5.0
    kelly_risk_aversion: float = 1.0

    benchmark_weights: Dict[str, float] = field(default_factory=lambda: {
        "ASSET_1": 0.35,
        "ASSET_2": 0.20,
        "ASSET_3": 0.20,
        "ASSET_4": 0.10,
        "ASSET_5": 0.08,
        "ASSET_6": 0.07,
    })


def validate_public_template_config(config: PortfolioConfig) -> None:
    """
    Prevent accidental execution with redacted placeholder tickers.
    """
    if any(str(ticker).startswith("ASSET_") for ticker in config.tickers):
        raise ValueError(
            "Public template uses placeholder tickers. "
            "Replace ASSET_1 ... ASSET_6 with a private/research universe before running."
        )


# =========================
# Data loading
# =========================

def download_prices(
    tickers: Iterable[str],
    start: str,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download adjusted OHLCV price data from Yahoo Finance and return close prices.
    """
    data = yf.download(
        list(tickers),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError("No price data downloaded. Check tickers or date range.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            prices = data.xs("Close", axis=1, level=0, drop_level=True).copy()
    else:
        prices = data.copy()

    return prices.dropna(how="all").ffill().dropna()


def download_risk_free_rate(
    start: str,
    end: Optional[str] = None,
    default_rf_annual: float = 0.03,
) -> pd.Series:
    """
    Download or construct a daily decimal risk-free-rate series.

    Preferred source:
        FRED DGS3MO, if pandas_datareader is available.

    Fallback:
        Constant annual risk-free assumption converted to daily rate.
    """
    if web is not None:
        try:
            rf = web.DataReader("DGS3MO", "fred", start, end)
            rf = rf.rename(columns={"DGS3MO": "rf_annual_pct"}).ffill().dropna()
            rf["rf_daily"] = (rf["rf_annual_pct"] / 100.0) / 252.0
            return rf["rf_daily"]
        except Exception:
            pass

    idx = pd.date_range(start=start, end=end or pd.Timestamp.today(), freq="B")
    return pd.Series(default_rf_annual / 252.0, index=idx, name="rf_daily")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price levels to daily simple returns.
    """
    return prices.pct_change().dropna(how="all")


def align_excess_returns(
    returns: pd.DataFrame,
    rf_daily: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Align risk-free-rate series to asset-return dates and compute excess returns.
    """
    rf_daily = rf_daily.reindex(returns.index).ffill().bfill()
    excess = returns.sub(rf_daily, axis=0)
    return excess, rf_daily


# =========================
# Covariance estimation
# =========================

def estimate_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> pd.DataFrame:
    """
    Estimate covariance using either sample covariance or Ledoit-Wolf shrinkage.
    """
    if method == "sample":
        cov = returns.cov()
    elif method == "ledoit_wolf":
        lw = LedoitWolf().fit(returns.values)
        cov = pd.DataFrame(
            lw.covariance_,
            index=returns.columns,
            columns=returns.columns,
        )
    else:
        raise ValueError(f"Unsupported covariance method: {method}")

    return cov


def annualize_mean_returns(
    daily_returns: pd.Series,
    annual_trading_days: int = 252,
) -> pd.Series:
    """
    Annualize mean daily returns.
    """
    return daily_returns * annual_trading_days


def annualize_covariance(
    cov_daily: pd.DataFrame,
    annual_trading_days: int = 252,
) -> pd.DataFrame:
    """
    Annualize daily covariance matrix.
    """
    return cov_daily * annual_trading_days


# =========================
# Optimization helpers
# =========================

def get_bounds(
    n_assets: int,
    min_weight: float,
    max_weight: float,
) -> Tuple[Tuple[float, float], ...]:
    """
    Construct box constraints for portfolio weights.
    """
    return tuple((min_weight, max_weight) for _ in range(n_assets))


def get_constraints() -> List[Dict]:
    """
    Fully invested portfolio constraint: sum(weights) = 1.
    """
    return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Portfolio volatility from weights and covariance matrix.
    """
    return float(np.sqrt(weights @ cov @ weights))


def portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    """
    Portfolio expected return from weights and expected returns.
    """
    return float(weights @ mu)


def negative_sharpe_ratio(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    rf_annual: float = 0.0,
) -> float:
    """
    Negative Sharpe ratio objective for scipy minimization.
    """
    vol = portfolio_volatility(weights, cov)

    if vol <= 0:
        return 1e9

    ret = portfolio_return(weights, mu)
    return -((ret - rf_annual) / vol)


def min_variance_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """
    Portfolio variance objective.
    """
    return float(weights @ cov @ weights)


def negative_mean_variance_utility(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
) -> float:
    """
    Negative mean-variance utility.

    Maximized utility:
        mu'w - (lambda / 2) * w'Sigma w
    """
    utility = weights @ mu - 0.5 * risk_aversion * (weights @ cov @ weights)
    return -float(utility)


def optimize_min_variance(
    cov: pd.DataFrame,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Solve constrained minimum-variance portfolio.
    """
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)

    result = minimize(
        fun=min_variance_objective,
        x0=x0,
        args=(cov.values,),
        method="SLSQP",
        bounds=get_bounds(n, min_weight, max_weight),
        constraints=get_constraints(),
        options={"maxiter": 300, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"Minimum-variance optimization failed: {result.message}")

    return pd.Series(result.x, index=cov.index, name="w_min_var")


def optimize_max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf_annual: float = 0.0,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    """
    Solve constrained maximum-Sharpe portfolio.
    """
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)

    result = minimize(
        fun=negative_sharpe_ratio,
        x0=x0,
        args=(mu.values, cov.values, rf_annual),
        method="SLSQP",
        bounds=get_bounds(n, min_weight, max_weight),
        constraints=get_constraints(),
        options={"maxiter": 300, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"Maximum-Sharpe optimization failed: {result.message}")

    return pd.Series(result.x, index=cov.index, name="w_max_sharpe")


def optimize_mean_variance(
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    name: str = "w_mv",
) -> pd.Series:
    """
    Solve constrained mean-variance utility portfolio.
    """
    n = cov.shape[0]
    x0 = np.repeat(1.0 / n, n)

    result = minimize(
        fun=negative_mean_variance_utility,
        x0=x0,
        args=(mu.values, cov.values, risk_aversion),
        method="SLSQP",
        bounds=get_bounds(n, min_weight, max_weight),
        constraints=get_constraints(),
        options={"maxiter": 300, "ftol": 1e-9},
    )

    if not result.success:
        raise RuntimeError(f"Mean-variance optimization failed: {result.message}")

    return pd.Series(result.x, index=cov.index, name=name)


# =========================
# Black-Litterman
# =========================

def reverse_optimization(
    cov_annual: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
) -> pd.Series:
    """
    Reverse optimization to infer equilibrium expected returns.

    pi = lambda * Sigma * w_market
    """
    aligned_weights = market_weights.reindex(cov_annual.index)

    if aligned_weights.isna().any():
        raise ValueError("Market weights must contain all assets in covariance matrix.")

    pi = risk_aversion * cov_annual.values @ aligned_weights.values

    return pd.Series(pi, index=cov_annual.index, name="pi_equilibrium")


def black_litterman_posterior(
    cov_annual: pd.DataFrame,
    pi: pd.Series,
    P: pd.DataFrame,
    Q: pd.Series,
    omega: pd.DataFrame,
    tau: float = 0.05,
) -> pd.Series:
    """
    Compute Black-Litterman posterior expected returns.
    """
    sigma = cov_annual.values
    tau_sigma = tau * sigma
    inv_tau_sigma = np.linalg.inv(tau_sigma)

    Pm = P.values
    omega_inv = np.linalg.inv(omega.values)

    middle = inv_tau_sigma + Pm.T @ omega_inv @ Pm
    rhs = inv_tau_sigma @ pi.values + Pm.T @ omega_inv @ Q.values

    mu_bl = np.linalg.inv(middle) @ rhs

    return pd.Series(mu_bl, index=cov_annual.index, name="mu_bl")


def simple_bl_omega(
    P: pd.DataFrame,
    cov_annual: pd.DataFrame,
    tau: float,
) -> pd.DataFrame:
    """
    Construct diagonal Black-Litterman omega matrix using projected uncertainty.
    """
    projected = tau * P.values @ cov_annual.values @ P.values.T

    return pd.DataFrame(
        np.diag(np.diag(projected)),
        index=P.index,
        columns=P.index,
    )


def idzorek_like_omega(
    P: pd.DataFrame,
    cov_annual: pd.DataFrame,
    tau: float,
    confidences: Iterable[float],
) -> pd.DataFrame:
    """
    Optional confidence-scaled omega construction.

    Included as a research extension but not used in the public main pipeline.
    """
    sigma = cov_annual.values
    Pm = P.values
    conf = np.array(list(confidences), dtype=float)

    if np.any(conf <= 0) or np.any(conf > 1):
        raise ValueError("Confidences must be in (0, 1].")

    base = np.diag(Pm @ (tau * sigma) @ Pm.T)
    diag = base * (1.0 - conf) / conf

    return pd.DataFrame(np.diag(diag), index=P.index, columns=P.index)


# =========================
# Backtest
# =========================

def split_train_test(
    returns: pd.DataFrame,
    train_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered train/test split.
    """
    split_idx = int(len(returns) * train_ratio)

    return returns.iloc[:split_idx].copy(), returns.iloc[split_idx:].copy()


def run_static_backtest(
    test_returns: pd.DataFrame,
    weights: pd.Series,
    name: str,
) -> pd.Series:
    """
    Static out-of-sample portfolio backtest using fixed weights.
    """
    aligned_weights = weights.reindex(test_returns.columns)
    port = test_returns @ aligned_weights
    port.name = name

    return port


def performance_summary(
    returns: pd.Series,
    rf_daily: Optional[pd.Series] = None,
    annual_trading_days: int = 252,
) -> pd.Series:
    """
    Compute basic performance diagnostics.

    Detailed outputs are not printed by default in the public script.
    """
    rf_aligned = pd.Series(0.0, index=returns.index)

    if rf_daily is not None:
        rf_aligned = rf_daily.reindex(returns.index).ffill().fillna(0.0)

    excess = returns - rf_aligned

    cum = (1.0 + returns).cumprod()
    total_return = cum.iloc[-1] - 1.0
    ann_return = (1.0 + total_return) ** (annual_trading_days / len(returns)) - 1.0
    ann_vol = returns.std() * np.sqrt(annual_trading_days)
    sharpe = np.nan if ann_vol == 0 else (excess.mean() * annual_trading_days) / ann_vol

    running_max = cum.cummax()
    drawdown = cum / running_max - 1.0
    max_dd = drawdown.min()

    return pd.Series(
        {
            "Total Return": total_return,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
        }
    )


# =========================
# Prior and views helpers
# =========================

def example_market_weights(
    tickers: Iterable[str],
    benchmark_weights: Optional[Dict[str, float]] = None,
) -> pd.Series:
    """
    Construct benchmark/market weights.

    In the public version, placeholder benchmark weights are provided only to show
    the structure of the prior-construction process.
    """
    if benchmark_weights is None:
        benchmark_weights = {
            "ASSET_1": 0.35,
            "ASSET_2": 0.20,
            "ASSET_3": 0.20,
            "ASSET_4": 0.10,
            "ASSET_5": 0.08,
            "ASSET_6": 0.07,
        }

    s = pd.Series(benchmark_weights).reindex(list(tickers)).fillna(0.0)

    if s.sum() == 0:
        raise ValueError("Benchmark weights sum to zero. Check ticker alignment.")

    return s / s.sum()


def build_manual_views(
    asset_names: List[str],
    manual_view_specs: List[Dict],
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build Black-Litterman P and Q matrices from manual view specifications.

    Example:
        [
            {"name": "view1", "long": "ASSET_A", "short": "ASSET_B", "q": 0.02},
            {"name": "view2", "long": "ASSET_C", "q": 0.01},
        ]

    In the public version, actual view specifications are omitted.
    """
    P = pd.DataFrame(
        0.0,
        index=[v["name"] for v in manual_view_specs],
        columns=asset_names,
    )

    Q = pd.Series(index=P.index, dtype=float, name="Q")

    for v in manual_view_specs:
        nm = v["name"]
        long_asset = v["long"]
        short_asset = v.get("short")
        q = float(v["q"])

        if long_asset not in asset_names:
            raise ValueError(f"{long_asset} missing from asset universe.")

        P.loc[nm, long_asset] = 1.0

        if short_asset is not None:
            if short_asset not in asset_names:
                raise ValueError(f"{short_asset} missing from asset universe.")
            P.loc[nm, short_asset] = -1.0

        Q.loc[nm] = q

    return P, Q


def default_manual_view_specs() -> List[Dict]:
    """
    Public version intentionally omits active view construction.
    """
    return []


def build_bl_view_model(
    cov_annual: pd.DataFrame,
    pi_prior: pd.Series,
    tau: float,
    manual_view_specs: Optional[List[Dict]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Build Black-Litterman view-adjusted expected returns.

    If views are omitted, return the prior as the posterior. This keeps the
    public template structurally complete without exposing private views.
    """
    asset_names = list(cov_annual.columns)
    specs = default_manual_view_specs() if manual_view_specs is None else manual_view_specs

    if not specs:
        empty_P = pd.DataFrame(columns=asset_names, dtype=float)
        empty_Q = pd.Series(dtype=float, name="Q")
        empty_omega = pd.DataFrame(dtype=float)
        return empty_P, empty_Q, empty_omega, pi_prior.copy()

    P, Q = build_manual_views(asset_names, specs)
    omega = simple_bl_omega(P, cov_annual, tau)
    mu_bl_views = black_litterman_posterior(cov_annual, pi_prior, P, Q, omega, tau)

    return P, Q, omega, mu_bl_views


# =========================
# Main project pipeline
# =========================

def run_project(
    config: PortfolioConfig,
    manual_view_specs: Optional[List[Dict]] = None,
) -> Dict[str, object]:
    """
    Run the portfolio construction pipeline.

    Public template will raise an error until placeholder tickers are replaced.
    """
    validate_public_template_config(config)

    prices = download_prices(config.tickers, config.start_date, config.end_date)
    returns = compute_returns(prices)

    rf_daily = download_risk_free_rate(
        config.start_date,
        config.end_date,
        config.default_rf_annual,
    )

    excess_returns, rf_daily = align_excess_returns(returns, rf_daily)

    train, test = split_train_test(returns, config.train_ratio)
    train_excess = excess_returns.loc[train.index]

    cov_daily = estimate_covariance(train, config.covariance_method)
    cov_annual = annualize_covariance(cov_daily, config.annual_trading_days)

    rf_annual_train = float(rf_daily.loc[train.index].mean() * config.annual_trading_days)

    # Baseline: Minimum Variance
    w_min_var = optimize_min_variance(
        cov_annual,
        config.min_weight,
        config.max_weight,
    )
    r_min_var = run_static_backtest(test, w_min_var, "MinVariance")

    # Prior: equilibrium baseline
    market_weights = example_market_weights(config.tickers, config.benchmark_weights)
    pi_equilibrium = reverse_optimization(
        cov_annual,
        market_weights,
        config.risk_aversion_market,
    )

    pi_prior = pi_equilibrium.copy()

    # Black-Litterman baseline with no active views
    mu_bl = pi_equilibrium.copy()
    w_bl = optimize_max_sharpe(
        mu_bl,
        cov_annual,
        rf_annual_train,
        config.min_weight,
        config.max_weight,
    )
    r_bl = run_static_backtest(test, w_bl, "BlackLitterman")

    # Black-Litterman with omitted/private views
    P_views, Q_views, omega_views, mu_bl_views = build_bl_view_model(
        cov_annual=cov_annual,
        pi_prior=pi_prior,
        tau=config.tau,
        manual_view_specs=manual_view_specs,
    )

    w_bl_views = optimize_max_sharpe(
        mu_bl_views,
        cov_annual,
        rf_annual_train,
        config.min_weight,
        config.max_weight,
    )
    r_bl_views = run_static_backtest(test, w_bl_views, "BL_Views")

    # Risk-aversion profile allocations
    profile_lambdas = {
        "trustee": config.trustee_risk_aversion,
        "market": config.risk_aversion_market,
        "kelly": config.kelly_risk_aversion,
    }

    mv_profile_weights = {
        profile: optimize_mean_variance(
            mu=mu_bl_views,
            cov=cov_annual,
            risk_aversion=lam,
            min_weight=config.min_weight,
            max_weight=config.max_weight,
            name=f"w_{profile}",
        )
        for profile, lam in profile_lambdas.items()
    }

    mv_profile_returns = {
        profile: run_static_backtest(test, weights, f"BL_Views_{profile.title()}")
        for profile, weights in mv_profile_weights.items()
    }

    # Generic benchmark: first asset in the private/research universe
    benchmark_name = test.columns[0]
    benchmark = test[benchmark_name].rename("Benchmark")

    summary = pd.DataFrame(
        {
            "MinVariance": performance_summary(r_min_var, rf_daily),
            "BlackLitterman": performance_summary(r_bl, rf_daily),
            "BL_Views": performance_summary(r_bl_views, rf_daily),
            "Benchmark": performance_summary(benchmark, rf_daily),
            "BL_Views_Trustee": performance_summary(mv_profile_returns["trustee"], rf_daily),
            "BL_Views_Market": performance_summary(mv_profile_returns["market"], rf_daily),
            "BL_Views_Kelly": performance_summary(mv_profile_returns["kelly"], rf_daily),
        }
    )

    cumulative = pd.concat(
        [
            (1 + r_min_var).cumprod().rename("MinVariance"),
            (1 + r_bl).cumprod().rename("BlackLitterman"),
            (1 + r_bl_views).cumprod().rename("BL_Views"),
            (1 + benchmark).cumprod().rename("Benchmark"),
            (1 + mv_profile_returns["trustee"]).cumprod().rename("BL_Views_Trustee"),
            (1 + mv_profile_returns["market"]).cumprod().rename("BL_Views_Market"),
            (1 + mv_profile_returns["kelly"]).cumprod().rename("BL_Views_Kelly"),
        ],
        axis=1,
    )

    return {
        "prices": prices,
        "returns": returns,
        "rf_daily": rf_daily,
        "train": train,
        "test": test,
        "train_excess": train_excess,
        "cov_annual": cov_annual,
        "market_weights": market_weights,
        "pi_equilibrium": pi_equilibrium,
        "pi_prior": pi_prior,
        "mu_bl": mu_bl,
        "mu_bl_views": mu_bl_views,
        "w_min_var": w_min_var,
        "w_bl": w_bl,
        "w_bl_views": w_bl_views,
        "P_views": P_views,
        "Q_views": Q_views,
        "omega_views": omega_views,
        "mv_profile_weights": mv_profile_weights,
        "summary": summary,
        "cumulative": cumulative,
    }


if __name__ == "__main__":
    cfg = PortfolioConfig()

    print("\nPublic template loaded.")
    print("Asset universe is intentionally redacted.")
    print("Replace placeholder tickers with a private/research universe before running the pipeline.")
    print("\nDetailed portfolio weights, active views, attribution outputs, and result tables are omitted from the public version.")

    # Uncomment after replacing placeholder assets:
    #
    # results = run_project(cfg)
    # print("\nPipeline completed.")
    # print("Available result keys:")
    # print(list(results.keys()))