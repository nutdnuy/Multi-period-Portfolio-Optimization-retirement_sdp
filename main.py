#!/usr/bin/env python3
"""Retirement SDP — CLI entry point.

Usage
-----
    python main.py [--config config/config.yaml] [--skip-fetch] [--fast]

Options
-------
--config   Path to config YAML (default: config/config.yaml)
--skip-fetch  Use only cached data; raise error if cache is missing
--fast     Reduce grid/samples for quick debugging (overrides config)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("retirement_sdp.main")


# ── Project root on sys.path ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from retirement_sdp.data.alpha_vantage import AlphaVantageClient
from retirement_sdp.data.cache import ParquetCache
from retirement_sdp.data.preprocessor import preprocess, monthly_prices_to_series
from retirement_sdp.models.return_model import LogNormalReturnModel
from retirement_sdp.models.inflation import InflationModel
from retirement_sdp.models.cashflow import build_schedule
from retirement_sdp.sdp.state_space import build_wealth_grid
from retirement_sdp.sdp.action_space import build_efficient_frontier
from retirement_sdp.sdp.bellman import bellman_backward
from retirement_sdp.sdp.policy import forward_simulate
from retirement_sdp.analysis.pdf_propagation import propagate_pdfs
from retirement_sdp.analysis.ruin import compute_ruin
from retirement_sdp.analysis.swr import compute_swr
from retirement_sdp.visualization.efficient_frontier import plot_efficient_frontier
from retirement_sdp.visualization.wealth_pdf import plot_wealth_pdfs
from retirement_sdp.visualization.ruin_curve import plot_ruin_curve


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def fetch_or_load_prices(
    cfg: dict,
    tickers: list[str],
    cache: ParquetCache,
    skip_fetch: bool,
) -> dict[str, dict]:
    """Return dict {ticker: raw monthly-adjusted data dict}."""
    av_cfg = cfg["alpha_vantage"]
    client = AlphaVantageClient(
        api_key=av_cfg["api_key"],
        base_url=av_cfg.get("base_url", "https://www.alphavantage.co/query"),
    )
    raw_prices: dict[str, dict] = {}
    for ticker in tickers:
        key = f"{ticker}_monthly"
        if skip_fetch and not cache.exists(key):
            raise FileNotFoundError(
                f"--skip-fetch requested but no cache for {ticker}. "
                "Run once without --skip-fetch to populate the cache."
            )
        raw_prices[ticker] = cache.load_or_fetch(
            key,
            lambda t=ticker: client.get_monthly_adjusted(t),
        ).to_dict()  # if cached as DataFrame, convert back to dict
    return raw_prices


def fetch_or_load_cpi(
    cfg: dict,
    cache: ParquetCache,
    skip_fetch: bool,
) -> dict[str, str]:
    """Return CPI data or empty dict on failure."""
    av_cfg = cfg["alpha_vantage"]
    assets_cfg = cfg.get("assets", {})
    cpi_series = assets_cfg.get("cpi_series", "CPIAUCSL")
    client = AlphaVantageClient(
        api_key=av_cfg["api_key"],
        base_url=av_cfg.get("base_url", "https://www.alphavantage.co/query"),
    )
    key = f"cpi_{cpi_series}"
    if skip_fetch:
        if cache.exists(key):
            df = cache.load(key)
            return dict(zip(df.index.astype(str), df["value"].astype(str)))
        else:
            logger.warning("CPI cache miss with --skip-fetch; using fixed inflation rate.")
            return {}
    try:
        cpi_raw = client.get_cpi(cpi_series)
        if cpi_raw:
            import pandas as pd
            df = pd.DataFrame(list(cpi_raw.items()), columns=["date", "value"])
            df = df.set_index("date")
            cache.save(key, df)
        return cpi_raw
    except Exception as exc:
        logger.warning("CPI fetch failed: %s — using fixed inflation rate.", exc)
        return {}


def _prices_from_cache_or_api(
    raw_prices: dict[str, dict],
    tickers: list[str],
    cache: ParquetCache,
    av_cfg: dict,
    skip_fetch: bool,
) -> dict[str, dict]:
    """Helper that handles cached-DataFrame → dict conversion."""
    import pandas as pd
    client = AlphaVantageClient(
        api_key=av_cfg["api_key"],
        base_url=av_cfg.get("base_url", "https://www.alphavantage.co/query"),
    )
    result: dict[str, dict] = {}
    for ticker in tickers:
        key = f"{ticker}_monthly"
        if skip_fetch and not cache.exists(key):
            raise FileNotFoundError(f"No cache for {ticker} and --skip-fetch was requested.")

        if cache.exists(key):
            df = cache.load(key)
            # DataFrame has DatetimeIndex + column "adj_close" → reconstruct raw dict
            result[ticker] = {
                str(idx.date()): {"5. adjusted close": str(v)}
                for idx, v in zip(df.index, df["adj_close"])
            }
        else:
            logger.info("Fetching %s from Alpha Vantage...", ticker)
            raw = client.get_monthly_adjusted(ticker)
            # Cache as DataFrame
            s = monthly_prices_to_series(raw)
            df = s.to_frame("adj_close")
            cache.save(key, df)
            result[ticker] = raw
    return result


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(config_path: str, skip_fetch: bool = False, fast: bool = False) -> None:
    t_start = time.time()

    # 1. Load config ────────────────────────────────────────────────────────────
    cfg = load_config(config_path)
    if cfg["alpha_vantage"]["api_key"] in ("YOUR_API_KEY_HERE", "", None):
        logger.error(
            "No Alpha Vantage API key set in %s. "
            "Copy config/config.example.yaml → config/config.yaml and add your key.",
            config_path,
        )
        sys.exit(1)

    output_cfg = cfg.get("output", {})
    cache_dir = output_cfg.get("cache_dir", "outputs/cache")
    figures_dir = output_cfg.get("figures_dir", "outputs/figures")
    dpi = output_cfg.get("figure_dpi", 150)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    sdp_cfg = cfg.get("sdp", {})
    analysis_cfg = cfg.get("analysis", {})
    cf_cfg = cfg.get("cashflow", {})
    assets_cfg = cfg.get("assets", {})
    tickers = assets_cfg.get("tickers", ["SPY", "AGG", "GLD"])
    phases = cfg.get("phases", [])

    # Fast-mode overrides
    if fast:
        sdp_cfg["wealth_grid_points"] = 100
        sdp_cfg["num_portfolios"] = 20
        sdp_cfg["mc_samples"] = 200
        analysis_cfg["num_paths"] = 1000
        logger.info("Fast mode: reduced grid/samples for quick run.")

    # 2. Data ────────────────────────────────────────────────────────────────────
    cache = ParquetCache(cache_dir)
    av_cfg = cfg["alpha_vantage"]

    raw_prices = _prices_from_cache_or_api(
        raw_prices={}, tickers=tickers, cache=cache,
        av_cfg=av_cfg, skip_fetch=skip_fetch,
    )
    cpi_raw = fetch_or_load_cpi(cfg, cache, skip_fetch)

    # 3. Preprocessing ──────────────────────────────────────────────────────────
    mu, Sigma, tickers, returns_df = preprocess(raw_prices, tickers)
    return_model = LogNormalReturnModel(mu, Sigma, tickers)
    logger.info("%s", return_model)

    # 4. Inflation & cash flows ─────────────────────────────────────────────────
    inflation = InflationModel(
        cpi_data=cpi_raw,
        fallback_rate=cf_cfg.get("inflation_rate", 0.03),
    )
    schedule = build_schedule(
        phases=phases,
        annual_contribution=cf_cfg.get("annual_contribution", 20_000),
        annual_withdrawal=cf_cfg.get("annual_withdrawal", 50_000),
        inflation_model=inflation,
    )
    T = schedule.total_periods
    cash_flows = schedule.cash_flows
    phase_labels = schedule.phase_labels
    logger.info("Total periods: %d", T)

    # 5. State & action spaces ──────────────────────────────────────────────────
    n_W = sdp_cfg.get("wealth_grid_points", 500)
    w_min = sdp_cfg.get("wealth_min", 1_000)
    w_max = sdp_cfg.get("wealth_max", 10_000_000)
    grid = build_wealth_grid(w_min, w_max, n_W)

    n_A = sdp_cfg.get("num_portfolios", 100)
    weights, frontier_mu, frontier_sigma = build_efficient_frontier(mu, Sigma, n_A)

    # 6. SDP backward induction ─────────────────────────────────────────────────
    gamma = sdp_cfg.get("gamma", 3.0)
    beta = sdp_cfg.get("beta", 0.97)
    mc_samples = sdp_cfg.get("mc_samples", 2_000)

    logger.info("Starting Bellman backward induction (T=%d, n_W=%d, n_A=%d, S=%d)...",
                T, n_W, len(weights), mc_samples)
    V, policy = bellman_backward(
        grid=grid,
        cash_flows=cash_flows,
        return_model=return_model,
        weights=weights,
        gamma=gamma,
        beta=beta,
        mc_samples=mc_samples,
        rng=np.random.default_rng(42),
    )

    # 7. Forward simulation ─────────────────────────────────────────────────────
    initial_wealth = cf_cfg.get("initial_wealth", 100_000)
    n_paths = analysis_cfg.get("num_paths", 20_000)

    logger.info("Forward simulating %d paths...", n_paths)
    paths = forward_simulate(
        initial_wealth=initial_wealth,
        cash_flows=cash_flows,
        policy=policy,
        grid=grid,
        weights=weights,
        return_model=return_model,
        n_paths=n_paths,
        rng=np.random.default_rng(0),
    )

    # 8. PDF propagation ────────────────────────────────────────────────────────
    # Sample ~5 representative periods (start, each phase transition, end)
    sample_periods = list(range(0, T + 1, max(1, T // 4)))
    if T not in sample_periods:
        sample_periods.append(T)
    sample_periods = sorted(set(sample_periods))[:5]

    period_pdfs = propagate_pdfs(paths, phase_labels, periods_to_sample=sample_periods)

    # 9. Ruin analysis ──────────────────────────────────────────────────────────
    ruin_results = compute_ruin(paths, phase_labels)
    logger.info(
        "Lifetime ruin probability: %.4f (%.2f%%)",
        ruin_results.ruin_lifetime,
        ruin_results.ruin_lifetime * 100,
    )

    # 10. Safe Withdrawal Rate ──────────────────────────────────────────────────
    tolerance = analysis_cfg.get("swr_tolerance", 0.05)
    bracket = analysis_cfg.get("swr_bracket", [0.01, 0.15])

    logger.info("Computing Safe Withdrawal Rate (tolerance=%.2f%%)...", tolerance * 100)

    # Build probe rates for the ruin curve (coarse grid for plotting)
    probe_rates = np.linspace(bracket[0], bracket[1], 10)
    probe_ruins = []

    from retirement_sdp.sdp.policy import forward_simulate as _fwd
    from retirement_sdp.models.cashflow import build_schedule as _sched
    from retirement_sdp.analysis.ruin import compute_ruin as _ruin

    for r in probe_rates:
        wd = r * initial_wealth
        s = _sched(phases, 0.0, wd, inflation, withdrawal_override=wd)
        cf_r = cash_flows.copy().astype(float)
        for i in range(min(T, len(s.cash_flows))):
            if cash_flows[i] < 0:
                cf_r[i] = s.cash_flows[i]
        p = _fwd(initial_wealth, cf_r, policy, grid, weights, return_model,
                 min(2000, n_paths), rng=np.random.default_rng(1))
        rr = _ruin(p, phase_labels)
        probe_ruins.append(rr.ruin_lifetime)
        logger.info("  rate=%.3f  P_ruin=%.4f", r, rr.ruin_lifetime)

    probe_ruins_arr = np.array(probe_ruins)

    swr, swr_ruin = compute_swr(
        initial_wealth=initial_wealth,
        cash_flows_template=cash_flows,
        policy=policy,
        grid=grid,
        weights=weights,
        return_model=return_model,
        phases=phases,
        inflation_model=inflation,
        phase_labels=phase_labels,
        n_paths=min(5000, n_paths),
        tolerance=tolerance,
        bracket=bracket,
        rng_seed=2,
    )

    print(f"\n{'='*60}")
    print(f"  Safe Withdrawal Rate:  {swr*100:.2f}%  of initial wealth")
    print(f"  = ${swr * initial_wealth:,.0f} / year (real)")
    print(f"  Ruin probability at SWR: {swr_ruin*100:.2f}%")
    print(f"  Lifetime ruin (base case): {ruin_results.ruin_lifetime*100:.2f}%")
    print(f"{'='*60}\n")

    # 11. Figures ──────────────────────────────────────────────────────────────
    # a) Efficient frontier
    logger.info("Saving efficient_frontier.png ...")
    individual_sigma = np.sqrt(np.diag(Sigma))
    plot_efficient_frontier(
        frontier_mu=frontier_mu,
        frontier_sigma=frontier_sigma,
        weights=weights,
        tickers=tickers,
        individual_mu=mu,
        individual_sigma=individual_sigma,
        output_path=Path(figures_dir) / "efficient_frontier.png",
        dpi=dpi,
    )

    # b) Wealth PDF
    logger.info("Saving wealth_pdf.png ...")
    w_eval = np.linspace(grid[0], np.percentile(paths[paths > 0], 99), 300)
    plot_wealth_pdfs(
        period_pdfs=period_pdfs,
        w_grid=w_eval,
        output_path=Path(figures_dir) / "wealth_pdf.png",
        dpi=dpi,
    )

    # c) Ruin curve
    logger.info("Saving ruin_curve.png ...")
    plot_ruin_curve(
        withdrawal_rates=probe_rates,
        ruin_probabilities=probe_ruins_arr,
        swr=swr,
        swr_ruin=swr_ruin,
        tolerance=tolerance,
        output_path=Path(figures_dir) / "ruin_curve.png",
        dpi=dpi,
    )

    elapsed = time.time() - t_start
    logger.info("All done in %.1fs.", elapsed)
    print(f"Figures saved to: {figures_dir}/")
    print(f"  • efficient_frontier.png")
    print(f"  • wealth_pdf.png")
    print(f"  • ruin_curve.png")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Period Retirement Portfolio Optimization via SDP"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Do not call Alpha Vantage; use only cached Parquet data",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Reduce grid/samples for quick debugging run",
    )
    args = parser.parse_args()
    run(config_path=args.config, skip_fetch=args.skip_fetch, fast=args.fast)


if __name__ == "__main__":
    main()
