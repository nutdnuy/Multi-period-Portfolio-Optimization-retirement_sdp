#!/usr/bin/env python3
"""Streamlit app — Retirement Portfolio SDP Simulator."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── Project src on path ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from retirement_sdp.analysis.pdf_propagation import propagate_pdfs
from retirement_sdp.analysis.ruin import compute_ruin
from retirement_sdp.analysis.swr import compute_swr
from retirement_sdp.data.alpha_vantage import AlphaVantageClient
from retirement_sdp.data.cache import ParquetCache
from retirement_sdp.data.preprocessor import monthly_prices_to_series, preprocess
from retirement_sdp.models.cashflow import build_schedule
from retirement_sdp.models.inflation import InflationModel
from retirement_sdp.models.return_model import LogNormalReturnModel
from retirement_sdp.sdp.action_space import build_efficient_frontier
from retirement_sdp.sdp.bellman import bellman_backward
from retirement_sdp.sdp.policy import forward_simulate
from retirement_sdp.sdp.state_space import build_wealth_grid

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retirement SDP Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CACHE_DIR = ROOT / "outputs" / "cache"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # ── Data ────────────────────────────────────────────────────────────────────
    st.subheader("Data")
    api_key = st.text_input(
        "Alpha Vantage API Key",
        value="97ENMFDNTX5BV5JI",
        type="password",
        help="Free tier: 25 requests/day. Leave as-is to use cached data.",
    )
    tickers_input = st.text_input("Tickers (comma-separated)", value="SPY,AGG,GLD")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.divider()

    # ── Cash flow ───────────────────────────────────────────────────────────────
    st.subheader("Cash Flow")
    initial_wealth = st.number_input(
        "Initial Wealth ($)", value=100_000, min_value=1_000, step=10_000, format="%d"
    )
    annual_contribution = st.number_input(
        "Annual Contribution ($)", value=20_000, min_value=0, step=1_000, format="%d"
    )
    annual_withdrawal = st.number_input(
        "Annual Withdrawal ($ real)", value=50_000, min_value=0, step=1_000, format="%d"
    )
    inflation_rate = (
        st.slider("Inflation Rate (%)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
        / 100
    )

    st.divider()

    # ── Life phases ─────────────────────────────────────────────────────────────
    st.subheader("Life Phases")
    _default_phases = pd.DataFrame([
        {"name": "Early Career",     "years": 15, "type": "accumulation"},
        {"name": "Peak Earning",     "years": 10, "type": "accumulation"},
        {"name": "Pre-Retirement",   "years":  5, "type": "accumulation"},
        {"name": "Early Retirement", "years": 10, "type": "drawdown"},
        {"name": "Late Retirement",  "years": 15, "type": "drawdown"},
    ])
    phases_df = st.data_editor(
        _default_phases,
        column_config={
            "name":  st.column_config.TextColumn("Phase Name"),
            "years": st.column_config.NumberColumn("Years", min_value=1, max_value=50),
            "type":  st.column_config.SelectboxColumn(
                "Type", options=["accumulation", "drawdown"]
            ),
        },
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )
    phases = phases_df.to_dict("records")

    st.divider()

    # ── SDP parameters ──────────────────────────────────────────────────────────
    st.subheader("SDP Parameters")
    fast_mode = st.toggle(
        "Fast Mode",
        value=True,
        help="100 grid points, 20 portfolios, 200 MC samples — runs in ~2 s.",
    )
    if fast_mode:
        n_W, n_A, mc_samples, n_paths = 100, 20, 200, 1_000
    else:
        col_a, col_b = st.columns(2)
        n_W      = col_a.slider("Wealth Grid", 100, 1000, 500, 50)
        n_A      = col_b.slider("Portfolios",   20,  200, 100, 10)
        mc_samples = st.slider("MC Samples (Bellman)", 200, 5_000, 2_000, 200)
        n_paths    = st.slider("Sim Paths", 1_000, 50_000, 20_000, 1_000)

    gamma = st.slider("Risk Aversion γ (CRRA)", 0.5, 10.0, 3.0, 0.5)
    beta  = st.slider("Discount Factor β",       0.90, 1.00, 0.97, 0.01)

    st.divider()

    # ── SWR ─────────────────────────────────────────────────────────────────────
    st.subheader("SWR Analysis")
    swr_tolerance = st.slider("Max Ruin Tolerance (%)", 1, 20, 5) / 100
    swr_col1, swr_col2 = st.columns(2)
    swr_lo = swr_col1.number_input("Bracket Low (%)",  value=1.0,  min_value=0.1,  max_value=10.0)  / 100
    swr_hi = swr_col2.number_input("Bracket High (%)", value=15.0, min_value=5.0,  max_value=50.0) / 100

    st.divider()
    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)

# ── Main header ────────────────────────────────────────────────────────────────
st.title("📊 Retirement Portfolio SDP Simulator")
st.caption(
    "Multi-period retirement optimization · "
    "Log-normal returns · CRRA utility · Bellman backward induction · Monte Carlo"
)

# ── Welcome screen ─────────────────────────────────────────────────────────────
if not run_btn and "results" not in st.session_state:
    st.info(
        "Configure your parameters in the sidebar and click **🚀 Run Simulation** to start."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Return Model",   "Log-Normal")
    c2.metric("Optimizer",      "Mean-Variance")
    c3.metric("DP Solver",      "Bellman Backward")
    c4.metric("Risk Measure",   "Lifetime Ruin P")
    st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    t_start = time.time()
    cache = ParquetCache(CACHE_DIR)
    client = AlphaVantageClient(api_key=api_key)

    with st.status("Running simulation…", expanded=True) as status:

        # 1. Price data ────────────────────────────────────────────────────────
        st.write("📥 Loading price data…")
        raw_prices: dict[str, dict] = {}
        for ticker in tickers:
            key = f"{ticker}_monthly"
            if cache.exists(key):
                df = cache.load(key)
                raw_prices[ticker] = {
                    str(idx.date()): {"5. adjusted close": str(v)}
                    for idx, v in zip(df.index, df["adj_close"])
                }
            else:
                st.write(f"  ↳ Fetching {ticker} from Alpha Vantage…")
                raw = client.get_monthly_adjusted(ticker)
                s = monthly_prices_to_series(raw)
                cache.save(key, s.to_frame("adj_close"))
                raw_prices[ticker] = raw

        # 2. Preprocess ────────────────────────────────────────────────────────
        st.write("📊 Estimating return parameters…")
        mu, Sigma, tickers_ok, returns_df = preprocess(raw_prices, tickers)
        return_model = LogNormalReturnModel(mu, Sigma, tickers_ok)

        # 3. Inflation & cash-flow schedule ────────────────────────────────────
        st.write("📅 Building cash-flow schedule…")
        inflation = InflationModel(cpi_data={}, fallback_rate=inflation_rate)
        schedule  = build_schedule(phases, annual_contribution, annual_withdrawal, inflation)
        T           = schedule.total_periods
        cash_flows  = schedule.cash_flows
        phase_labels = schedule.phase_labels

        # 4. State & action spaces ─────────────────────────────────────────────
        st.write("🔧 Building efficient frontier…")
        grid = build_wealth_grid(1_000, 10_000_000, n_W)
        weights, frontier_mu, frontier_sigma = build_efficient_frontier(mu, Sigma, n_A)

        # 5. Bellman backward induction ────────────────────────────────────────
        st.write(f"⚙️ Bellman backward induction (T={T}, n_W={n_W}, n_A={n_A}, S={mc_samples})…")
        V, policy = bellman_backward(
            grid=grid, cash_flows=cash_flows, return_model=return_model,
            weights=weights, gamma=gamma, beta=beta, mc_samples=mc_samples,
            rng=np.random.default_rng(42),
        )

        # 6. Forward simulation ────────────────────────────────────────────────
        st.write(f"🎲 Forward simulating {n_paths:,} paths…")
        paths = forward_simulate(
            initial_wealth=initial_wealth, cash_flows=cash_flows, policy=policy,
            grid=grid, weights=weights, return_model=return_model,
            n_paths=n_paths, rng=np.random.default_rng(0),
        )

        # 7. Wealth PDFs ───────────────────────────────────────────────────────
        st.write("📈 Propagating wealth distributions…")
        sample_periods = sorted(set(list(range(0, T + 1, max(1, T // 4))) + [T]))[:5]
        period_pdfs = propagate_pdfs(paths, phase_labels, periods_to_sample=sample_periods)

        # 8. Ruin analysis ─────────────────────────────────────────────────────
        st.write("⚠️ Computing ruin probabilities…")
        ruin_results = compute_ruin(paths, phase_labels)

        # 9. SWR probe curve ───────────────────────────────────────────────────
        st.write("💰 Scanning withdrawal rates for ruin curve…")
        probe_rates = np.linspace(swr_lo, swr_hi, 10)
        probe_ruins: list[float] = []
        for r in probe_rates:
            wd  = r * initial_wealth
            s   = build_schedule(phases, 0.0, wd, inflation, withdrawal_override=wd)
            cf_r = cash_flows.copy().astype(float)
            for i in range(min(T, len(s.cash_flows))):
                if cash_flows[i] < 0:
                    cf_r[i] = s.cash_flows[i]
            pp = forward_simulate(
                initial_wealth, cf_r, policy, grid, weights, return_model,
                min(1_000, n_paths), rng=np.random.default_rng(1),
            )
            probe_ruins.append(compute_ruin(pp, phase_labels).ruin_lifetime)

        # 10. SWR via Brent's method ───────────────────────────────────────────
        st.write("🎯 Solving for Safe Withdrawal Rate…")
        swr, swr_ruin = compute_swr(
            initial_wealth=initial_wealth, cash_flows_template=cash_flows,
            policy=policy, grid=grid, weights=weights, return_model=return_model,
            phases=phases, inflation_model=inflation, phase_labels=phase_labels,
            n_paths=min(3_000, n_paths), tolerance=swr_tolerance,
            bracket=[swr_lo, swr_hi], rng_seed=2,
        )

        elapsed = time.time() - t_start
        status.update(label=f"✅ Done in {elapsed:.1f}s", state="complete")

    # Store all results ─────────────────────────────────────────────────────────
    st.session_state["results"] = dict(
        mu=mu, Sigma=Sigma, tickers=tickers_ok,
        return_model=return_model, returns_df=returns_df,
        paths=paths, cash_flows=cash_flows,
        phase_labels=phase_labels, period_pdfs=period_pdfs,
        ruin_results=ruin_results,
        frontier_mu=frontier_mu, frontier_sigma=frontier_sigma, weights=weights,
        probe_rates=probe_rates, probe_ruins=np.array(probe_ruins),
        swr=swr, swr_ruin=swr_ruin, swr_tolerance=swr_tolerance,
        initial_wealth=initial_wealth, inflation_rate=inflation_rate,
        T=T, elapsed=elapsed,
    )

# ── Display results ────────────────────────────────────────────────────────────
if "results" in st.session_state:
    res = st.session_state["results"]
    mu        = res["mu"]
    Sigma     = res["Sigma"]
    tickers_ok = res["tickers"]
    paths     = res["paths"]
    cash_flows = res["cash_flows"]
    phase_labels = res["phase_labels"]
    period_pdfs  = res["period_pdfs"]
    ruin_results = res["ruin_results"]
    frontier_mu  = res["frontier_mu"]
    frontier_sigma = res["frontier_sigma"]
    weights      = res["weights"]
    probe_rates  = res["probe_rates"]
    probe_ruins  = res["probe_ruins"]
    swr          = res["swr"]
    swr_ruin     = res["swr_ruin"]
    swr_tolerance = res["swr_tolerance"]
    initial_wealth = res["initial_wealth"]
    T            = res["T"]

    # ── Key metrics ─────────────────────────────────────────────────────────────
    st.subheader("Key Results")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Safe Withdrawal Rate",    f"{swr * 100:.2f}%")
    m2.metric("Annual SWR Amount",       f"${swr * initial_wealth:,.0f}")
    m3.metric("Ruin P at SWR",           f"{swr_ruin * 100:.2f}%")
    m4.metric("Lifetime Ruin (base)",    f"{ruin_results.ruin_lifetime * 100:.2f}%")
    m5.metric("Run Time",                f"{res['elapsed']:.1f}s")

    st.divider()

    # ── Tabs ────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Efficient Frontier",
        "💰 Wealth Distribution",
        "⚠️ Ruin Curve",
        "📈 Wealth Paths",
        "🗂️ Data & Stats",
    ])

    # ── Tab 1: Efficient Frontier ──────────────────────────────────────────────
    with tab1:
        st.markdown("### Mean-Variance Efficient Frontier")
        fig, ax = plt.subplots(figsize=(9, 5))
        sharpe = frontier_mu / np.maximum(frontier_sigma, 1e-8)
        sc = ax.scatter(
            frontier_sigma * 100, frontier_mu * 100,
            c=sharpe, cmap="RdYlGn", s=40, zorder=3, label="Frontier portfolios",
        )
        plt.colorbar(sc, ax=ax, label="Sharpe Ratio (μ/σ)")
        individual_sigma = np.sqrt(np.diag(Sigma))
        for i, ticker in enumerate(tickers_ok):
            ax.scatter(
                individual_sigma[i] * 100, mu[i] * 100,
                marker="*", s=250, zorder=5, label=ticker,
            )
        ax.set_xlabel("Annual Volatility (%)")
        ax.set_ylabel("Annual Expected Return (%)")
        ax.set_title("Mean-Variance Efficient Frontier")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Frontier table
        st.markdown("#### Portfolio Weights on Frontier")
        frontier_df = pd.DataFrame(weights, columns=tickers_ok)
        frontier_df.insert(0, "μ (%)", (frontier_mu * 100).round(2))
        frontier_df.insert(1, "σ (%)", (frontier_sigma * 100).round(2))
        frontier_df.insert(2, "Sharpe", (frontier_mu / np.maximum(frontier_sigma, 1e-8)).round(3))
        st.dataframe(
            frontier_df.style.background_gradient(subset=tickers_ok, cmap="Blues"),
            use_container_width=True, height=300,
        )

    # ── Tab 2: Wealth Distribution ─────────────────────────────────────────────
    with tab2:
        st.markdown("### Wealth Distribution by Phase")
        alive = paths[paths > 0]
        w_max_plot = float(np.percentile(alive, 99)) if len(alive) > 0 else 5e6
        w_eval = np.linspace(1_000, w_max_plot, 400)

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(period_pdfs)))
        for color, pdf_obj in zip(colors, period_pdfs):
            density = pdf_obj.evaluate_on_grid(w_eval)
            label = f"{pdf_obj.phase_label} (t={pdf_obj.t}, ruin={pdf_obj.ruin_fraction*100:.1f}%)"
            ax.plot(w_eval / 1e6, density, label=label, color=color, linewidth=2)
        ax.set_xlabel("Wealth ($M)")
        ax.set_ylabel("Density")
        ax.set_title("Wealth PDF Across Life Phases")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Summary stats per period
        st.markdown("#### Wealth Statistics by Period")
        stats_rows = []
        for pdf_obj in period_pdfs:
            w_surv = paths[:, pdf_obj.t]
            w_surv = w_surv[w_surv > 0]
            stats_rows.append({
                "Period": pdf_obj.t,
                "Phase": pdf_obj.phase_label,
                "Ruin (%)": f"{pdf_obj.ruin_fraction*100:.2f}",
                "Median ($)": f"{np.median(w_surv):,.0f}" if len(w_surv) else "—",
                "Mean ($)":   f"{w_surv.mean():,.0f}"      if len(w_surv) else "—",
                "P10 ($)":    f"{np.percentile(w_surv,10):,.0f}" if len(w_surv) else "—",
                "P90 ($)":    f"{np.percentile(w_surv,90):,.0f}" if len(w_surv) else "—",
            })
        st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # ── Tab 3: Ruin Curve ──────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Ruin Probability vs Withdrawal Rate")

        col_left, col_right = st.columns([2, 1])
        with col_left:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                probe_rates * 100, probe_ruins * 100,
                "b-o", linewidth=2, markersize=6, label="P(ruin)",
            )
            ax.axhline(
                swr_tolerance * 100, color="orange", linestyle="--", linewidth=1.5,
                label=f"Tolerance = {swr_tolerance*100:.0f}%",
            )
            ax.axvline(
                swr * 100, color="green", linestyle="--", linewidth=1.5,
                label=f"SWR = {swr*100:.2f}%",
            )
            ax.scatter([swr * 100], [swr_ruin * 100], color="red", s=120, zorder=5,
                       label=f"Ruin at SWR = {swr_ruin*100:.1f}%")
            ax.set_xlabel("Withdrawal Rate (% of initial wealth)")
            ax.set_ylabel("Ruin Probability (%)")
            ax.set_title("Ruin Probability vs Withdrawal Rate")
            ax.set_ylim(-2, 102)
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_right:
            st.markdown("#### Ruin Probe Table")
            ruin_table = pd.DataFrame({
                "Rate (%)":    (probe_rates * 100).round(2),
                "P(ruin) (%)": (probe_ruins * 100).round(2),
                "Amount ($)":  [f"${r*initial_wealth:,.0f}" for r in probe_rates],
            })
            st.dataframe(ruin_table, use_container_width=True, hide_index=True, height=380)

        # Per-period ruin
        st.markdown("### Cumulative Ruin Probability Over Time")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(ruin_results.periods, ruin_results.ruin_empirical * 100,
                 color="crimson", linewidth=2)
        ax2.fill_between(ruin_results.periods, ruin_results.ruin_empirical * 100, alpha=0.15, color="crimson")
        # Phase transition lines
        seen: set[str] = set()
        for t, label in enumerate(phase_labels):
            if label not in seen:
                ax2.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
                ax2.text(t + 0.2, ax2.get_ylim()[1] * 0.95, label,
                         fontsize=7, rotation=90, va="top", color="gray")
                seen.add(label)
        ax2.set_xlabel("Period (years)")
        ax2.set_ylabel("Cumulative Ruin (%)")
        ax2.set_title("Cumulative Ruin Probability Over Time")
        ax2.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Tab 4: Wealth Paths ────────────────────────────────────────────────────
    with tab4:
        st.markdown("### Simulated Wealth Trajectories")
        n_show = st.slider("Paths to display", 20, 500, 100, 20)

        fig, ax = plt.subplots(figsize=(11, 5))
        t_axis = np.arange(T + 1)
        rng_show = np.random.default_rng(99)
        idx_show = rng_show.choice(len(paths), size=min(n_show, len(paths)), replace=False)

        # Colour paths by terminal wealth
        terminal_w = paths[idx_show, -1]
        norm = plt.Normalize(vmin=0, vmax=float(np.percentile(terminal_w[terminal_w > 0], 95)) if (terminal_w > 0).any() else 1)
        cmap = plt.cm.RdYlGn

        for i in idx_show:
            w = paths[i]
            color = cmap(norm(w[-1]))
            ax.plot(t_axis, w / 1e6, color=color, alpha=0.3, linewidth=0.8)

        # Percentile bands
        pct10 = np.percentile(paths, 10, axis=0)
        pct50 = np.percentile(paths, 50, axis=0)
        pct90 = np.percentile(paths, 90, axis=0)
        ax.plot(t_axis, pct50 / 1e6, "k-",  linewidth=2.5, label="Median", zorder=5)
        ax.plot(t_axis, pct10 / 1e6, "r--", linewidth=1.5, label="P10",    zorder=5)
        ax.plot(t_axis, pct90 / 1e6, "g--", linewidth=1.5, label="P90",    zorder=5)
        ax.fill_between(t_axis, pct10 / 1e6, pct90 / 1e6, alpha=0.1, color="blue")

        # Phase labels
        seen2: set[str] = set()
        for t, label in enumerate(phase_labels):
            if label not in seen2:
                ax.axvline(t, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
                ax.text(t + 0.2, ax.get_ylim()[1] * 0.98, label,
                        fontsize=7, rotation=90, va="top", color="gray")
                seen2.add(label)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Terminal Wealth ($M)")
        ax.set_xlabel("Period (years)")
        ax.set_ylabel("Wealth ($M)")
        ax.set_title(f"Wealth Paths Under Optimal SDP Policy (n={len(idx_show):,})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Cash-flow schedule
        st.markdown("#### Cash-Flow Schedule")
        cf_df = pd.DataFrame({
            "Period": range(T),
            "Phase":  phase_labels,
            "Cash Flow ($)": [f"{v:+,.0f}" for v in cash_flows],
        })
        st.dataframe(cf_df, use_container_width=True, height=300, hide_index=True)

    # ── Tab 5: Data & Stats ────────────────────────────────────────────────────
    with tab5:
        st.markdown("### Asset Return Statistics")
        individual_sigma = np.sqrt(np.diag(Sigma))
        stats_df = pd.DataFrame({
            "Ticker":      tickers_ok,
            "Ann. μ (%)":  [f"{v*100:.2f}" for v in mu],
            "Ann. σ (%)":  [f"{v*100:.2f}" for v in individual_sigma],
            "Sharpe":      [f"{m/s:.3f}" for m, s in zip(mu, individual_sigma)],
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        st.markdown("### Correlation Matrix")
        returns_df = res["returns_df"]
        corr = pd.DataFrame(
            np.corrcoef(returns_df.T.values),
            index=tickers_ok, columns=tickers_ok,
        )
        st.dataframe(
            corr.style
                .background_gradient(cmap="RdYlGn", vmin=-1, vmax=1)
                .format("{:.3f}"),
            use_container_width=True,
        )

        st.markdown("### Monthly Returns (tail)")
        st.dataframe(
            returns_df.tail(24).style.format("{:.4f}").background_gradient(cmap="RdYlGn"),
            use_container_width=True,
        )

        st.markdown("### Configuration Summary")
        st.json({
            "tickers":             tickers_ok,
            "initial_wealth":      f"${initial_wealth:,}",
            "annual_contribution": f"${annual_contribution:,}",
            "annual_withdrawal":   f"${annual_withdrawal:,}",
            "inflation_rate":      f"{inflation_rate*100:.1f}%",
            "total_periods":       T,
            "wealth_grid_points":  n_W,
            "num_portfolios":      n_A,
            "mc_samples":          mc_samples,
            "n_paths":             n_paths,
            "gamma":               gamma,
            "beta":                beta,
            "swr":                 f"{swr*100:.2f}%",
            "swr_tolerance":       f"{swr_tolerance*100:.0f}%",
        })
