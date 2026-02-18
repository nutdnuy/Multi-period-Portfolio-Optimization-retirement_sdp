"""Safe Withdrawal Rate (SWR) computation via Brent root-finding.

The SWR is the largest annual withdrawal rate r such that
    P_ruin(r) ≤ tolerance

P_ruin(r) is estimated empirically from a fresh forward simulation that
overrides the withdrawal amount with r × initial_wealth.

Brent's method on [r_lo, r_hi] solves  P_ruin(r) - tolerance = 0.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from .ruin import compute_ruin

logger = logging.getLogger(__name__)


def compute_swr(
    initial_wealth: float,
    cash_flows_template: np.ndarray,   # (T,) — positives & negatives
    policy: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
    return_model,
    phases: list[dict],
    inflation_model,
    phase_labels: list[str],
    n_paths: int,
    tolerance: float = 0.05,
    bracket: tuple[float, float] = (0.01, 0.15),
    rng_seed: int = 1,
) -> tuple[float, float]:
    """Find the safe withdrawal rate by Brent root-finding.

    For each candidate withdrawal rate *r*, rebuilds the cash-flow schedule
    (keeping contributions unchanged, overriding annual withdrawal to
    r × initial_wealth), re-runs forward simulation, and returns lifetime
    ruin probability.

    Parameters
    ----------
    initial_wealth : W_0
    cash_flows_template : Original cash-flow array (used to identify
        accumulation vs drawdown periods — sign determines treatment).
    policy, grid, weights, return_model : SDP solution artefacts.
    phases : phase list from config.
    inflation_model : InflationModel instance.
    phase_labels : per-period phase name list.
    n_paths : paths for each forward sim evaluation.
    tolerance : maximum acceptable ruin probability (e.g. 0.05 = 5%).
    bracket : (r_lo, r_hi) search interval.
    rng_seed : seed for reproducibility.

    Returns
    -------
    swr : safe withdrawal rate as a fraction of initial wealth
    p_ruin_at_swr : ruin probability at the computed SWR
    """
    from ..models.cashflow import build_schedule
    from ..sdp.policy import forward_simulate

    rng = np.random.default_rng(rng_seed)
    T = len(cash_flows_template)

    def _ruin_for_rate(r: float) -> float:
        annual_withdrawal = r * initial_wealth
        schedule = build_schedule(
            phases=phases,
            annual_contribution=0.0,   # overridden — contributions kept from template
            annual_withdrawal=annual_withdrawal,
            inflation_model=inflation_model,
            withdrawal_override=annual_withdrawal,
        )
        # Merge: keep original contributions, replace drawdown with r-based amount
        cf = cash_flows_template.copy().astype(float)
        new_cf = schedule.cash_flows
        for i in range(min(T, len(new_cf))):
            if cash_flows_template[i] < 0:   # drawdown period
                cf[i] = new_cf[i]

        paths = forward_simulate(
            initial_wealth=initial_wealth,
            cash_flows=cf,
            policy=policy,
            grid=grid,
            weights=weights,
            return_model=return_model,
            n_paths=n_paths,
            rng=np.random.default_rng(rng_seed),
        )
        ruin_res = compute_ruin(paths, phase_labels)
        p = float(ruin_res.ruin_lifetime)
        logger.debug("SWR probe: r=%.4f  P_ruin=%.4f", r, p)
        return p

    r_lo, r_hi = bracket
    p_lo = _ruin_for_rate(r_lo)
    p_hi = _ruin_for_rate(r_hi)

    logger.info(
        "SWR bracket: r=[%.4f, %.4f]  P_ruin=[%.4f, %.4f]  tol=%.4f",
        r_lo, r_hi, p_lo, p_hi, tolerance,
    )

    # Check if tolerance is achievable within bracket
    if p_lo > tolerance:
        logger.warning(
            "Even minimum withdrawal rate %.4f gives P_ruin=%.4f > tolerance=%.4f. "
            "Returning r_lo as conservative SWR.",
            r_lo, p_lo, tolerance,
        )
        return r_lo, p_lo

    if p_hi <= tolerance:
        logger.info(
            "Maximum bracket rate %.4f has P_ruin=%.4f ≤ tolerance. "
            "SWR may be higher than bracket; returning r_hi.",
            r_hi, p_hi,
        )
        return r_hi, p_hi

    # f(r) = P_ruin(r) - tolerance: f(r_lo) < 0, f(r_hi) > 0
    def _objective(r: float) -> float:
        return _ruin_for_rate(r) - tolerance

    swr = brentq(_objective, r_lo, r_hi, xtol=1e-4, rtol=1e-4, maxiter=20)
    p_ruin_at_swr = _ruin_for_rate(swr)

    logger.info(
        "Safe Withdrawal Rate: %.4f (%.2f%%)  P_ruin=%.4f",
        swr, swr * 100, p_ruin_at_swr,
    )
    return swr, p_ruin_at_swr
