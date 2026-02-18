"""Contribution and withdrawal schedule with inflation adjustment.

Generates per-period (annual) nominal cash flows across life phases:
- Accumulation: positive contributions  c_t > 0
- Drawdown: negative withdrawals       d_t < 0 (stored as negative number)

All values are in nominal dollars based on the inflation model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .inflation import InflationModel

logger = logging.getLogger(__name__)


@dataclass
class Phase:
    name: str
    years: int
    phase_type: str   # "accumulation" or "drawdown"


@dataclass
class CashFlowSchedule:
    """Per-period cash flows and metadata."""
    periods: np.ndarray       # integer period indices 0 … T-1
    cash_flows: np.ndarray    # nominal cash flow per period (+ = contribution, - = withdrawal)
    phase_labels: list[str]   # phase name for each period
    total_periods: int


def build_schedule(
    phases: list[dict],
    annual_contribution: float,
    annual_withdrawal: float,
    inflation_model: InflationModel,
    withdrawal_override: float | None = None,
) -> CashFlowSchedule:
    """Build the full cash-flow schedule across all life phases.

    Parameters
    ----------
    phases:
        List of phase dicts with keys ``name``, ``years``, ``type``.
    annual_contribution:
        Real (base-year) annual contribution amount.
    annual_withdrawal:
        Real (base-year) annual withdrawal amount.
    inflation_model:
        Used to convert real cash flows to nominal.
    withdrawal_override:
        If provided, overrides ``annual_withdrawal`` (used by SWR search).

    Returns
    -------
    CashFlowSchedule
    """
    if withdrawal_override is not None:
        annual_withdrawal = withdrawal_override

    phase_objs = [
        Phase(name=p["name"], years=p["years"], phase_type=p["type"])
        for p in phases
    ]

    periods = []
    cash_flows = []
    phase_labels = []

    t = 0
    for phase in phase_objs:
        for _ in range(phase.years):
            if phase.phase_type == "accumulation":
                cf = inflation_model.real_to_nominal(annual_contribution, t)
            else:
                cf = -inflation_model.real_to_nominal(annual_withdrawal, t)
            periods.append(t)
            cash_flows.append(cf)
            phase_labels.append(phase.name)
            t += 1

    T = t
    logger.info(
        "CashFlowSchedule: %d periods, contrib=%.0f, withdrawal=%.0f (real), pi=%.3f",
        T,
        annual_contribution,
        annual_withdrawal,
        inflation_model.annual_rate,
    )

    return CashFlowSchedule(
        periods=np.array(periods),
        cash_flows=np.array(cash_flows),
        phase_labels=phase_labels,
        total_periods=T,
    )
