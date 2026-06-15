"""
simulation.py
=============
Main loop: use real historical bars as initial context,
then let the agent pool simulate N forward bars.

Flow
----
1. Load real OHLCV (warmup context)
2. Call MarketEngine.step() for each bar
3. Each new bar is appended to the rolling history
4. Return simulated bar DataFrame

Drift policy
------------
Drift is NOT estimated automatically from warmup history.
Warmup drift reflects the past regime, not the future window,
and injecting it causes systematic directional bias.
Pass an explicit drift value via the `drift_per_bar` argument
(or --drift in run_sim.py) only when you have a principled reason.
Default is 0.0 (agents determine direction through net order flow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .agents import BaseAgent, build_default_agents
from .market import MarketEngine


def run_simulation(
    df_real: pd.DataFrame,
    sim_bars: int = 200,
    warmup_bars: int = 100,
    impact_coeff: float = 0.001,
    intra_noise_scale: float = 1.0,
    drift_per_bar: float = 0.0,
    n_institution: int = 5,
    n_momentum:    int = 40,
    n_random:      int = 100,
    n_contrarian:  int = 15,
    seed: int | None = 42,
    agents: list[BaseAgent] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ABM simulation.

    Parameters
    ----------
    df_real           : Real historical OHLCV (Date/Open/High/Low/Close/Volume)
    sim_bars          : Number of bars to simulate
    warmup_bars       : History bars used to initialise agent state
    impact_coeff      : Market impact coefficient
    intra_noise_scale : ATR multiplier for intra-bar noise
    drift_per_bar     : Per-bar log-return drift (default 0.0 = no forced trend)
    seed              : Random seed
    agents            : Custom agent list; None uses build_default_agents()

    Returns
    -------
    df_warmup : Real history used as context (last warmup_bars rows)
    df_sim    : Simulated OHLCV DataFrame
    """
    rng = np.random.default_rng(seed)

    if agents is None:
        agents = build_default_agents(
            n_institution=n_institution,
            n_momentum=n_momentum,
            n_random=n_random,
            n_contrarian=n_contrarian,
            rng=rng,
        )

    # Warmup context
    df_ctx  = df_real.tail(warmup_bars).reset_index(drop=True)
    closes  = df_ctx["Close"].values.astype(float)
    highs   = df_ctx["High"].values.astype(float)
    lows    = df_ctx["Low"].values.astype(float)
    volumes = df_ctx["Volume"].values.astype(float)

    print(f"[sim] drift_per_bar: {drift_per_bar:.6f}  "
          f"(annualised ~{drift_per_bar * 252:.2%})")

    engine = MarketEngine(
        agents=agents,
        impact_coeff=impact_coeff,
        intra_noise_scale=intra_noise_scale,
        drift_per_bar=drift_per_bar,
        seed=int(rng.integers(0, 2**31)),
    )

    rows = []
    last_date = pd.Timestamp(df_ctx["Date"].iloc[-1])

    for i in range(sim_bars):
        bar = engine.step(
            close_history=closes,
            high_history=highs,
            low_history=lows,
            volume_history=volumes,
            bar_idx=warmup_bars + i,
        )

        next_date = last_date + pd.offsets.BDay(1)
        last_date = next_date

        rows.append({
            "Date":     next_date,
            "Open":     bar["open"],
            "High":     bar["high"],
            "Low":      bar["low"],
            "Close":    bar["close"],
            "Volume":   bar["volume"],
            "NetOrder": bar["net_order"],
        })

        closes  = np.append(closes,  bar["close"])
        highs   = np.append(highs,   bar["high"])
        lows    = np.append(lows,    bar["low"])
        volumes = np.append(volumes, bar["volume"])

    df_sim = pd.DataFrame(rows)
    return df_ctx, df_sim
