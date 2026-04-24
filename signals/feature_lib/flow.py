"""Flow and institutional indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fii_net_cash_normalized(fii_net_cash: float | None) -> float:
    if fii_net_cash is None:
        return np.nan
    return fii_net_cash / 1e5


def compute_fii_participation_series(
    fii_buying: pd.Series | None = None, fii_selling: pd.Series | None = None, period: int = 20
) -> pd.Series:
    if fii_buying is None or fii_selling is None:
        if fii_buying is not None:
            return pd.Series(np.nan, index=fii_buying.index)
        elif fii_selling is not None:
            return pd.Series(np.nan, index=fii_selling.index)
        else:
            return pd.Series(np.nan)
    total_vol = fii_buying + fii_selling
    total_vol = total_vol.replace(0, np.nan)
    net_participation = (fii_buying - fii_selling) / total_vol
    return net_participation.rolling(period).mean()


def compute_dii_participation_series(
    dii_buying: pd.Series | None = None, dii_selling: pd.Series | None = None, period: int = 20
) -> pd.Series:
    if dii_buying is None or dii_selling is None:
        if dii_buying is not None:
            return pd.Series(np.nan, index=dii_buying.index)
        elif dii_selling is not None:
            return pd.Series(np.nan, index=dii_selling.index)
        else:
            return pd.Series(np.nan)
    total_vol = dii_buying + dii_selling
    total_vol = total_vol.replace(0, np.nan)
    net_participation = (dii_buying - dii_selling) / total_vol
    return net_participation.rolling(period).mean()


def compute_net_flow_to_volume_ratio(
    net_flow: pd.Series | None = None, volume: pd.Series | None = None, period: int = 20
) -> pd.Series:
    if net_flow is None or volume is None:
        if net_flow is not None:
            return pd.Series(np.nan, index=net_flow.index)
        elif volume is not None:
            return pd.Series(np.nan, index=volume.index)
        else:
            return pd.Series(np.nan)
    volume_safe = volume.replace(0, np.nan)
    ratio = net_flow / volume_safe
    return ratio.rolling(period).mean()


def compute_mf_inflow_trend(mf_inflow: pd.Series | None = None, period: int = 20) -> pd.Series:
    if mf_inflow is None:
        return pd.Series(np.nan)
    return mf_inflow.rolling(period).mean()


def compute_institutional_holding_trend(
    institutional_holding: pd.Series | None = None, period: int = 20
) -> pd.Series:
    if institutional_holding is None:
        return pd.Series(np.nan)
    return institutional_holding.rolling(period).mean()


def compute_retail_participation(
    total_volume: pd.Series | None = None,
    institutional_volume: pd.Series | None = None,
    period: int = 20,
) -> pd.Series:
    if total_volume is None or institutional_volume is None:
        if total_volume is not None:
            return pd.Series(np.nan, index=total_volume.index)
        elif institutional_volume is not None:
            return pd.Series(np.nan, index=institutional_volume.index)
        else:
            return pd.Series(np.nan)
    total_vol_safe = total_volume.replace(0, np.nan)
    retail_volume = total_volume - institutional_volume
    retail_ratio = retail_volume / total_vol_safe
    return retail_ratio.rolling(period).mean()
