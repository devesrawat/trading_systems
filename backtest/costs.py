"""
Realistic NSE transaction cost model.

WARNING: A strategy needs >0.75% per trade edge just to break even.
Build this constraint into every signal threshold decision.

Breakdown per trade (Zerodha, NSE equity):
  Brokerage    = min(₹20, 0.03% of trade value)
  STT          = 0.1% of sell-side value (delivery); 0.025% of sell-side (intraday)
  Exchange fee = 0.00345% of trade value
  SEBI fee     = 0.0001% of trade value
  GST          = 18% on (brokerage + exchange fee)
  Stamp duty   = 0.015% on buy-side only
"""
from __future__ import annotations


class NSECostModel:

    # ------------------------------------------------------------------
    # equity_cost
    # ------------------------------------------------------------------

    def equity_cost(
        self,
        trade_value: float,
        side: str,
        intraday: bool = False,
    ) -> float:
        """
        Total transaction cost for one equity order leg.

        side : 'BUY' or 'SELL'
        """
        brokerage = min(20.0, trade_value * 0.0003)

        if side == "SELL":
            stt = trade_value * (0.00025 if intraday else 0.001)
        else:
            stt = 0.0

        exchange_fee = trade_value * 0.0000345
        sebi_fee = trade_value * 0.000001
        gst = (brokerage + exchange_fee) * 0.18
        stamp_duty = trade_value * 0.00015 if side == "BUY" else 0.0

        return brokerage + stt + exchange_fee + sebi_fee + gst + stamp_duty

    # ------------------------------------------------------------------
    # slippage
    # ------------------------------------------------------------------

    def slippage(self, trade_value: float, liquidity_tier: str = "large_cap") -> float:
        """
        Estimated market-impact slippage as an INR amount.

        Tiers:
          large_cap  — 0.05%
          mid_cap    — 0.12%
          small_cap  — 0.20%
        """
        rates = {
            "large_cap": 0.0005,
            "mid_cap": 0.0012,
            "small_cap": 0.0020,
        }
        rate = rates.get(liquidity_tier, rates["large_cap"])
        return trade_value * rate

    # ------------------------------------------------------------------
    # round_trip_cost
    # ------------------------------------------------------------------

    def round_trip_cost(
        self,
        trade_value: float,
        liquidity_tier: str = "large_cap",
        intraday: bool = False,
    ) -> float:
        """
        Total cost for a complete buy + sell cycle.

        Includes: buy cost + sell cost + buy slippage + sell slippage.
        For ₹10,000 large_cap delivery: approx ₹55–75 (0.55–0.75%).
        """
        buy_cost = self.equity_cost(trade_value, side="BUY", intraday=intraday)
        sell_cost = self.equity_cost(trade_value, side="SELL", intraday=intraday)
        buy_slip = self.slippage(trade_value, liquidity_tier)
        sell_slip = self.slippage(trade_value, liquidity_tier)
        return buy_cost + sell_cost + buy_slip + sell_slip
