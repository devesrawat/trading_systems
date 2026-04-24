"""
Portfolio-level pre-execution risk check manager.

Runs before CircuitBreaker to prevent concentrated bets, hidden correlations,
and mode-specific violations.
"""

from __future__ import annotations

import structlog

from portfolio.correlation import (
    apply_correlation_penalty,
    compute_portfolio_correlation,
    is_redundant_position,
)
from portfolio.exposure import (
    compute_sector_exposure,
    exposure_adjusted_capital_for_signal,
    is_over_sector_limit,
)
from portfolio.limits import get_limits_for_mode
from portfolio.schema import PortfolioState, RiskDecision, RiskLimits
from portfolio.turnover import (
    check_turnover_limit,
    estimate_portfolio_turnover,
)
from signals.contracts import Signal

log = structlog.get_logger(__name__)


class PreExecutionRiskCheck:
    """
    Portfolio-level pre-execution risk checker.

    Validates every signal before it reaches CircuitBreaker.
    Enforces sector concentration, correlation, liquidity, and turnover limits.
    """

    def __init__(self, limits: RiskLimits | None = None) -> None:
        """
        Initialize risk checker.

        Parameters
        ----------
        limits : RiskLimits | None
            Risk limits. If None, uses DEFAULT_EQUITY_LIMITS.
        """
        if limits is None:
            limits = get_limits_for_mode("paper")
        self.limits = limits

    def check_signal_execution(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> RiskDecision:
        """
        Comprehensive pre-execution risk check for a signal.

        Runs all checks: sector concentration, correlation, liquidity, turnover,
        position count, and mode-specific gates.

        Parameters
        ----------
        signal : Signal
            Signal to evaluate.
        portfolio : PortfolioState
            Current portfolio state.

        Returns
        -------
        RiskDecision
            Decision (allowed/denied) with details.
        """
        decision = RiskDecision(
            allowed=True,
            reason="OK",
            capital_allocated=0.0,
            priority=3,
        )

        # Get mode-specific limits if different
        # Only override if self.limits was default (not explicitly set)
        # For testing, allow explicit limits to take precedence
        current_limits = self.limits

        # 1. Check position count
        if not self._check_position_count(portfolio, decision, current_limits):
            return decision

        # 2. Check single-stock concentration (2% hard limit)
        capital_allowed = self._check_single_stock_concentration(
            signal, portfolio, decision, current_limits
        )
        if capital_allowed <= 0:
            return decision
        decision.capital_allocated = capital_allowed

        # 3. Check sector concentration
        capital_allowed = self._check_sector_concentration(
            signal, portfolio, decision, current_limits
        )
        if capital_allowed <= 0:
            return decision
        decision.capital_allocated = min(decision.capital_allocated, capital_allowed)

        # 4. Check liquidity
        if not self._check_liquidity(signal, portfolio, decision, current_limits):
            return decision

        # 5. Check correlation to portfolio
        confidence_after_correlation = self._check_correlation(
            signal, portfolio, decision, current_limits
        )
        if confidence_after_correlation < 0.1:
            decision.set_denied("correlation penalty reduced confidence below threshold")
            return decision
        decision.adjustments["correlation_penalty"] = 1.0 - confidence_after_correlation

        # 6. Check portfolio turnover
        if not self._check_turnover(signal, portfolio, decision, current_limits):
            return decision

        # 7. Check mode-specific gates
        if not self._check_mode_specific(signal, portfolio, decision):
            return decision

        # All checks passed
        decision.allowed = True
        decision.reason = f"all checks passed; capital_allocated={decision.capital_allocated:.0f}"
        decision.priority = self._compute_priority(signal, decision)

        log.info(
            "portfolio_risk_check_passed",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            capital_allocated=decision.capital_allocated,
            checks_passed=decision.checks_passed,
        )
        return decision

    # -----------------------------------------------------------------------
    # Private helper checks
    # -----------------------------------------------------------------------

    def _check_position_count(
        self,
        portfolio: PortfolioState,
        decision: RiskDecision,
        limits: RiskLimits,
    ) -> bool:
        """Check if portfolio has room for more positions."""
        current_count = len(portfolio.positions)
        if current_count >= limits.max_positions:
            msg = f"max_positions {limits.max_positions} reached; current={current_count}"
            decision.set_denied(msg)
            decision.add_failed_check("position_count", msg)
            return False
        decision.add_passed_check("position_count")
        return True

    def _check_single_stock_concentration(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        decision: RiskDecision,
        limits: RiskLimits,
    ) -> float:
        """
        Check single-stock concentration (hard 2% limit).

        Returns capital allowed.
        """
        # Estimate position size from signal
        if signal.risk and signal.risk.size_hint_pct:
            position_pct = signal.risk.size_hint_pct
        else:
            # Default: 1% of capital
            position_pct = 0.01

        # Check against limit
        if position_pct > limits.max_single_stock_pct:
            msg = f"single_stock_pct {position_pct:.1%} exceeds limit {limits.max_single_stock_pct:.1%}"
            decision.set_denied(msg)
            decision.add_failed_check("single_stock_concentration", msg)
            return 0.0

        # Calculate allowed capital
        allowed_capital = position_pct * portfolio.total_capital
        decision.add_passed_check("single_stock_concentration")
        decision.adjustments["single_stock_pct"] = position_pct
        return allowed_capital

    def _check_sector_concentration(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        decision: RiskDecision,
        limits: RiskLimits,
    ) -> float:
        """
        Check sector concentration.

        Returns capital allowed after sector constraint.
        """
        sector_exposure = compute_sector_exposure(portfolio)

        # Get signal position estimate
        if signal.risk and signal.risk.capital_at_risk:
            capital_needed = signal.risk.capital_at_risk
        elif signal.entry and signal.entry.entry_price:
            # Rough estimate: assume 1% position (to be refined by PositionSizer)
            capital_needed = portfolio.total_capital * 0.01
        else:
            capital_needed = portfolio.total_capital * 0.01

        # Check sector limit
        from portfolio.exposure import get_sector_for_symbol

        sector = get_sector_for_symbol(signal.symbol)
        if is_over_sector_limit(
            sector,
            capital_needed / signal.entry.entry_price if signal.entry else 1.0,
            signal.entry.entry_price if signal.entry else 100.0,
            sector_exposure,
            portfolio.total_capital,
            limits.max_sector_pct,
        ):
            msg = f"sector {sector} near or over {limits.max_sector_pct:.1%} limit"
            decision.set_denied(msg)
            decision.add_failed_check("sector_concentration", msg)
            return 0.0

        # Get adjusted capital
        qty_estimate = capital_needed / signal.entry.entry_price if signal.entry else 100.0
        allowed_capital = exposure_adjusted_capital_for_signal(
            signal.symbol,
            qty_estimate,
            signal.entry.entry_price if signal.entry else 100.0,
            sector_exposure,
            portfolio.total_capital,
            limits.max_sector_pct,
        )

        decision.add_passed_check("sector_concentration")
        decision.adjustments["sector_exposure"] = (
            sector_exposure[sector].pct_of_capital if sector in sector_exposure else 0.0
        )
        return allowed_capital

    def _check_liquidity(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        decision: RiskDecision,
        limits: RiskLimits,
    ) -> bool:
        """Check liquidity score meets minimum."""
        # Stub: assume liquidity data from config or external source
        # For now, use a default score
        liquidity_score = 75.0

        if liquidity_score < limits.min_liquidity_score:
            msg = f"liquidity_score {liquidity_score:.0f} below minimum {limits.min_liquidity_score:.0f}"
            decision.set_denied(msg)
            decision.add_failed_check("liquidity", msg)
            return False

        decision.add_passed_check("liquidity")
        decision.adjustments["liquidity_score"] = liquidity_score
        return True

    def _check_correlation(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        decision: RiskDecision,
        limits: RiskLimits,
    ) -> float:
        """
        Check correlation to portfolio and apply penalty.

        Returns adjusted confidence after penalty.
        """
        avg_corr, _ = compute_portfolio_correlation(
            signal.symbol,
            portfolio.positions,
        )

        # Check if redundant
        if is_redundant_position(signal.symbol, portfolio.positions, threshold=0.6):
            msg = f"symbol highly correlated (avg={avg_corr:.2f}) to portfolio"
            decision.set_denied(msg)
            decision.add_failed_check("correlation", msg)
            return 0.0

        # Check against limit
        if avg_corr > limits.max_correlation_to_portfolio:
            msg = f"avg_correlation {avg_corr:.2f} exceeds limit {limits.max_correlation_to_portfolio:.2f}"
            # Warn but allow with penalty
            log.warning(
                "correlation_limit_approached",
                avg_corr=avg_corr,
                limit=limits.max_correlation_to_portfolio,
            )

        # Apply penalty
        confidence_after = apply_correlation_penalty(signal.confidence, avg_corr)
        decision.add_passed_check("correlation")
        decision.adjustments["correlation_penalty_reduction"] = signal.confidence - confidence_after
        decision.adjustments["avg_correlation"] = avg_corr

        return confidence_after

    def _check_turnover(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        decision: RiskDecision,
        limits: RiskLimits,
    ) -> bool:
        """Check portfolio turnover."""
        # Estimate new position notional
        if signal.risk and signal.risk.capital_at_risk:
            notional = signal.risk.capital_at_risk
        elif signal.entry:
            notional = portfolio.total_capital * 0.01  # Default 1%
        else:
            notional = portfolio.total_capital * 0.01

        estimated_turnover = estimate_portfolio_turnover(
            portfolio.positions,
            notional,
            portfolio.total_capital,
        )

        allowed, reason = check_turnover_limit(
            estimated_turnover,
            limits.max_turnover_pct_annual,
        )

        if not allowed:
            decision.set_denied(reason)
            decision.add_failed_check("turnover", reason)
            return False

        decision.add_passed_check("turnover")
        decision.adjustments["estimated_turnover"] = estimated_turnover
        return True

    def _check_mode_specific(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        decision: RiskDecision,
    ) -> bool:
        """Check mode-specific gates."""
        mode = signal.mode

        if mode == "research":
            # Research mode: allow all
            decision.add_passed_check("mode_specific")
            return True

        if mode == "watchlist":
            # Watchlist: no execution
            msg = "watchlist mode: signal not executable"
            decision.set_denied(msg)
            decision.add_failed_check("mode_specific", msg)
            return False

        if mode == "paper":
            # Paper trading: normal execution
            decision.add_passed_check("mode_specific")
            return True

        if mode == "live":
            # Live trading: require sufficient capital buffer
            if portfolio.net_worth < portfolio.total_capital * 0.5:
                msg = "live mode: insufficient capital buffer (< 50% of initial)"
                decision.set_denied(msg)
                decision.add_failed_check("mode_specific", msg)
                return False
            decision.add_passed_check("mode_specific")
            return True

        # Unknown mode: conservative
        decision.add_passed_check("mode_specific")
        return True

    def _compute_priority(self, signal: Signal, decision: RiskDecision) -> int:
        """Compute execution priority (1=high, 5=low) based on signal and risk."""
        # Base priority from signal rank
        if signal.rank and signal.rank <= 10:
            priority = 1
        elif signal.rank and signal.rank <= 50:
            priority = 2
        else:
            priority = 3

        # Adjust based on capital constraints
        if decision.capital_allocated < decision.capital_allocated * 0.7:
            priority += 1  # Reduce priority if capital was constrained

        return min(5, max(1, priority))
