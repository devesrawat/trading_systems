from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from signals.alpha_composite import AlphaEngine
from signals.sector_alpha import SectorRanker


@pytest.fixture
def mock_ohlcv():
    def _mock(token, start, end, interval):
        # Create dummy data for 30 days
        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=30, freq="D")

        # Adjust price based on token to simulate different sector performance
        # NIFTY 50 (256265) will have 0% return
        # NIFTY BANK (260105) will have 10% return (top sector)
        # NIFTY IT (262665) will have -10% return (bottom sector)

        start_price = 100
        if token == 260105:  # BANK
            prices = np.linspace(100, 110, 30)
        elif token == 262665:  # IT
            prices = np.linspace(100, 90, 30)
        else:
            prices = np.linspace(100, 100, 30)

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 1,
                "low": prices - 1,
                "close": prices,
                "volume": 1000,
            },
            index=dates,
        )
        df.index.name = "time"
        return df

    return _mock


def test_sector_ranker(mock_ohlcv):
    with patch("signals.sector_alpha.get_ohlcv", side_effect=mock_ohlcv):
        ranker = SectorRanker()
        ranks = ranker.get_ranks(force_refresh=True)

        assert not ranks.empty
        assert "NIFTY BANK" in ranks["sector"].values

        # NIFTY BANK should be rank 1 (top)
        top_sector = ranks.iloc[0]["sector"]
        assert top_sector == "NIFTY BANK"

        assert ranker.is_top_sector("Banking", top_n=3)
        assert not ranker.is_top_sector("IT", top_n=3)


def test_alpha_engine_multiplier(mock_ohlcv):
    with patch("signals.sector_alpha.get_ohlcv", side_effect=mock_ohlcv):
        engine = AlphaEngine()

        # Case 1: Stock in top sector
        mult_top = engine.calculate_multiplier("SBIN", "Banking", "normal", "BUY")
        assert mult_top > 1.0
        assert mult_top == 1.2

        # Case 2: Stock in bottom sector
        mult_bot = engine.calculate_multiplier("INFY", "IT", "normal", "BUY")
        assert mult_bot == 1.0

        # Case 3: Sell side for bottom sector
        mult_sell = engine.calculate_multiplier("INFY", "IT", "normal", "SELL")
        assert mult_sell > 1.0
        assert mult_sell == 1.2
