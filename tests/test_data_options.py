"""Tests for data/options_scraper.py — OptionsPCRScraper."""

from __future__ import annotations

from unittest.mock import MagicMock

from data.options_scraper import OptionsPCRScraper


def _make_nse_response(ce_oi: int = 500, pe_oi: int = 800) -> dict:
    """Minimal NSE option chain response structure."""
    return {
        "records": {
            "data": [
                {
                    "CE": {"openInterest": ce_oi, "strikePrice": 22000},
                    "PE": {"openInterest": pe_oi, "strikePrice": 22000},
                },
                {
                    "CE": {"openInterest": ce_oi + 100, "strikePrice": 22100},
                    "PE": {"openInterest": pe_oi + 50, "strikePrice": 22100},
                },
            ]
        }
    }


class TestOptionsPCRScraper:
    def _scraper_with_mock_session(self, response_data: dict) -> OptionsPCRScraper:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = response_data
        mock_resp.raise_for_status.return_value = None
        mock_session.get.return_value = mock_resp
        return OptionsPCRScraper(session=mock_session)

    def test_get_pcr_returns_float(self):
        scraper = self._scraper_with_mock_session(_make_nse_response(ce_oi=500, pe_oi=1000))
        pcr = scraper.get_pcr("NIFTY")
        assert isinstance(pcr, float)

    def test_pcr_calculation_correct(self):
        """PCR = total PE OI / total CE OI."""
        scraper = self._scraper_with_mock_session(_make_nse_response(ce_oi=500, pe_oi=1000))
        pcr = scraper.get_pcr("NIFTY")
        # CE total: 500 + 600 = 1100; PE total: 1000 + 850 = 1850
        # PCR ≈ 1850/1100 ≈ 1.68
        assert pcr > 1.0

    def test_pcr_below_one_when_ce_dominates(self):
        scraper = self._scraper_with_mock_session(_make_nse_response(ce_oi=2000, pe_oi=500))
        pcr = scraper.get_pcr("NIFTY")
        assert pcr < 1.0

    def test_get_pcr_signal_bullish(self):
        """High PCR (PE > CE) = bearish put build-up = bullish for contrarians."""
        scraper = self._scraper_with_mock_session(_make_nse_response(ce_oi=500, pe_oi=2000))
        signal = scraper.get_pcr_signal("NIFTY")
        assert signal in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_get_pcr_signal_bearish(self):
        scraper = self._scraper_with_mock_session(_make_nse_response(ce_oi=2000, pe_oi=200))
        signal = scraper.get_pcr_signal("NIFTY")
        assert signal in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_get_pcr_signal_neutral_range(self):
        scraper = self._scraper_with_mock_session(_make_nse_response(ce_oi=1000, pe_oi=1000))
        signal = scraper.get_pcr_signal("NIFTY")
        assert signal == "NEUTRAL"

    def test_empty_data_returns_none(self):
        scraper = self._scraper_with_mock_session({"records": {"data": []}})
        pcr = scraper.get_pcr("NIFTY")
        assert pcr is None

    def test_api_error_returns_none(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("NSE API unreachable")
        scraper = OptionsPCRScraper(session=mock_session)
        pcr = scraper.get_pcr("NIFTY")
        assert pcr is None

    def test_get_pcr_signal_returns_neutral_on_none_pcr(self):
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("timeout")
        scraper = OptionsPCRScraper(session=mock_session)
        signal = scraper.get_pcr_signal("NIFTY")
        assert signal == "NEUTRAL"
