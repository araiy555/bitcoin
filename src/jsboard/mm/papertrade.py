"""Bookkeeping for paper trading against a live book.

The replays measured the strategy on recorded data; paper trading runs the
same maker on the live feed with simulated fills, around the clock, so the
day's result arrives as a number rather than as a claim. What this module
holds is only the daily ledger: the maker's own summary is cumulative, and
Slack wants the day that just ended.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DailyTally:
    """Cumulative totals at the start of the current UTC day."""

    day: str
    start_total: float = 0.0
    start_fills: int = 0
    start_fees: float = 0.0
    start_volume: float = 0.0
    start_filled: float = 0.0
    start_gap: float = 0.0

    def roll(self, today: str, summary: dict) -> dict | None:
        """Close the day once the date changes; return that day's figures."""
        if today == self.day:
            return None
        total = float(summary.get("total", 0.0))
        fills = int(summary.get("fills", 0))
        fees = float(summary.get("fees", 0.0))
        volume = float(summary.get("volume", 0.0))
        filled = float(summary.get("filled", 0.0))
        gap = float(summary.get("gap_filled", 0.0))
        report = {
            "day": self.day,
            "pnl": total - self.start_total,
            "fills": fills - self.start_fills,
            "fees": fees - self.start_fees,
            "position": summary.get("position", 0.0),
            # Size traded, and how much of it the venue model filled because
            # the book moved through our price rather than a print reaching
            # it: the two numbers that separate paper from the replay.
            "volume": volume - self.start_volume,
            "gap_share": (
                (gap - self.start_gap) / (filled - self.start_filled) * 100
                if filled > self.start_filled else None
            ),
        }
        self.day, self.start_total, self.start_fills, self.start_fees = today, total, fills, fees
        self.start_volume, self.start_filled, self.start_gap = volume, filled, gap
        return report


def slack_text(label: str, quote: str, base: str, report: dict, cumulative: dict) -> str:
    """One day of paper trading, then the running total, for Slack."""
    fee_word = "リベート" if report["fees"] <= 0 else "手数料"
    return "\n".join(
        [
            f"*紙上トレード {label} {report['day']}（UTC）*",
            f"損益 {report['pnl']:+,.0f} {quote}（{fee_word} {abs(report['fees']):,.0f} 込み）"
            f"  約定 {report['fills']:,} 回  在庫 {report['position']:+,.4g} {base}",
            *(
                [f"約定量 {report['volume']:,.0f} {base}"
                 + (f"（約 {report['notional'] / 1e4:,.0f}万円分）" if report.get("notional") else "")
                 + (f"  飛び越え約定 {report['gap_share']:.0f}%"
                    if report.get("gap_share") is not None else "")]
                if "volume" in report else []
            ),
            f"開始からの合計: 損益 {cumulative['total']:+,.0f} {quote}"
            f"  約定 {int(cumulative['fills']):,} 回",
            "_仮想の注文です。約定は本物の板と約定履歴から推定しています。_",
        ]
    )
