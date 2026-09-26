"""Yesterday's recording, replayed under today's fixed settings.

A book that paid for twelve hours can stop paying the next day; the only
defence is to keep checking. Every morning this replays the previous UTC day
of each live-recorded book with one fixed configuration — the one the
twelve-hour bitbank ADA replay settled on — and reports the quoting edge
(10秒内計), the result after fees, and the fill count.

The settings are fixed on purpose. Choosing the best of several settings
each day and reporting it would be the in-sample selection trap on a daily
schedule: some setting always looks good in hindsight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

FIXED_FLAGS = (
    "--min-half-spread", "1",
    "--min-edge-bps", "2",
    "--lead-threshold-bps", "2",
    "--max-inventory-age-s", "120",
    "--cancel-ahead", "0.5",
    "--requote-ms", "250",
    "--latency-ms", "0.5",
    "--cancel-latency-ms", "50",
    "--min-book-levels", "2",
)
"""The configuration that held up on twelve hours of bitbank ADA."""

MAKER_BPS = {"bitbank": -2.0, "gmo": -1.0}
"""Per venue; GMO's altcoin books pay 3bps, so check the fee table for those."""

ALERT_AFTER_DAYS = 2
"""Consecutive days of a negative quoting edge before Slack is told to worry."""


@dataclass(frozen=True, slots=True)
class Target:
    venue: str
    symbol: str

    @classmethod
    def parse(cls, text: str) -> Target:
        venue, _, symbol = text.partition(":")
        if venue not in MAKER_BPS or not symbol:
            raise ValueError(f"{text}: bitbank:ada_jpy / gmo:SOL の形で指定してください")
        return cls(venue, symbol)

    @property
    def folder_symbol(self) -> str:
        return self.symbol.upper()

    @property
    def label(self) -> str:
        return f"{self.venue} {self.symbol}"


def day_folder(bucket: str, prefix: str, target: Target, date: str) -> str:
    return f"s3://{bucket}/{prefix.strip('/')}/symbol={target.folder_symbol}/date={date}/"


def size_for(order_jpy: float, price: float, lot: Decimal) -> str:
    """Base units for an order worth about `order_jpy`, on the lot grid."""
    # Whole lots, not quantize(lot): quantizing to Decimal("10") keeps unit
    # precision and would order 41 on a venue that trades in tens.
    lots = (Decimal(str(order_jpy)) / Decimal(str(price)) / lot).to_integral_value(ROUND_DOWN)
    units = max(lots, Decimal(1)) * lot
    return f"{units.normalize():f}"


@dataclass(slots=True)
class DayResult:
    target: Target
    fills: int = 0
    short_bps: float = math.nan
    after_fees_bps: float = math.nan
    note: str = ""

    def record(self) -> dict:
        return {
            "target": f"{self.target.venue}:{self.target.symbol}",
            "fills": self.fills,
            "short_bps": None if math.isnan(self.short_bps) else round(self.short_bps, 3),
            "after_fees_bps": (
                None if math.isnan(self.after_fees_bps) else round(self.after_fees_bps, 3)
            ),
            "note": self.note,
        }


def losing_streak(key: str, today: float, history: list[dict]) -> int:
    """Days in a row, ending today, with a negative quoting edge."""
    if math.isnan(today) or today >= 0:
        return 0
    streak = 1
    for day in history:  # newest first
        row = next((r for r in day.get("results", []) if r.get("target") == key), None)
        if row is None or row.get("short_bps") is None or row["short_bps"] >= 0:
            break
        streak += 1
    return streak


def slack_text(date: str, results: list[DayResult], history: list[dict]) -> str:
    lines = [f"*日次検証 {date}（UTC の1日分、設定は固定）*"]
    for r in results:
        if r.note:
            lines.append(f"• {r.target.label}: {r.note}")
            continue
        key = f"{r.target.venue}:{r.target.symbol}"
        streak = losing_streak(key, r.short_bps, history)
        flag = f"  :warning: {streak}日連続マイナス" if streak >= ALERT_AFTER_DAYS else ""
        lines.append(
            f"• {r.target.label}: 10秒内計 {r.short_bps:+.2f}bps  "
            f"手数料後 {r.after_fees_bps:+.2f}bps  約定 {r.fills:,}{flag}"
        )
    lines.append("_10秒内計 = スプレッド + 10秒以内の在庫損益 + 手数料（その日の値動きの運を除いた実力）。_")
    return "\n".join(lines)
