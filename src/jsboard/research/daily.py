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
    pnl: float = math.nan
    """The day's profit in the quote currency (yen), fees and inventory included."""
    notional: float = math.nan
    """Yen traded, at the day's opening price."""
    gap_share: float = math.nan
    """Share of filled size the model filled on a book move, not a print."""
    note: str = ""
    trial: dict | None = None
    """The same day under the book's trial settings, when it has any."""

    def record(self) -> dict:
        return {
            **({"trial": self.trial} if self.trial else {}),
            "target": f"{self.target.venue}:{self.target.symbol}",
            "fills": self.fills,
            "short_bps": None if math.isnan(self.short_bps) else round(self.short_bps, 3),
            "after_fees_bps": (
                None if math.isnan(self.after_fees_bps) else round(self.after_fees_bps, 3)
            ),
            "pnl": None if math.isnan(self.pnl) else round(self.pnl),
            "notional": None if math.isnan(self.notional) else round(self.notional),
            "gap_share": None if math.isnan(self.gap_share) else round(self.gap_share, 1),
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


def _volume(r: DayResult) -> str:
    if math.isnan(r.notional):
        return ""
    gap = "" if math.isnan(r.gap_share) else f"・飛び越え {r.gap_share:.0f}%"
    return f"（約 {r.notional / 1e4:,.0f}万円分{gap}）"


def _yen(value: float) -> str:
    return "損益 —" if math.isnan(value) else f"*損益 {value:+,.0f}円*"


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
            f"• {r.target.label}: {_yen(r.pnl)}  10秒内計 {r.short_bps:+.2f}bps  "
            f"手数料後 {r.after_fees_bps:+.2f}bps  約定 {r.fills:,}{_volume(r)}{flag}"
        )
        if r.trial:
            t = r.trial
            if t.get("short_bps") is None:
                lines.append(f"    試験中（{t['settings']}）: {t.get('note') or '結果なし'}")
            else:
                pnl = t.get("pnl")
                lines.append(
                    f"    試験中（{t['settings']}）: "
                    f"{_yen(math.nan if pnl is None else pnl)}  10秒内計 {t['short_bps']:+.2f}bps  "
                    f"手数料後 {t['after_fees_bps']:+.2f}bps  約定 {t['fills']:,}"
                )
    lines.append(
        "_損益 = 1回約1万円で1日出した場合の円の損益（リベートと在庫の値動き込み）。"
        "10秒内計 = スプレッド + 10秒以内の在庫損益 + 手数料（その日の値動きの運を除いた実力）。_"
    )
    return "\n".join(lines)


HALT_PREFIX = "control/halt"
"""A flag here stops paper (and later live) quoting for one book until removed."""

PAPER_PREFIX = "reports/paper"

READY_DAYS = 10
READY_MIN_POSITIVE = 8
"""Go-live bar: of the last ten daily replays, eight with a positive quoting
edge, no two-day losing streak among them, and paper trading ahead overall."""


TARGETS_PREFIX = "control/targets"
"""What was recorded each UTC day, so the morning replay checks all of it."""


def targets_key(day: str) -> str:
    return f"{TARGETS_PREFIX}/{day}.json"


def halt_key(target: Target) -> str:
    return f"{HALT_PREFIX}/{target.venue}_{target.symbol}.json"


def live_halt_key(target: Target) -> str:
    """Set when live trading stops on a loss; only a person clears it."""
    return f"{HALT_PREFIX}/live_{target.venue}_{target.symbol}.json"


def paper_key(target: Target, day: str) -> str:
    return f"{PAPER_PREFIX}/{day}/{target.venue}_{target.symbol}.json"


def _edges(key: str, today: float, history: list[dict]) -> list[float | None]:
    """Quoting edge per day, today first, None where the day had no replay."""
    out: list[float | None] = [None if math.isnan(today) else today]
    for day in history:
        row = next((r for r in day.get("results", []) if r.get("target") == key), None)
        out.append(None if row is None else row.get("short_bps"))
    return out


@dataclass(frozen=True, slots=True)
class Readiness:
    days: int
    positive: int
    streak_seen: bool
    paper_total: float | None

    @property
    def ready(self) -> bool:
        return (
            self.days >= READY_DAYS
            and self.positive >= READY_MIN_POSITIVE
            and not self.streak_seen
            and self.paper_total is not None
            and self.paper_total > 0
        )

    def text(self) -> str:
        def mark(ok: bool) -> str:
            return ":white_check_mark:" if ok else ":x:"

        paper = "記録なし" if self.paper_total is None else f"{self.paper_total:+,.0f}円"
        verdict = (
            "*本番に進める条件を満たしました*"
            if self.ready
            else f"条件まで あと {max(0, READY_DAYS - self.days)} 日分のデータが必要"
            if self.days < READY_DAYS
            else "条件を満たしていません"
        )
        return (
            f"  本番の条件: {mark(self.positive >= READY_MIN_POSITIVE)} 実力プラス "
            f"{self.positive}/{self.days}日（{READY_MIN_POSITIVE}/{READY_DAYS}以上）  "
            f"{mark(not self.streak_seen)} 2日連続マイナスなし  "
            f"{mark(self.paper_total is not None and self.paper_total > 0)} 紙上の合計 {paper}"
            f"\n  → {verdict}"
        )


def readiness(
    key: str, today: float, history: list[dict], paper_total: float | None
) -> Readiness:
    edges = _edges(key, today, history)[:READY_DAYS]
    known = [e for e in edges if e is not None]
    streak = any(
        a is not None and b is not None and a < 0 and b < 0 for a, b in zip(edges, edges[1:], strict=False)
    )
    return Readiness(len(known), sum(e > 0 for e in known), streak, paper_total)


SETTINGS_PREFIX = "control/settings"
"""A trial change to one book's settings, applied by paper within a minute
and replayed beside the fixed settings each morning. Deleting it reverts."""

TUNABLE = {
    "inventory_skew_bps": ("--inventory-skew-bps", "在庫の片寄せ", "bps"),
}
"""Settings a trial may change: name → (flag, label, unit)."""

TRIAL_REVERT_DAYS = 2
"""Days in a row a trial may trail the fixed settings before it is removed."""


def settings_key(target: Target) -> str:
    return f"{SETTINGS_PREFIX}/{target.venue}_{target.symbol}.json"


def parse_setting(text: str) -> tuple[str, float]:
    name, sep, value = text.partition("=")
    name = name.strip().replace("-", "_")
    if not sep or name not in TUNABLE:
        known = ", ".join(TUNABLE)
        raise ValueError(f"{text}: 変えられる設定は {known} です（例: inventory_skew_bps=20）")
    return name, float(value)


def settings_flags(settings: dict) -> list[str]:
    flags: list[str] = []
    for name, value in sorted(settings.items()):
        if name in TUNABLE:
            flags += [TUNABLE[name][0], f"{value:g}"]
    return flags


def describe_settings(settings: dict) -> str:
    parts = [
        f"{TUNABLE[name][1]} {value:g}{TUNABLE[name][2]}"
        for name, value in sorted(settings.items())
        if name in TUNABLE
    ]
    return "、".join(parts) if parts else "元の設定"


def trial_trailing_days(key: str, today: dict | None, history: list[dict]) -> int:
    """Days in a row, ending today, the trial finished behind the fixed settings."""

    def behind(row: dict | None) -> bool:
        trial = (row or {}).get("trial")
        if not trial or trial.get("after_fees_bps") is None or row.get("after_fees_bps") is None:
            return False
        return trial["after_fees_bps"] < row["after_fees_bps"]

    if not behind(today):
        return 0
    days = 1
    for day in history:  # newest first
        row = next((r for r in day.get("results", []) if r.get("target") == key), None)
        if not behind(row):
            break
        days += 1
    return days
