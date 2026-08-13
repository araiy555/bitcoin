"""First-pass screen: kill a symbol before writing any code for it.

The WIFUSDT study cost days and ended at a conclusion available in five
minutes from three numbers. It is worth stating the arithmetic that would
have stopped it:

    tick    = 0.0001 on a 0.1356 price  ->  7.37 bps per tick
    spread  = 1 tick                    ->  7.37 bps
    maker   = 10 bps                    ->  20 bps round trip

Quoting both sides at the touch and getting filled on both earns the spread,
once, and that is the *ceiling* — no strategy exceeds it, because there is
nothing above the touch to capture. 7.37 against 20 is not a tuning problem
or a signal problem. It is not close.

So the screen is deliberately crude and deliberately generous. It grants the
best case that physics allows and then asks whether the fee alone eats it:

    headroom = spread_bps − 2 × maker_bps

A symbol that fails this cannot be rescued by anything downstream, and the
screen is worth exactly that much: it is a *reject* filter, not a promise.
Passing means "not yet disproved", which is why the pipeline continues with
simulation on the survivors rather than stopping here.

Two secondary readings matter as much as the headroom, and both come from the
same three numbers:

  **spread in ticks.** A one-tick spread leaves no position between joining
  the touch and not trading at all. Inventory cannot be skewed, quotes cannot
  be nudged, and the only two states are "in front of everyone" and
  "invisible". Room to manoeuvre needs at least two or three ticks.

  **tick width in bps.** A coarse tick sets a floor on two separate costs.
  The smallest move the price can make is one tick, so adverse selection
  cannot be cheaper than that — WIF lost ~2.9 bps in the 0-100ms bucket
  against a 7.37 bps tick. And crossing to hedge costs about half a spread,
  which on a two-tick book is one whole tick. Both scale with tick width, and
  on WIF they came to ~25 bps against 7.5 bps of captured spread.

  The 2 bps default ceiling is a judgement, not a measurement: the loss/tick
  ratio rests on a single symbol. It is set where it is because a symbol that
  fails it cannot be rescued downstream and a rejected symbol costs nothing,
  whereas a wrongly accepted one costs an hour of recording and a day of
  analysis. Raise it with --max-tick-bps to see what it is excluding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class Triage:
    """One symbol judged on fee, tick and spread alone."""

    symbol: str
    mid: float
    tick_size: float
    spread_bps: float
    quote_volume: float = 0.0
    trades: int = 0
    """24h trade count. A wide spread on a book nobody trades is wide because
    it is dead, not because it is an opportunity — and a maker there fills a
    handful of times a day. Volume alone does not catch this: one large
    transfer can carry a symbol past a volume floor it never traded through."""

    @property
    def tick_bps(self) -> float:
        """What one tick is worth, in basis points of the price."""
        if self.mid <= 0 or self.tick_size <= 0:
            return math.inf
        return self.tick_size / self.mid * 10_000.0

    @property
    def spread_ticks(self) -> float:
        """How many ticks wide the spread is — the room to manoeuvre."""
        tick = self.tick_bps
        if not math.isfinite(tick) or tick <= 0:
            return 0.0
        return self.spread_bps / tick

    def headroom_bps(self, maker_bps: float) -> float:
        """Best case minus the unavoidable fee, per round trip.

        The best case is the whole spread: quoting both sides at the touch and
        being filled on both. Nothing beats it, so a negative number here
        settles the symbol outright.
        """
        return self.spread_bps - 2.0 * maker_bps

    def verdict(
        self, maker_bps: float, *, min_ticks: float = 2.0, max_tick_bps: float = 2.0
    ) -> str:
        if self.spread_bps <= 0:
            return "板なし"
        if self.headroom_bps(maker_bps) <= 0:
            return "手数料負け"
        if self.tick_bps > max_tick_bps:
            return "tickが太い"
        if self.spread_ticks < min_ticks:
            return "tickが粗い"
        return "候補"

    def passes(
        self, maker_bps: float, *, min_ticks: float = 2.0, max_tick_bps: float = 2.0
    ) -> bool:
        return (
            self.verdict(maker_bps, min_ticks=min_ticks, max_tick_bps=max_tick_bps) == "候補"
        )


def build(
    symbol: str,
    *,
    bid: float,
    ask: float,
    tick_size: float | Decimal,
    quote_volume: float = 0.0,
    trades: int = 0,
    spread_bps: float | None = None,
) -> Triage | None:
    """A triage row from a top of book, or None when there is no book.

    `spread_bps` overrides the snapshot when several observations have been
    folded into a median. One look at a thin book is close to worthless — the
    same symbol was seen at 4.93, 7.11 and 11.70 bps in three runs seconds
    apart — and the ranking is only as good as that number.
    """
    if bid <= 0 or ask <= bid:
        return None
    mid = (bid + ask) / 2.0
    return Triage(
        symbol=symbol,
        mid=mid,
        tick_size=float(tick_size),
        spread_bps=(ask - bid) / mid * 10_000.0 if spread_bps is None else spread_bps,
        quote_volume=quote_volume,
        trades=trades,
    )


def rank(
    rows: list[Triage],
    maker_bps: float,
    *,
    min_ticks: float = 2.0,
    max_tick_bps: float = 2.0,
) -> list[Triage]:
    """Survivors first, by how much room is left after the fee.

    Headroom alone is a misleading key, which the first live run showed: the
    two widest symbols were both a 6.3 bps tick with a two-tick spread —
    WIFUSDT's shape almost exactly. Both costs that killed WIF scale with tick
    width, so the gates run first and the ranking only orders what survives
    them.
    """
    return sorted(
        (r for r in rows if r.passes(maker_bps, min_ticks=min_ticks, max_tick_bps=max_tick_bps)),
        key=lambda r: -r.headroom_bps(maker_bps),
    )


def tally(
    rows: list[Triage],
    maker_bps: float,
    *,
    min_ticks: float = 2.0,
    max_tick_bps: float = 2.0,
) -> dict[str, int]:
    """How many symbols died at each gate, which is the useful summary."""
    counts = {"板なし": 0, "手数料負け": 0, "tickが太い": 0, "tickが粗い": 0, "候補": 0}
    for row in rows:
        counts[row.verdict(maker_bps, min_ticks=min_ticks, max_tick_bps=max_tick_bps)] += 1
    return counts
