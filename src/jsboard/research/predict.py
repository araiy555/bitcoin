"""Does any signal beat the accuracy the fee demands?

The horizon study fixed the target: at one hour, trading only the most
volatile tenth of moments, direction has to be right 55.4% of the time to
break even. Everything shorter is closed. So the remaining question is
narrow and answerable — can anything in the tape reach that number?

Three things decide whether an answer here means anything.

**The split is chronological.** Choosing a signal on the same data used to
report its accuracy measures how well the choice was made, not how well the
signal works. Candidates are ranked on the earlier days and the winner is
reported on the later ones, once.

**The baseline is "always long".** BTC rose over most windows anyone will
test, so a coin that always says up scores above 50% and looks like a signal.
The number that matters is the margin over that, not the level.

**Both directions of every signal are candidates.** Order flow could predict
continuation or reversal; assuming which without checking would be picking an
answer rather than measuring one. The cost is that with N candidates the best
one is partly luck — hence a single test-set report and no re-selection.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from .archive import SecondBar

WINDOWS = (60, 300, 900)


@dataclass(frozen=True, slots=True)
class Series:
    """Per-bar arrays plus prefix sums, so any window costs a subtraction."""

    secs: list[int]
    price: list[float]
    cum_signed: list[float]
    cum_volume: list[float]
    cum_absret: list[float]

    @classmethod
    def build(cls, bars: list[SecondBar]) -> Series:
        secs = [b.sec for b in bars]
        price = [b.last for b in bars]
        signed = [0.0]
        volume = [0.0]
        absret = [0.0]
        for i, bar in enumerate(bars):
            signed.append(signed[-1] + bar.signed_qty)
            volume.append(volume[-1] + bar.buy_qty + bar.sell_qty)
            step = 0.0 if i == 0 else abs(price[i] - price[i - 1]) / price[i - 1] * 10_000.0
            absret.append(absret[-1] + step)
        return cls(secs, price, signed, volume, absret)

    def __len__(self) -> int:
        return len(self.secs)

    def start_of_window(self, i: int, seconds: int) -> int:
        """First bar at or after `seconds` before bar `i`."""
        return bisect.bisect_left(self.secs, self.secs[i] - seconds)

    def flow(self, i: int, seconds: int) -> float:
        """Signed volume over the window, as a share of volume traded.

        Normalising matters: raw signed volume is dominated by how busy the
        window was, so an unnormalised version would mostly rank activity.
        """
        j = self.start_of_window(i, seconds)
        vol = self.cum_volume[i + 1] - self.cum_volume[j]
        if vol <= 0:
            return 0.0
        return (self.cum_signed[i + 1] - self.cum_signed[j]) / vol

    def momentum(self, i: int, seconds: int) -> float:
        j = self.start_of_window(i, seconds)
        base = self.price[j]
        return (self.price[i] - base) / base * 10_000.0 if base else 0.0

    def volatility(self, i: int, seconds: int) -> float:
        """Sum of absolute per-second moves — a proxy, not an estimator."""
        j = self.start_of_window(i, seconds)
        return self.cum_absret[i + 1] - self.cum_absret[j]

    def forward(self, i: int, horizon_s: int) -> float | None:
        target = self.secs[i] + horizon_s
        if target > self.secs[-1]:
            return None
        j = bisect.bisect_right(self.secs, target) - 1
        return (self.price[j] - self.price[i]) / self.price[i] * 10_000.0


def candidates() -> list[tuple[str, str, int, int]]:
    """(name, kind, window, sign) for every signal considered.

    Both signs are included because whether flow leads to continuation or to
    reversal is exactly the sort of thing that should be measured rather than
    assumed.
    """
    out = []
    for kind in ("flow", "mom"):
        for w in WINDOWS:
            for sign in (1, -1):
                label = f"{kind}_{w}s{'' if sign > 0 else ' (逆張り)'}"
                out.append((label, kind, w, sign))
    return out


def signal_value(series: Series, i: int, kind: str, window: int, sign: int) -> float:
    raw = series.flow(i, window) if kind == "flow" else series.momentum(i, window)
    return raw * sign


@dataclass(frozen=True, slots=True)
class Score:
    name: str
    samples: int
    accuracy: float
    mean_move_bps: float
    cost_bps: float

    @property
    def edge_bps(self) -> float:
        """Expected bps per round trip at this accuracy on these moves."""
        return (2.0 * self.accuracy - 1.0) * self.mean_move_bps - self.cost_bps


def evaluate(
    series: Series,
    name: str,
    kind: str,
    window: int,
    sign: int,
    *,
    horizon_s: int,
    cost_bps: float,
    vol_window: int,
    vol_threshold: float | None,
) -> Score:
    """Accuracy of one signal, optionally only on the most volatile moments.

    `vol_threshold` comes from the training half. Recomputing it on the test
    half would leak the test period's volatility distribution into the
    decision of when to trade.
    """
    hits = 0
    n = 0
    total_move = 0.0
    warmup = max(window, vol_window)
    for i in range(len(series)):
        if series.secs[i] - series.secs[0] < warmup:
            continue  # not enough history behind this bar to form the signal
        if vol_threshold is not None and series.volatility(i, vol_window) < vol_threshold:
            continue
        fwd = series.forward(i, horizon_s)
        if fwd is None or fwd == 0.0:
            continue  # a flat window is neither a hit nor a miss
        value = signal_value(series, i, kind, window, sign)
        if value == 0.0:
            continue
        n += 1
        total_move += abs(fwd)
        if (value > 0) == (fwd > 0):
            hits += 1
    return Score(
        name=name,
        samples=n,
        accuracy=hits / n if n else 0.0,
        mean_move_bps=total_move / n if n else 0.0,
        cost_bps=cost_bps,
    )


def always_long(series: Series, *, horizon_s: int, cost_bps: float) -> Score:
    """The benchmark every signal has to beat, not 50%."""
    hits = n = 0
    total = 0.0
    for i in range(len(series)):
        fwd = series.forward(i, horizon_s)
        if fwd is None or fwd == 0.0:
            continue
        n += 1
        total += abs(fwd)
        if fwd > 0:
            hits += 1
    return Score("常に買い", n, hits / n if n else 0.0, total / n if n else 0.0, cost_bps)


def volatility_threshold(series: Series, window: int, quantile: float) -> float:
    values = sorted(
        series.volatility(i, window)
        for i in range(len(series))
        if series.secs[i] - series.secs[0] >= window
    )
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(quantile * len(values)))
    return values[idx]


def split(bars: list[SecondBar], train_fraction: float) -> tuple[list[SecondBar], list[SecondBar]]:
    """Chronological split. Never shuffled: shuffling would put future bars in
    the training half, and the resulting accuracy would be meaningless."""
    cut = int(len(bars) * train_fraction)
    return bars[:cut], bars[cut:]
