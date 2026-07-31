"""Terminal order-book display (the 板).

A depth ladder with asks stacked above bids, our own resting quotes called out
in the margin, and the maker's state — fair value, inventory, P&L, risk — in
panels around it. Everything renders from a `MarketMaker`, so the same view
works over a live feed, a replay, or a synthetic market.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.align import Align
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.types import Side
from ..mm.strategy import MarketMaker

BAR = "█"
MAX_BAR = 7
"""Depth-bar width. Kept short on purpose: the ladder has seven columns and
the price must never be the thing that gets truncated when the pane narrows."""

STATUS_STYLE = {
    "live": "bold green",
    "connecting": "bold yellow",
    "resyncing": "bold yellow",
    "disconnected": "bold red",
}


@dataclass(slots=True)
class BoardConfig:
    depth: int = 12
    tape_rows: int = 8
    show_queue: bool = True


class Board:
    """Renders a `MarketMaker` as a live terminal dashboard."""

    def __init__(self, mm: MarketMaker, config: BoardConfig | None = None) -> None:
        self.mm = mm
        self.config = config or BoardConfig()

    # ------------------------------------------------------------- layout

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(name="body"),
            Layout(self._footer(), name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(self._ladder(), name="ladder", ratio=5),
            Layout(name="side", ratio=3),
        )
        layout["side"].split_column(
            Layout(self._position_panel(), name="position"),
            Layout(self._signals_panel(), name="signals"),
            Layout(self._tape_panel(), name="tape"),
        )
        return layout

    # ------------------------------------------------------------- header

    def _header(self) -> Panel:
        m = self.mm.market
        inst = m.instrument
        status = Text(m.status.upper(), style=STATUS_STYLE.get(m.status, "white"))

        mid = m.mid_price
        mid_txt = f"{mid:,.2f}" if mid is not None else "—"
        spread = m.spread_ticks
        spread_txt = f"{spread} tick" if spread is not None else "—"

        line = Text.assemble(
            (f"{inst.symbol}  ", "bold cyan"),
            ("│ ", "dim"),
            ("mid ", "dim"),
            (mid_txt, "bold white"),
            ("  │ ", "dim"),
            ("spread ", "dim"),
            (spread_txt, "white"),
            ("  │ ", "dim"),
            ("status ", "dim"),
            status,
            ("  │ ", "dim"),
            ("age ", "dim"),
            (f"{m.age_ms:,.0f}ms" if m.age_ms != float("inf") else "—", "white"),
            ("  │ ", "dim"),
            ("events ", "dim"),
            (f"{m.events_seen:,}", "white"),
        )
        return Panel(Align.center(line), border_style="cyan", title="jsboard", title_align="left")

    # ------------------------------------------------------------- ladder

    def _our_sizes(self) -> dict[tuple[int, int], tuple[int, int]]:
        """(side, price) -> (our resting lots, queue ahead)."""
        out: dict[tuple[int, int], tuple[int, int]] = {}
        for order in self.mm.venue.open_orders():
            key = (int(order.side), order.price)
            lots, ahead = out.get(key, (0, order.queue_ahead))
            out[key] = (lots + order.remaining, min(ahead, order.queue_ahead))
        return out

    def _ladder(self) -> Panel:
        cfg = self.config
        inst = self.mm.market.instrument
        snap = self.mm.market.snapshot(cfg.depth)
        ours = self._our_sizes()

        table = Table(
            show_header=True, header_style="bold dim", box=None, expand=True, padding=(0, 1)
        )
        table.add_column("ours", justify="right", width=11)
        table.add_column("bid", justify="right", width=7)
        table.add_column("depth", justify="left", width=MAX_BAR)
        table.add_column("price", justify="center", width=10)
        table.add_column("depth ", justify="right", width=MAX_BAR)
        table.add_column("ask", justify="left", width=7)
        table.add_column("ours ", justify="left", width=11)

        peak = max(
            [lvl.qty for lvl in snap.bids] + [lvl.qty for lvl in snap.asks] or [1]
        )

        # A quote resting *inside* the spread sits at a price with no public
        # depth, so it would never appear if the ladder were built from market
        # levels alone — and those are precisely the quotes at the front of the
        # queue. Build each side from the union of both instead.
        bid_rows = self._rows(snap.bids, ours, Side.BUY, cfg.depth)
        ask_rows = self._rows(snap.asks, ours, Side.SELL, cfg.depth)

        # Asks descend toward the touch, so the spread sits in the middle.
        for price, qty in reversed(ask_rows):
            our = ours.get((int(Side.SELL), price))
            table.add_row(
                "",
                "",
                "",
                Text(f"{inst.price_f(price):,.2f}", style="red"),
                self._bar(qty, peak, "red", right=True),
                Text(f"{inst.qty_f(qty):,.4f}" if qty else "", style="red"),
                self._ours_cell(our, "bold red"),
            )

        table.add_row(*self._spread_row(snap))

        for price, qty in bid_rows:
            our = ours.get((int(Side.BUY), price))
            table.add_row(
                self._ours_cell(our, "bold green", right=True),
                Text(f"{inst.qty_f(qty):,.4f}" if qty else "", style="green"),
                self._bar(qty, peak, "green", right=False),
                Text(f"{inst.price_f(price):,.2f}", style="green"),
                "",
                "",
                "",
            )

        return Panel(table, title="板 / depth", border_style="blue", title_align="left")

    def _rows(self, levels, ours, side: Side, depth: int) -> list[tuple[int, int]]:
        """(price, public qty) for each row, ours included even at empty prices."""
        qty_by_price = {lvl.price: lvl.qty for lvl in levels}
        prices = set(qty_by_price)
        prices |= {price for (s, price) in ours if s == int(side)}
        ordered = sorted(prices, reverse=side is Side.BUY)[:depth]
        return [(p, qty_by_price.get(p, 0)) for p in ordered]

    def _spread_row(self, snap) -> list:
        inst = self.mm.market.instrument
        fair = self.mm.last_quotes.fair_value
        reservation = self.mm.last_quotes.reservation

        middle = Text("─" * 12, style="dim")
        if fair is not None:
            middle = Text(f"{inst.price_f(int(round(fair))):,.2f}", style="bold yellow")

        left = Text("fair →", style="dim yellow")
        right = Text("← resv", style="dim magenta")
        resv_txt = (
            Text(f"{inst.price_f(int(round(reservation))):,.2f}", style="magenta")
            if reservation is not None
            else Text("")
        )
        spread_txt = Text(
            f"{snap.spread} tick" if snap.spread is not None else "—", style="dim"
        )
        return [Text(""), spread_txt, left, middle, right, resv_txt, Text("")]

    def _bar(self, qty: int, peak: int, colour: str, right: bool) -> Text:
        if qty <= 0:
            # A row that exists only because we quote there has no public depth.
            return Text(" " * MAX_BAR)
        width = max(1, int(MAX_BAR * qty / peak)) if peak else 0
        bar = BAR * width
        padded = bar.rjust(MAX_BAR) if right else bar.ljust(MAX_BAR)
        return Text(padded, style=colour)

    def _ours_cell(self, our, style: str, right: bool = False) -> Text:
        if our is None:
            return Text("")
        lots, ahead = our
        inst = self.mm.market.instrument
        label = f"{inst.qty_f(lots):,.4f}"
        if self.config.show_queue and ahead > 0:
            # How much size still has to trade before we do.
            label += f" q{inst.qty_f(ahead):,.1f}"
        return Text(label, style=style)

    # ------------------------------------------------------------- panels

    def _position_panel(self) -> Panel:
        mm = self.mm
        inst = mm.market.instrument
        s = mm.position.summary(mm.market.mid)

        t = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        t.add_column(style="dim", width=12)
        t.add_column(justify="right")

        pos = s["position"]
        pos_style = "green" if pos > 0 else "red" if pos < 0 else "dim"
        t.add_row("position", Text(f"{pos:+,.5f} {inst.base or ''}".strip(), style=pos_style))
        t.add_row("avg price", f"{s['avg_price']:,.2f}" if pos else "—")
        t.add_row("realized", self._money(s["realized"]))
        t.add_row("unrealized", self._money(s["unrealized"]))
        t.add_row("total P&L", self._money(s["total"], bold=True))
        t.add_row("fees", Text(f"-{s['fees']:,.2f}", style="dim red"))
        t.add_row("fills", f"{int(s['fills']):,}")
        t.add_row("volume", f"{s['volume']:,.5f}")

        return Panel(t, title="position", border_style="magenta", title_align="left")

    def _signals_panel(self) -> Panel:
        mm = self.mm
        m = mm.market
        q = mm.last_quotes

        t = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        t.add_column(style="dim", width=12)
        t.add_column(justify="right")

        micro = m.microprice
        t.add_row("microprice", f"{m.instrument.price_f(int(round(micro))):,.2f}" if micro else "—")
        t.add_row("imbalance", self._signed(m.imbalance(5)))
        t.add_row("flow", self._signed(m.flow.value))
        t.add_row("σ (ticks)", f"{mm.sigma_ticks:,.2f}")
        t.add_row("σ (bps)", f"{m.vol.bps:,.2f}")
        t.add_row(
            "half spread", f"{q.half_spread:,.2f} tick" if q.half_spread is not None else "—"
        )
        t.add_row("resting", f"{len(mm.venue.open_orders())}")
        t.add_row("cycles", f"{mm.stats.cycles:,}")

        return Panel(t, title="signals", border_style="yellow", title_align="left")

    def _tape_panel(self) -> Panel:
        inst = self.mm.market.instrument
        trades = self.mm.market.recent_trades(self.config.tape_rows)
        our_fill_ids = {f.ts_ns for f in self.mm.recent_fills[-40:]}

        t = Table(show_header=False, box=None, expand=True, padding=(0, 1))
        t.add_column(width=10)
        t.add_column(justify="right")
        t.add_column(justify="right", width=6)

        for trade in reversed(trades):
            style = "green" if trade.aggressor is Side.BUY else "red"
            marker = "◆" if trade.ts_ns in our_fill_ids else " "
            t.add_row(
                Text(f"{marker} {inst.price_f(trade.price):,.2f}", style=style),
                Text(f"{inst.qty_f(trade.qty):,.4f}", style=style),
                Text("BUY" if trade.aggressor is Side.BUY else "SELL", style=f"dim {style}"),
            )
        if not trades:
            t.add_row(Text("no prints yet", style="dim"), "", "")

        return Panel(t, title="tape", border_style="green", title_align="left")

    # ------------------------------------------------------------- footer

    def _footer(self) -> Panel:
        mm = self.mm
        decision = mm.last_decision
        if decision is None:
            body = Text("waiting for first requote…", style="dim")
        else:
            colour = {
                "QUOTE": "green",
                "ONE_SIDED": "yellow",
                "PULL": "yellow",
                "HALT": "bold red",
            }.get(decision.action.value, "white")
            body = Text.assemble(
                ("risk ", "dim"),
                (decision.action.value, colour),
                ("  │ ", "dim"),
                (decision.reason, "white"),
                ("  │ ", "dim"),
                ("placed ", "dim"),
                (f"{mm.stats.orders_placed:,}", "white"),
                ("  cancelled ", "dim"),
                (f"{mm.stats.orders_cancelled:,}", "white"),
                ("  kept ", "dim"),
                (f"{mm.stats.orders_kept:,}", "green"),
                ("  fills ", "dim"),
                (f"{mm.stats.fills:,}", "bold cyan"),
            )
        return Panel(Align.center(body), border_style="dim")

    # ------------------------------------------------------------ helpers

    def _money(self, value: float, bold: bool = False) -> Text:
        style = "green" if value > 0 else "red" if value < 0 else "dim"
        if bold:
            style = f"bold {style}"
        return Text(f"{value:+,.2f}", style=style)

    def _signed(self, value: float) -> Text:
        style = "green" if value > 0.05 else "red" if value < -0.05 else "dim"
        return Text(f"{value:+.3f}", style=style)


def static_summary(mm: MarketMaker) -> Group:
    """Non-live render, for the end of a headless run."""
    board = Board(mm)
    return Group(board._header(), board._position_panel(), board._signals_panel())
