"""Real orders behind the maker: venue bookkeeping, executor, fills, guards."""

import asyncio
import time
from decimal import Decimal

import pytest

from jsboard.core.types import Instrument, Side
from jsboard.feed.base import DepthSnapshot, Feed, FeedStatus, TradeTick
from jsboard.live.bitbank import NOT_FOUND, BitbankError
from jsboard.live.runner import (
    Breaker,
    DryRunApi,
    Health,
    cancel_everything,
    execute_once,
    poll_fills_once,
    run_watchdog,
)
from jsboard.live.venue import CANCELLING, DONE, OPEN, LiveVenue
from jsboard.mm.quoter import Quote

INST = Instrument("ada_jpy", Decimal("0.001"), Decimal("0.0001"), "ADA", "JPY")
LEAD = Instrument("ADAUSDT", Decimal("0.0001"), Decimal("1"), "ADA", "USDT")


def venue(jpy="10000", ada="200"):
    return LiveVenue(INST, quote_balance=Decimal(jpy), base_balance=Decimal(ada))


def bid(price_ticks=37_000, lots=1_000_000):  # 100 ADA at 37.000
    return Quote(Side.BUY, price_ticks, lots)


def test_quotes_that_would_take_or_cannot_be_paid_for_are_refused():
    v = venue(jpy="5000", ada="50")
    assert v.place(bid(37_010), 0, best_opposite=37_010) is None  # crosses
    assert v.place(bid(), 0, best_opposite=37_010) is not None  # 3,700 yen: fits
    assert v.place(bid(), 0, best_opposite=37_010) is None  # the second does not
    assert v.place(Quote(Side.SELL, 37_020, 1_000_000), 0, best_opposite=36_990) is None
    assert v.place(Quote(Side.SELL, 37_020, 500_000), 0, best_opposite=36_990) is not None
    assert v.rejected == 3


def test_a_blocked_venue_places_nothing_but_still_cancels():
    v = venue()
    order = v.place(bid(), 0, best_opposite=37_010)
    v.blocked = "板のデータが止まっています"
    assert v.place(bid(36_990), 0, best_opposite=37_010) is None
    assert v.cancel_all() == 1 and order.state == CANCELLING
    assert v.open_orders() == []


@pytest.mark.asyncio
async def test_the_executor_sends_and_cancels_and_skips_what_was_cancelled_first():
    v, api, health = venue(), DryRunApi(), Health()
    first = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", health)
    assert first.state == OPEN and first.exchange_id == 1

    second = v.place(bid(36_990), 0, best_opposite=37_010)
    v.cancel(second.order_id)
    v.cancel(first.order_id)
    while await execute_once(v, api, "ada_jpy", health):
        pass
    assert second.state == DONE and second.exchange_id is None  # never sent
    assert first.state == DONE and api.open == set()
    assert api.orders_sent == 1


@pytest.mark.asyncio
async def test_a_cancel_that_arrives_before_the_order_lands_is_retried():
    class Slow(DryRunApi):
        def __init__(self):
            super().__init__()
            self.misses = 2

        async def cancel(self, pair, order_id):
            if self.misses:
                self.misses -= 1
                raise BitbankError(NOT_FOUND, "cancel")
            await super().cancel(pair, order_id)

    v, api, health = venue(), Slow(), Health()
    order = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", health)
    v.cancel(order.order_id)
    while await execute_once(v, api, "ada_jpy", health):
        pass
    assert order.state == DONE and api.open == set() and health.consecutive_errors == 0


@pytest.mark.asyncio
async def test_repeated_refusals_end_the_run():
    class Refusing(DryRunApi):
        async def order(self, pair, side, price, amount):
            raise BitbankError(20001, "order")  # authentication: the connection is sick

    v, api, health = venue(), Refusing(), Health(max_errors=3)
    for price in (37_000, 36_990, 36_980):
        v.place(bid(price, 10_000), 0, best_opposite=37_010)
    while await execute_once(v, api, "ada_jpy", health):
        pass
    assert "3回続けて失敗" in health.fatal


@pytest.mark.asyncio
async def test_our_executions_reach_the_maker_as_fills():
    from jsboard.cli import build_maker, build_parser
    from jsboard.research.daily import FIXED_FLAGS

    args = build_parser().parse_args([
        "sweep", "-", "--source", "bitbank", *FIXED_FLAGS, "--maker-bps=-2",
        "--size", "100", "--max-position", "500",
    ])
    mm = build_maker(INST, args)
    v = venue()
    mm.venue = v
    order = v.place(bid(), 0, best_opposite=37_010)
    api = DryRunApi()
    await execute_once(v, api, "ada_jpy", Health())

    class Filled(DryRunApi):
        async def trade_history(self, pair, since_ms):
            return [
                {"trade_id": 9, "order_id": order.exchange_id, "price": "37.000",
                 "amount": "40", "maker_taker": "maker", "executed_at": 1_790_000_000_000},
                {"trade_id": 10, "order_id": 999, "price": "37.000", "amount": "5",
                 "maker_taker": "maker", "executed_at": 1_790_000_000_001},  # not ours
            ]

    state = {}
    assert await poll_fills_once(v, Filled(), "ada_jpy", state) == 1
    assert await poll_fills_once(v, Filled(), "ada_jpy", state) == 0  # seen already
    mm.on_event(DepthSnapshot(((36_990, 9_000_000),), ((37_010, 9_000_000),), 1))
    assert mm.position.lots == INST.to_lots("40")
    assert v.base_balance == Decimal("240") and v.quote_balance == Decimal("10000") - 37 * 40
    assert order.remaining == INST.to_lots("60") and order.state == OPEN


def test_the_breaker_trips_on_a_fast_move_and_cools_down():
    b = Breaker(move_pct=1.5, window_s=60, cooldown_s=300)
    assert not b.observe(100.0, 0.0)
    assert not b.observe(101.0, 30.0)
    assert b.observe(98.4, 50.0)  # 2.6% range inside a minute
    assert b.active(100.0) and not b.active(351.0)


@pytest.mark.asyncio
async def test_the_watchdog_pulls_everything_when_the_feed_stops():
    v = venue()
    v.place(bid(), 0, best_opposite=37_010)
    task = asyncio.create_task(run_watchdog(
        v, Health(), lambda: 0.0, lambda: None, Breaker(), lambda text: None,
        stale_s=0.01, tick_s=0.01,
    ))
    await asyncio.sleep(0.05)
    task.cancel()
    assert "止まっています" in v.blocked and v.open_orders() == []


@pytest.mark.asyncio
async def test_every_run_ends_with_nothing_resting():
    v, api = venue(), DryRunApi()
    for price in (37_000, 36_990):
        v.place(bid(price, 10_000), 0, best_opposite=37_010)
        await execute_once(v, api, "ada_jpy", Health())
    api.open.add(77)  # an order the venue holds that we lost track of
    assert await cancel_everything(api, "ada_jpy", v) == []
    assert api.open == set()


class Book(Feed):
    async def stream(self):
        yield FeedStatus("live", "fake")
        for i in range(200):
            mid = 37_200 + (i // 40) % 3
            yield DepthSnapshot(tuple((mid - 5 - j, 9_000_000) for j in range(5)),
                                tuple((mid + 5 + j, 9_000_000) for j in range(5)), i)
            yield TradeTick(mid + 12 if i % 2 else mid - 12, 5_000_000,
                            Side.BUY if i % 2 else Side.SELL, i)
            await asyncio.sleep(0)


class Lead(Feed):
    async def stream(self):
        yield FeedStatus("live", "fake")
        yield DepthSnapshot(((2480, 90_000),), ((2481, 90_000),), 1)


def wire(monkeypatch, posted):
    import jsboard.cli as cli

    async def live_book(target):
        return INST, Book(INST)

    async def futures(symbol):
        return LEAD

    async def post(text):
        posted.append(text)
        return True

    monkeypatch.setattr(cli, "_live_book", live_book)
    monkeypatch.setattr(cli, "fetch_futures_instrument", futures)
    monkeypatch.setattr(cli, "_binance_lead", lambda inst, ms: Lead(inst))
    monkeypatch.setattr(cli, "_post_slack", post)
    return cli


@pytest.mark.asyncio
async def test_a_practice_run_quotes_without_sending_anything(monkeypatch, capsys):
    posted = []
    cli = wire(monkeypatch, posted)
    args = cli.build_parser().parse_args(["trade", "--slack", "--log-every", "0"])
    assert await args.func(args) == 1  # the fake feed ends
    out = capsys.readouterr().out
    assert "練習（注文は出しません）" in out and "注文中" in out
    assert "注文はすべて取り消しました" in posted[-1]


@pytest.mark.asyncio
async def test_a_loss_flag_keeps_the_run_from_starting(monkeypatch, capsys):
    import jsboard.sim.s3 as s3

    class Bucket:
        def head_object(self, Bucket, Key):  # noqa: N803
            if Key != "control/halt/live_bitbank_ada_jpy.json":
                raise KeyError(Key)

    monkeypatch.setattr(s3, "default_client", Bucket)
    cli = wire(monkeypatch, [])
    args = cli.build_parser().parse_args(["trade", "--s3-bucket", "b"])
    assert await args.func(args) == 0
    assert "始めません" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_stopping_the_service_still_cancels_everything(monkeypatch):
    posted = []
    cli = wire(monkeypatch, posted)

    class Endless(Feed):
        async def stream(self):
            yield FeedStatus("live", "fake")
            while True:
                yield DepthSnapshot(((37_195, 9_000_000),), ((37_205, 9_000_000),), 1)
                await asyncio.sleep(0.01)

    async def live_book(target):
        return INST, Endless(INST)

    async def maker_bps(target):
        return -2.0

    monkeypatch.setattr(cli, "_live_book", live_book)
    monkeypatch.setattr(cli, "_maker_bps_for", maker_bps)
    args = cli.build_parser().parse_args(["trade", "--slack"])
    run = asyncio.create_task(args.func(args))
    for _ in range(200):
        if posted:  # quoting has started
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.1)
    run.cancel()  # what SIGTERM becomes
    assert await run == 0
    assert "停止の指示" in posted[-1] and "注文はすべて取り消しました" in posted[-1]


@pytest.mark.asyncio
async def test_queued_cancels_go_out_as_one_request():
    v, api, health = venue(), DryRunApi(), Health()
    orders = [v.place(bid(37_000 - i, 10_000), 0, best_opposite=37_010) for i in range(3)]
    while await execute_once(v, api, "ada_jpy", health):
        pass
    sent_before = api.cancels_sent
    v.cancel_all()
    while await execute_once(v, api, "ada_jpy", health):
        pass
    assert all(o.state == DONE for o in orders) and api.open == set()
    assert api.cancels_sent - sent_before == 1


def _churn(tolerance: int) -> int:
    from jsboard.cli import build_maker, build_parser
    from jsboard.research.daily import FIXED_FLAGS

    args = build_parser().parse_args([
        "sweep", "-", "--source", "bitbank", *FIXED_FLAGS, "--maker-bps=-2",
        "--size", "100", "--max-position", "1000",
        "--price-tolerance-ticks", str(tolerance), "--requote-ms", "0",
    ])
    mm = build_maker(INST, args)
    mm.on_event(FeedStatus("live", "fake"))
    for i in range(40):
        mid = 37_200 + (i % 2)  # the book wobbles by one tick
        mm.on_event(DepthSnapshot(tuple((mid - 20 - j, 9_000_000) for j in range(5)),
                                  tuple((mid + 20 + j, 9_000_000) for j in range(5)), i,
                                  time.time_ns()))
        mm.requote(force=True)
    return mm.stats.orders_placed + mm.stats.orders_cancelled


def test_a_price_tolerance_stops_one_tick_wobbles_from_moving_orders():
    assert _churn(2) < _churn(0) / 3


def test_the_change_rate_is_measured_on_market_time_not_the_replay_banner():
    from jsboard.cli import build_maker, build_parser

    args = build_parser().parse_args([
        "sweep", "-", "--source", "bitbank", "--maker-bps=-2", "--size", "100",
        "--max-position", "1000",
    ])
    mm = build_maker(INST, args)
    mm.on_event(FeedStatus("connecting", "replay"))  # stamped now, days later
    day = 1_790_380_800_000_000_000
    for i in range(3):
        mm.on_event(DepthSnapshot(((37_190, 9_000_000),), ((37_210, 9_000_000),), i,
                                  day + i * 60_000_000_000))
    assert (mm.stats.last_event_ns - mm.stats.first_event_ns) / 1e9 == 120.0


def test_orders_cancelled_before_sending_never_wait_for_the_budget():
    from jsboard.live.runner import drop_noops

    v = venue()
    for price in (37_000, 36_990, 36_980):
        order = v.place(bid(price, 10_000), 0, best_opposite=37_010)
        v.cancel(order.order_id)  # the lead gate pulled the side at once
    kept = v.place(bid(36_970, 10_000), 0, best_opposite=37_010)
    drop_noops(v)
    assert list(v.intents) == [("new", kept.order_id)]


@pytest.mark.asyncio
async def test_a_fill_read_after_the_order_was_cancelled_is_still_booked():
    # The first live run bought 271 ADA and booked none of it: each fill was
    # read a second after its order had been cancelled and forgotten.
    v, api = venue(), DryRunApi()
    order = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", Health())
    v.cancel(order.order_id)
    await execute_once(v, api, "ada_jpy", Health())
    v.forget_done()
    assert order.order_id not in v.orders

    class Late(DryRunApi):
        async def trade_history(self, pair, since_ms):
            return [{"trade_id": 1, "order_id": order.exchange_id, "price": "37.000",
                     "amount": "100", "maker_taker": "maker", "executed_at": 1}]

    assert await poll_fills_once(v, Late(), "ada_jpy", {}) == 1
    assert v.base_balance == Decimal("300") and len(v.pending_fills) == 1


@pytest.mark.asyncio
async def test_an_account_that_drifts_from_the_records_stops_the_run():
    from jsboard.live.runner import check_balance_once

    class Account(DryRunApi):
        held = Decimal("100")

        async def onhand(self):
            return {"ada": self.held}

    v, api = venue(), Account()
    state = {"tolerance": Decimal("50")}
    lots = INST.to_lots("80")  # the run booked 80 ADA bought
    assert await check_balance_once(v, api, Decimal("20"), lots, state) == ""
    api.held = Decimal("271")  # bought far more than booked
    assert await check_balance_once(v, api, Decimal("20"), lots, state) == ""  # once may be late
    assert "記録と合いません" in await check_balance_once(v, api, Decimal("20"), lots, state)


@pytest.mark.asyncio
async def test_a_cancel_the_batch_skipped_is_retried_not_forgotten():
    # bitbank's cancel_orders succeeds while skipping an order that has not
    # reached the book; treating the batch as done left such orders resting.
    class Skips(DryRunApi):
        async def cancel_many(self, pair, order_ids):
            self.cancels_sent += 1
            done = set(order_ids[:-1])  # the newest one was not on the book yet
            self.open.difference_update(done)
            return done

    v, api, health = venue(), Skips(), Health()
    orders = [v.place(bid(37_000 - i, 10_000), 0, best_opposite=37_010) for i in range(3)]
    while await execute_once(v, api, "ada_jpy", health):
        pass
    v.cancel_all()
    while await execute_once(v, api, "ada_jpy", health):
        pass
    assert all(o.state == DONE for o in orders) and api.open == set()


@pytest.mark.asyncio
async def test_orders_the_run_lost_track_of_are_swept():
    from jsboard.live.runner import sweep_strays

    v, api = venue(), DryRunApi()
    kept = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", Health())
    api.open.add(999)  # resting on the venue, unknown to the run
    assert await sweep_strays(v, api, "ada_jpy") == [999]
    assert api.open == {kept.exchange_id}


@pytest.mark.asyncio
async def test_a_cancel_refused_while_the_venue_is_busy_is_retried_quietly():
    class Busy(DryRunApi):
        def __init__(self):
            super().__init__()
            self.refusals = 3

        async def cancel(self, pair, order_id):
            if self.refusals:
                self.refusals -= 1
                raise BitbankError(50010, "cancel")
            await super().cancel(pair, order_id)

    v, api, health = venue(), Busy(), Health(max_errors=2)
    order = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", health)
    order.cancel_tries = 1  # skip the batch path
    v.cancel(order.order_id)
    while await execute_once(v, api, "ada_jpy", health):
        pass
    assert order.state == DONE and api.open == set() and not health.fatal


@pytest.mark.asyncio
async def test_fills_on_orders_being_pulled_are_counted():
    v, api = venue(), DryRunApi()
    order = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", Health())
    v.cancel(order.order_id)
    after = int((order.cancel_asked + 0.1) * 1e9)

    class Late(DryRunApi):
        async def trade_history(self, pair, since_ms):
            return [{"trade_id": 5, "order_id": order.exchange_id, "price": "37.000",
                     "amount": "10", "maker_taker": "maker", "executed_at": after // 1_000_000}]

    await poll_fills_once(v, Late(), "ada_jpy", {})
    await execute_once(v, api, "ada_jpy", Health())
    assert v.doomed_fills == 1 and v.fills_seen == 1 and len(v.cancel_ms) == 1
    assert "取消し中の約定 1/1回（100%）" in v.cancel_report()


@pytest.mark.asyncio
async def test_our_own_resting_orders_are_taken_out_of_the_public_book():
    from jsboard.feed.base import DepthDelta

    v, api = venue(), DryRunApi()
    order = v.place(bid(37_000, 1_000_000), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", Health())
    # The venue's book: our 100 ADA is the whole best bid at 37.000.
    snap = DepthSnapshot(((37_000, 1_000_000), (36_990, 5_000_000)),
                         ((37_010, 2_000_000),), 1)
    seen = v.strip_own(snap)
    assert seen.bids == ((36_990, 5_000_000),)  # the touch is somebody else's again
    delta = DepthDelta(((37_000, 1_500_000),), (), 2, 2)  # someone joined behind us
    assert v.strip_own(delta).bids == ((37_000, 500_000),)
    v.cancel(order.order_id)
    await execute_once(v, api, "ada_jpy", Health())
    assert v.strip_own(snap) is snap  # nothing of ours left to take out


def test_without_the_correction_the_unwind_would_chase_its_own_order():
    from jsboard.cli import build_maker, build_parser
    from jsboard.research.daily import FIXED_FLAGS

    args = build_parser().parse_args([
        "sweep", "-", "--source", "bitbank", *FIXED_FLAGS, "--maker-bps=-2",
        "--size", "100", "--max-position", "1000", "--requote-ms", "0",
    ])
    mm = build_maker(INST, args)
    v = venue()
    mm.venue = v
    mm.on_event(FeedStatus("live", "fake"))
    now = time.time_ns()
    book = DepthSnapshot(((36_990, 9_000_000),), ((37_020, 9_000_000),), 1, now)
    mm.on_event(v.strip_own(book))
    first = mm.quoter.quote(fair_value=mm.fair_value.estimate(mm.market), sigma_ticks=0,
                            inventory_lots=0, best_bid=36_990, best_ask=37_020)
    assert first.bids  # sanity: the maker has a bid to place
    # Rest a bid one tick inside, as an unwind would, then show the venue's
    # book with it at the touch: stripped, the touch is still 36.990.
    order = v.place(Quote(Side.BUY, 36_991, 1_000_000), 0, best_opposite=37_020)
    order.exchange_id, order.state = 1, OPEN
    mm.on_event(v.strip_own(DepthSnapshot(((36_991, 1_000_000), (36_990, 9_000_000)),
                                          ((37_020, 9_000_000),), 2, now)))
    assert mm.market.book.best_bid() == 36_990


def test_a_cancel_waiting_to_retry_does_not_hold_up_the_queue():
    from jsboard.live.runner import head_due

    v = venue()
    slow = v.place(bid(37_000, 10_000), 0, best_opposite=37_010)
    fast = v.place(bid(36_990, 10_000), 0, best_opposite=37_010)
    for o, i in ((slow, 1), (fast, 2)):
        o.exchange_id, o.state = i, OPEN
    v.intents.clear()
    v.cancel(slow.order_id)
    v.cancel(fast.order_id)
    slow.retry_at = time.monotonic() + 60
    assert head_due(v) and v.intents[0] == ("cancel", fast.order_id)
    v.intents.popleft()
    assert not head_due(v)  # only the waiting one is left


@pytest.mark.asyncio
async def test_an_order_the_venue_cannot_find_yet_stays_tracked_until_it_appears():
    # bitbank answered 50009 for many seconds about orders that then rested;
    # dropping them left them untracked, unstripped and holding balance.
    class Slow(DryRunApi):
        hidden = True

        async def cancel(self, pair, order_id):
            if self.hidden:
                raise BitbankError(NOT_FOUND, "cancel")
            await super().cancel(pair, order_id)

        async def status(self, pair, order_id):
            if self.hidden:
                raise BitbankError(NOT_FOUND, "status")
            return await super().status(pair, order_id)

    v, api, health = venue(), Slow(), Health()
    order = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, api, "ada_jpy", health)
    order.cancel_tries = 39  # past the quick retries
    v.cancel(order.order_id)
    await execute_once(v, api, "ada_jpy", health)
    assert order.state == CANCELLING and v.intents  # still ours, still being cancelled
    assert not v.can_afford(Side.BUY, 37_000, INST.to_lots("200"))  # its yen stays reserved
    api.hidden = False
    order.retry_at = 0
    while await execute_once(v, api, "ada_jpy", health):
        pass
    assert order.state == DONE and api.open == set() and not health.fatal


def test_a_gap_fill_share_fills_only_part_of_the_gap_throughs():
    from jsboard.sim.paper import PaperConfig, PaperVenue

    def fills(share):
        v = PaperVenue(INST, config=PaperConfig(latency_ms=0, gap_fill_share=share),
                       clock=lambda: 1)
        for _ in range(200):
            v.place(Quote(Side.BUY, 37_000, 10_000), 0, best_opposite=37_010)
        # The book moves through every bid at once, twice.
        first = len(v.on_book(36_990, 37_000, ts_ns=1))
        again = len(v.on_book(36_990, 37_000, ts_ns=2))
        return first, again, v.gap_passed

    assert fills(1.0)[:2] == (200, 0)
    first, again, passed = fills(0.15)
    assert 15 <= first <= 50 and again == 0  # a pass is not re-rolled
    assert passed == 2 * (200 - first)


@pytest.mark.asyncio
async def test_a_refused_post_only_order_is_dropped_without_counting_as_a_failure():
    from jsboard.live.bitbank import OrderRefused

    class Refuses(DryRunApi):
        async def order(self, pair, side, price, amount):
            raise OrderRefused(5, "REJECTED")

    v, health = venue(), Health(max_errors=1)
    order = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, Refuses(), "ada_jpy", health)
    assert order.state == DONE and v.post_only_refused == 1 and not health.fatal
    assert v.strip_own(DepthSnapshot(((37_000, 1_000_000),), ((37_010, 1),), 1)).bids


@pytest.mark.asyncio
async def test_a_balance_refusal_drops_the_order_without_stopping_the_run():
    class Short(DryRunApi):
        async def order(self, pair, side, price, amount):
            raise BitbankError(60001, "order")

    v, health = venue(), Health(max_errors=1)
    order = v.place(bid(), 0, best_opposite=37_010)
    await execute_once(v, Short(), "ada_jpy", health)
    assert order.state == DONE and v.insufficient == 1 and not health.fatal
