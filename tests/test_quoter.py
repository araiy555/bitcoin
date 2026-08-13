"""Quoter: skew direction, spread floors, sizing, and the no-cross invariant."""

import pytest

from jsboard.mm.quoter import Quoter, QuoterConfig


def make_quoter(**overrides):
    cfg = QuoterConfig(
        gamma=0.6,
        kappa=1.4,
        levels=3,
        level_step_ticks=2,
        base_size_lots=100,
        max_position_lots=1000,
        min_half_spread_ticks=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return Quoter(cfg)


def quote(q, *, inventory=0, sigma=2.0, bid=1000, ask=1010, fair=None):
    return q.quote(
        fair_value=fair if fair is not None else (bid + ask) / 2.0,
        sigma_ticks=sigma,
        inventory_lots=inventory,
        best_bid=bid,
        best_ask=ask,
    )


class TestSkew:
    def test_flat_inventory_quotes_symmetrically(self):
        qs = quote(make_quoter())

        assert qs.reservation == pytest.approx(1005.0)

    def test_long_inventory_pushes_quotes_down(self):
        flat = quote(make_quoter(), inventory=0)
        long = quote(make_quoter(), inventory=800)

        assert long.reservation < flat.reservation
        assert long.bids[0].price <= flat.bids[0].price
        assert long.asks[0].price <= flat.asks[0].price

    def test_short_inventory_pushes_quotes_up(self):
        flat = quote(make_quoter(), inventory=0)
        short = quote(make_quoter(), inventory=-800)

        assert short.reservation > flat.reservation

    def test_skew_scales_with_volatility(self):
        calm = make_quoter().reservation_price(1000.0, 500, sigma_ticks=1.0)
        wild = make_quoter().reservation_price(1000.0, 500, sigma_ticks=20.0)

        assert (1000.0 - wild) > (1000.0 - calm)

    def test_skew_is_capped(self):
        q = make_quoter(max_skew_ticks=5)
        r = q.reservation_price(1000.0, 1000, sigma_ticks=500.0)

        assert r == pytest.approx(995.0)


class TestHalfSpread:
    def test_widens_with_volatility(self):
        q = make_quoter()

        assert q.half_spread(50.0) > q.half_spread(1.0)

    def test_thin_book_widens_the_quote(self):
        thick = make_quoter(kappa=10.0).half_spread(5.0)
        thin = make_quoter(kappa=0.5).half_spread(5.0)

        assert thin > thick

    def test_tick_floor_binds(self):
        q = make_quoter(min_half_spread_ticks=8, vol_multiplier=0.0, liquidity_premium_ticks=0.0)

        assert q.half_spread(0.0) == 8

    def test_cap_binds(self):
        q = make_quoter(max_half_spread_ticks=12)

        assert q.half_spread(10_000.0) == 12

    def test_fee_floor_overrides_the_cap(self):
        # A 2bp floor on a 1,000,000-tick mid is 200 ticks; a maker must not
        # quote inside its own fee just because the cap says so.
        q = make_quoter(min_edge_bps=2.0, max_half_spread_ticks=10)

        assert q.half_spread(1.0, mid_ticks=1_000_000) == pytest.approx(200.0)

    def test_fee_floor_ignored_without_a_mid(self):
        q = make_quoter(min_edge_bps=2.0)

        assert q.half_spread(1.0) < 10


class TestNoCross:
    def test_quotes_never_cross_the_touch(self):
        qs = quote(make_quoter(), bid=1000, ask=1001)

        for b in qs.bids:
            assert b.price < 1001
        for a in qs.asks:
            assert a.price > 1000

    def test_wide_model_spread_stays_outside(self):
        qs = quote(make_quoter(min_half_spread_ticks=50), bid=1000, ask=1010)

        assert qs.bids[0].price <= 955
        assert qs.asks[0].price >= 1055

    def test_tight_model_spread_improves_the_touch(self):
        qs = quote(
            make_quoter(min_half_spread_ticks=1, vol_multiplier=0.0, liquidity_premium_ticks=0.0),
            bid=1000,
            ask=1010,
        )

        # Fair is 1005, half-spread 1 -> quote inside the 10-tick market spread.
        assert qs.bids[0].price > 1000
        assert qs.asks[0].price < 1010

    def test_improvement_can_be_disabled(self):
        qs = quote(
            make_quoter(
                allow_price_improvement=False,
                min_half_spread_ticks=1,
                vol_multiplier=0.0,
                liquidity_premium_ticks=0.0,
            ),
            bid=1000,
            ask=1010,
        )

        assert qs.bids[0].price == 1000
        assert qs.asks[0].price == 1010


class TestSizing:
    def test_size_decays_along_the_ladder(self):
        qs = quote(make_quoter(level_size_decay=0.5))
        sizes = [b.qty for b in qs.bids]

        assert sizes == sorted(sizes, reverse=True)
        assert len(set(sizes)) > 1

    def test_long_inventory_shrinks_the_bid_and_not_the_ask(self):
        flat = quote(make_quoter(), inventory=0)
        long = quote(make_quoter(), inventory=500)

        assert sum(b.qty for b in long.bids) < sum(b.qty for b in flat.bids)
        assert sum(a.qty for a in long.asks) == sum(a.qty for a in flat.asks)

    def test_full_position_stops_quoting_that_side(self):
        qs = quote(make_quoter(), inventory=1000)

        assert qs.bids == ()
        assert qs.asks != ()

    def test_full_short_stops_the_ask(self):
        qs = quote(make_quoter(), inventory=-1000)

        assert qs.asks == ()
        assert qs.bids != ()

    def test_total_size_respects_remaining_capacity(self):
        qs = quote(make_quoter(base_size_lots=1000), inventory=900)

        assert sum(b.qty for b in qs.bids) <= 100


class TestDegenerate:
    def test_one_sided_book_produces_no_quotes(self):
        q = make_quoter()
        qs = q.quote(
            fair_value=1000.0, sigma_ticks=1.0, inventory_lots=0, best_bid=None, best_ask=1010
        )

        assert qs.is_empty
        assert "two-sided" in qs.reason

    def test_missing_fair_value_produces_no_quotes(self):
        q = make_quoter()
        qs = q.quote(
            fair_value=None, sigma_ticks=1.0, inventory_lots=0, best_bid=1000, best_ask=1010
        )

        assert qs.is_empty

    def test_levels_landing_on_one_tick_are_merged(self):
        # A one-tick market leaves no room for a ladder; all levels collapse.
        qs = quote(make_quoter(levels=3, level_step_ticks=0), bid=1000, ask=1001)

        assert len({b.price for b in qs.bids}) == len(qs.bids)
        assert sum(b.qty for b in qs.bids) > 0

    def test_quote_keys_are_unique(self):
        qs = quote(make_quoter())
        keys = [q.key() for q in qs.all()]

        assert len(keys) == len(set(keys))


class TestMaxDistance:
    """The cap that decides whether we ever join the queue at all."""

    def cfg(self, **kw):
        base = dict(levels=3, level_step_ticks=2, base_size_lots=10, max_position_lots=100)
        base.update(kw)
        return QuoterConfig(**base)

    def quote(self, **kw):
        q = Quoter(self.cfg(**kw))
        return q.quote(
            fair_value=1000.5,
            sigma_ticks=0.5,
            inventory_lots=0,
            best_bid=1000,
            best_ask=1001,
        )

    def test_uncapped_quotes_rest_behind_the_touch(self):
        qs = self.quote(max_distance_ticks=None)
        assert max(q.price for q in qs.bids) < 1000
        assert min(q.price for q in qs.asks) > 1001

    def test_zero_clamps_every_level_onto_the_touch(self):
        qs = self.quote(max_distance_ticks=0)
        assert {q.price for q in qs.bids} == {1000}
        assert {q.price for q in qs.asks} == {1001}

    def test_the_collapsed_ladder_keeps_its_total_size(self):
        loose = self.quote(max_distance_ticks=None)
        tight = self.quote(max_distance_ticks=0)
        assert sum(q.qty for q in tight.bids) == sum(q.qty for q in loose.bids)

    def test_a_wider_cap_allows_stepping_back_that_far(self):
        qs = self.quote(max_distance_ticks=2)
        assert min(q.price for q in qs.bids) >= 998
        assert max(q.price for q in qs.asks) <= 1003

    def test_the_cap_never_makes_a_quote_cross(self):
        qs = self.quote(max_distance_ticks=0, min_half_spread_ticks=0)
        assert all(q.price < 1001 for q in qs.bids)
        assert all(q.price > 1000 for q in qs.asks)

    def test_the_cap_does_not_widen_a_quote_that_is_already_inside(self):
        # Price improvement puts us inside the touch; the cap only pulls in.
        qs = self.quote(max_distance_ticks=0, min_half_spread_ticks=0, liquidity_premium_ticks=0.0)
        assert max(q.price for q in qs.bids) >= 1000
