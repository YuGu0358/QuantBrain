"""Tests for FactorFactory skeleton enumeration (m_factor_factory.py)."""
from __future__ import annotations

import pytest

from alpha_miner.modules.m_factor_factory import (
    Skeleton,
    enumerate_skeleton,
    enumerate_skeletons,
    get_skeletons,
)
from alpha_miner.modules.m3_validator import ExpressionValidator


# ---------------------------------------------------------------------------
# get_skeletons
# ---------------------------------------------------------------------------

def test_get_skeletons_returns_all_when_no_filter():
    all_sk = get_skeletons()
    assert len(all_sk) >= 10  # at least the 12 defined skeletons


def test_get_skeletons_filters_by_category():
    momentum = get_skeletons("MOMENTUM")
    assert all(s.category == "MOMENTUM" for s in momentum)
    assert len(momentum) >= 2


def test_get_skeletons_case_insensitive():
    assert get_skeletons("momentum") == get_skeletons("MOMENTUM")


def test_get_skeletons_unknown_category_returns_empty():
    assert get_skeletons("NONEXISTENT_CAT") == []


# ---------------------------------------------------------------------------
# enumerate_skeleton — single skeleton expansion
# ---------------------------------------------------------------------------

def test_enumerate_skeleton_produces_valid_dicts():
    sk = get_skeletons("MOMENTUM")[0]
    candidates = enumerate_skeleton(sk, max_per_skeleton=2)
    assert len(candidates) <= 2
    for c in candidates:
        assert "expression" in c
        assert "hypothesis" in c
        assert "category" in c
        assert "origin_refs" in c
        assert c["category"] == "MOMENTUM"
        assert "factor_factory" in c["origin_refs"]
        assert sk.name in c["origin_refs"]


def test_enumerate_skeleton_expression_fills_placeholders():
    """No {placeholder} should survive in the final expression."""
    for sk in get_skeletons():
        candidates = enumerate_skeleton(sk, max_per_skeleton=2)
        for c in candidates:
            assert "{" not in c["expression"], (
                f"Unfilled placeholder in {sk.name}: {c['expression']}"
            )


def test_enumerate_skeleton_respects_max_per_skeleton():
    sk = Skeleton(
        name="test_sk",
        category="MOMENTUM",
        template="rank(ts_mean({field}, {window}))",
        hypothesis="Test skeleton with {field} and {window}.",
        params={"field": ["close", "returns", "vwap"], "window": [21, 42, 63, 126]},
    )
    candidates = enumerate_skeleton(sk, max_per_skeleton=3)
    assert len(candidates) <= 3


def test_enumerate_skeleton_returns_all_when_small_space():
    """When combos <= max_per_skeleton, all combos are returned."""
    sk = Skeleton(
        name="small_sk",
        category="VALUE",
        template="rank({field})",
        hypothesis="Rank {field}.",
        params={"field": ["close"]},
    )
    candidates = enumerate_skeleton(sk, max_per_skeleton=10)
    assert len(candidates) == 1
    assert candidates[0]["expression"] == "rank(close)"


# ---------------------------------------------------------------------------
# enumerate_skeletons — multi-skeleton top-level API
# ---------------------------------------------------------------------------

def test_enumerate_skeletons_returns_n_or_fewer():
    candidates = enumerate_skeletons(n=5, seed=42)
    assert len(candidates) <= 5


def test_enumerate_skeletons_category_filter_respected():
    candidates = enumerate_skeletons(category="VALUE", n=20, seed=0)
    assert all(c["category"] == "VALUE" for c in candidates)


def test_enumerate_skeletons_reproducible_with_seed():
    a = enumerate_skeletons(category="MOMENTUM", n=8, seed=7)
    b = enumerate_skeletons(category="MOMENTUM", n=8, seed=7)
    assert [c["expression"] for c in a] == [c["expression"] for c in b]


def test_enumerate_skeletons_different_seeds_produce_different_results():
    a = enumerate_skeletons(n=20, seed=1)
    b = enumerate_skeletons(n=20, seed=999)
    # With enough candidates it's very unlikely both seeds produce identical order
    exprs_a = [c["expression"] for c in a]
    exprs_b = [c["expression"] for c in b]
    assert exprs_a != exprs_b


def test_enumerate_skeletons_all_have_required_fields():
    candidates = enumerate_skeletons(n=30, seed=42)
    for c in candidates:
        assert c.get("expression"), "expression must be non-empty"
        assert c.get("hypothesis"), "hypothesis must be non-empty"
        assert c.get("category"), "category must be set"
        assert isinstance(c.get("origin_refs"), list)
        assert c["origin_refs"][0] == "factor_factory"
        assert c.get("opt_rounds") == 0


def test_enumerate_skeletons_no_unfilled_placeholders():
    candidates = enumerate_skeletons(n=50, seed=0)
    for c in candidates:
        assert "{" not in c["expression"], (
            f"Unfilled placeholder: {c['expression']}"
        )


def test_enumerate_skeletons_returns_empty_for_unknown_category():
    candidates = enumerate_skeletons(category="DOES_NOT_EXIST", n=10)
    assert candidates == []


def test_enumerate_skeletons_ids_are_unique():
    candidates = enumerate_skeletons(n=40, seed=42)
    ids = [c["id"] for c in candidates]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_enumerate_skeletons_validates_for_default_profile_representative_categories():
    validator = ExpressionValidator()
    for category in ["MOMENTUM", "VALUE", "QUALITY", "REVERSAL", "GROWTH"]:
        candidates = enumerate_skeletons(category=category, n=12, seed=17)
        assert candidates, f"{category} should produce candidates"
        for candidate in candidates:
            result = validator.validate(candidate["expression"])
            assert result.is_valid, (
                f"{category} candidate failed validation: {candidate['expression']} "
                f"errors={result.errors}"
            )


def test_no_factor_factory_skeleton_references_unverified_fields_or_operators():
    validator = ExpressionValidator(max_complexity=999)
    for skeleton in get_skeletons():
        candidates = enumerate_skeleton(skeleton, max_per_skeleton=10_000)
        assert candidates, f"{skeleton.name} should enumerate at least one candidate"
        for candidate in candidates:
            result = validator.validate(candidate["expression"])
            assert result.is_valid, (
                f"{skeleton.name} references unverified symbols in "
                f"{candidate['expression']}: {result.errors}"
            )
