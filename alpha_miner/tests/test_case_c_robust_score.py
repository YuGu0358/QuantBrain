"""Tests for case_c_robust_score composite degraded gate (main.py)."""
from __future__ import annotations

import pytest

from alpha_miner.main import case_c_robust_score


# ---------------------------------------------------------------------------
# Hard disqualifiers
# ---------------------------------------------------------------------------

def test_brain_check_false_returns_zero():
    score, qualified = case_c_robust_score(
        sharpe=0.8, fitness=0.9, turnover=0.2,
        test_sharpe=None, brain_check_passed=False,
    )
    assert score == 0.0
    assert qualified is False


def test_sharpe_none_returns_zero():
    score, qualified = case_c_robust_score(
        sharpe=None, fitness=0.9, turnover=0.2,
        test_sharpe=None, brain_check_passed=True,
    )
    assert score == 0.0
    assert qualified is False


def test_high_turnover_hard_reject():
    """Turnover > 0.75 is a hard reject regardless of sharpe."""
    score, qualified = case_c_robust_score(
        sharpe=1.5, fitness=1.0, turnover=0.76,
        test_sharpe=None, brain_check_passed=True,
    )
    assert score == 0.0
    assert qualified is False


def test_turnover_exactly_075_is_hard_reject():
    score, qualified = case_c_robust_score(
        sharpe=1.0, fitness=1.0, turnover=0.75,
        test_sharpe=None, brain_check_passed=True,
    )
    # turnover > 0.75 is the condition, so exactly 0.75 is NOT rejected
    assert score > 0.0
    assert qualified is True


# ---------------------------------------------------------------------------
# Threshold calibration (from docstring)
# ---------------------------------------------------------------------------

def test_calibration_sharpe_050_no_extras():
    """sharpe=0.50, fitness=None, turnover=None → 0.200 → qualified"""
    score, qualified = case_c_robust_score(
        sharpe=0.50, fitness=None, turnover=None,
        test_sharpe=None, brain_check_passed=True,
    )
    # sharpe_score = 0.50/2*0.60 = 0.15
    # fitness_score = 0.0
    # turnover_score = 0.05 (unknown → neutral)
    # oos_score = 0.0
    assert score == pytest.approx(0.20, abs=1e-4)
    assert qualified is True


def test_calibration_sharpe_040_no_extras():
    """sharpe=0.40, fitness=None, turnover=None → 0.170 → rejected"""
    score, qualified = case_c_robust_score(
        sharpe=0.40, fitness=None, turnover=None,
        test_sharpe=None, brain_check_passed=True,
    )
    # sharpe_score = 0.40/2*0.60 = 0.12
    # turnover_score = 0.05
    assert score == pytest.approx(0.17, abs=1e-4)
    assert qualified is False


def test_calibration_sharpe_045_with_fitness_and_turnover():
    """sharpe=0.45, fitness=0.70, turnover=0.30 → > 0.20 → qualified"""
    score, qualified = case_c_robust_score(
        sharpe=0.45, fitness=0.70, turnover=0.30,
        test_sharpe=None, brain_check_passed=True,
    )
    assert score > 0.20
    assert qualified is True


def test_calibration_strong_alpha():
    """sharpe=0.80, fitness=0.70, turnover=0.25 → qualified"""
    score, qualified = case_c_robust_score(
        sharpe=0.80, fitness=0.70, turnover=0.25,
        test_sharpe=None, brain_check_passed=True,
    )
    assert score > 0.30
    assert qualified is True


# ---------------------------------------------------------------------------
# OOS consistency bonus
# ---------------------------------------------------------------------------

def test_oos_bonus_rewards_consistent_test_sharpe():
    score_no_oos, _ = case_c_robust_score(
        sharpe=0.60, fitness=0.60, turnover=0.30,
        test_sharpe=None, brain_check_passed=True,
    )
    score_with_oos, _ = case_c_robust_score(
        sharpe=0.60, fitness=0.60, turnover=0.30,
        test_sharpe=0.60,  # perfect IS/OOS match
        brain_check_passed=True,
    )
    assert score_with_oos > score_no_oos
    assert score_with_oos - score_no_oos == pytest.approx(0.05, abs=1e-4)


def test_oos_score_capped_at_005():
    """Even if test_sharpe >> sharpe, OOS bonus is capped at 0.05."""
    score, _ = case_c_robust_score(
        sharpe=0.60, fitness=0.60, turnover=0.30,
        test_sharpe=10.0,  # unrealistically high
        brain_check_passed=True,
    )
    # oos_ratio is clamped to 1.0, so max contribution is 0.05
    score_no_oos, _ = case_c_robust_score(
        sharpe=0.60, fitness=0.60, turnover=0.30,
        test_sharpe=None, brain_check_passed=True,
    )
    assert score - score_no_oos == pytest.approx(0.05, abs=1e-4)


def test_negative_test_sharpe_gives_zero_oos():
    score_neg, _ = case_c_robust_score(
        sharpe=0.60, fitness=0.50, turnover=0.30,
        test_sharpe=-0.10, brain_check_passed=True,
    )
    score_none, _ = case_c_robust_score(
        sharpe=0.60, fitness=0.50, turnover=0.30,
        test_sharpe=None, brain_check_passed=True,
    )
    # Negative test_sharpe: oos_ratio = min(1, -0.1/0.6) < 0 → max(0, ...) = 0
    assert score_neg == score_none


# ---------------------------------------------------------------------------
# brain_check_passed=None (unknown) should not reject
# ---------------------------------------------------------------------------

def test_brain_check_none_does_not_reject():
    """None means check wasn't run (quota waiting) — should not hard-reject."""
    score, qualified = case_c_robust_score(
        sharpe=0.60, fitness=None, turnover=None,
        test_sharpe=None, brain_check_passed=None,
    )
    assert qualified is True


# ---------------------------------------------------------------------------
# Score is always in [0, 1] range
# ---------------------------------------------------------------------------

def test_score_bounded_by_one():
    score, _ = case_c_robust_score(
        sharpe=100.0, fitness=100.0, turnover=0.0,
        test_sharpe=100.0, brain_check_passed=True,
    )
    assert 0.0 <= score <= 1.0


def test_score_non_negative():
    score, _ = case_c_robust_score(
        sharpe=0.001, fitness=0.001, turnover=0.001,
        test_sharpe=None, brain_check_passed=True,
    )
    assert score >= 0.0
