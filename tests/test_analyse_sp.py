"""Tests for the Part I stability rules."""

import math

import numpy as np

from analyse_sp import stabilisation_budget

B = [10**4, 10**5, 10**6, 10**7, 10**8, 10**9]


def test_stable_throughout():
    assert stabilisation_budget(np.array([10, 12, 11, 13, 12, 12]), B) == 10**4


def test_stabilises_part_way():
    # steps: 20, 8, 3, 2, 1 -> stable from 10^6 onwards
    assert stabilisation_budget(np.array([40, 20, 12, 9, 11, 12]), B) == 10**6


def test_late_jump_resets():
    # steps: 1, 1, 9, 1, 1 -> the 10^6 -> 10^7 step breaks it; stable from 10^7
    assert stabilisation_budget(np.array([10, 11, 12, 21, 22, 23]), B) == 10**7


def test_not_stabilised_when_final_step_large():
    assert math.isnan(stabilisation_budget(np.array([10, 11, 12, 13, 14, 25]), B))


def test_boundary_counts_as_stable():
    assert stabilisation_budget(np.array([0, 5, 10, 15, 20, 25]), B) == 10**4
