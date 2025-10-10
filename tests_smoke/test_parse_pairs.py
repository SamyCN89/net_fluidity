import pytest
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct execution
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_speed_bootstrap import _parse_pairs


def test_parse_pairs_arity_two():
    pairs = "(A,B)-(C,D);(E,F)-(G,H)"
    out = _parse_pairs(pairs, arity=2)
    assert out == [(('A', 'B'), ('C', 'D')), (('E', 'F'), ('G', 'H'))]


def test_parse_pairs_arity_three():
    pairs = "(A,B,C)-(D,E,F)"
    out = _parse_pairs(pairs, arity=3)
    assert out == [(('A', 'B', 'C'), ('D', 'E', 'F'))]


def test_parse_pairs_mismatch_raises():
    pairs = "(A,B)-(C)"
    with pytest.raises(ValueError):
        _parse_pairs(pairs, arity=2)
