"""Deterministic answer parsing and exact-equivalence scoring for PHYBench."""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import sympy
from sympy.core.relational import Equality


VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "phybench_eed"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from latex_pre_process import master_convert  # noqa: E402
from extended_zss import ext_distance  # noqa: E402


@dataclass(frozen=True)
class Score:
    correct: int
    eed_fitness: float
    reason: str
    canonical_candidate: str
    canonical_gold: str
    candidate_parse_ok: bool
    gold_parse_ok: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_latex(value: Any) -> str:
    """Extract a compact mathematical answer without changing its semantics."""
    if value is None:
        return ""
    text = str(value).strip()
    # Nova tool arguments occasionally contain JSON control characters where
    # the intended LaTeX command used a single backslash (for example, form
    # feed + "rac" for ``\\frac``). Recover those commands deterministically.
    text = (
        text.replace("\f", r"\f")
        .replace("\t", r"\t")
        .replace("\r", r"\r")
        .replace("\b", r"\b")
        .replace("\nabla", r"\nabla")
        .replace("\nu", r"\nu")
    )
    text = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", text, flags=re.I)
    fence = chr(96) * 3
    text = text.replace(fence + "latex", "").replace(fence + "json", "").replace(fence, "")
    text = text.strip()
    for left, right in ((r"\[", r"\]"), ("$$", "$$"), (r"\(", r"\)"), ("$", "$")):
        if text.startswith(left) and text.endswith(right) and len(text) >= len(left) + len(right):
            text = text[len(left) : len(text) - len(right)].strip()
    return text.strip().rstrip(".")


@lru_cache(maxsize=4096)
def parse_latex(value: str) -> sympy.Basic:
    cleaned = clean_latex(value)
    if not cleaned:
        raise ValueError("empty answer")
    return master_convert(cleaned)


def _safe_simplify(expr: sympy.Basic) -> sympy.Basic:
    try:
        return sympy.simplify(expr)
    except Exception:
        return expr


def _canonical(expr: sympy.Basic) -> str:
    if isinstance(expr, Equality):
        lhs = _safe_simplify(expr.lhs)
        rhs = _safe_simplify(expr.rhs)
        return f"Eq({sympy.srepr(lhs)},{sympy.srepr(rhs)})"
    return sympy.srepr(_safe_simplify(expr))


def _expression_equal(left: sympy.Basic, right: sympy.Basic) -> bool:
    if left == right:
        return True
    try:
        difference = _safe_simplify(left - right)
        if difference == 0 or getattr(difference, "is_zero", None) is True:
            return True
    except Exception:
        pass
    try:
        equals = left.equals(right)
        if equals is True:
            return True
    except Exception:
        pass
    if not getattr(left, "free_symbols", set()) and not getattr(right, "free_symbols", set()):
        try:
            lv = complex(sympy.N(left, 15))
            rv = complex(sympy.N(right, 15))
            scale = max(1.0, abs(lv), abs(rv))
            return abs(lv - rv) <= 1e-9 * scale
        except Exception:
            pass
    return False


def _equivalent(candidate: sympy.Basic, gold: sympy.Basic) -> tuple[bool, str]:
    if isinstance(candidate, Equality) and isinstance(gold, Equality):
        if _expression_equal(candidate.lhs, gold.lhs) and _expression_equal(candidate.rhs, gold.rhs):
            return True, "matching_equation"
        if _expression_equal(candidate.lhs, gold.rhs) and _expression_equal(candidate.rhs, gold.lhs):
            return True, "reversed_equation"
        try:
            c_residual = _safe_simplify(candidate.lhs - candidate.rhs)
            g_residual = _safe_simplify(gold.lhs - gold.rhs)
            ratio = _safe_simplify(c_residual / g_residual)
            if ratio != 0 and not getattr(ratio, "free_symbols", set()):
                return True, "equivalent_relation"
        except Exception:
            pass
        return False, "different_equation"

    if isinstance(gold, Equality):
        if _expression_equal(candidate, gold.rhs):
            return True, "matches_gold_rhs"
        return False, "candidate_not_gold_equation"

    if isinstance(candidate, Equality):
        if _expression_equal(candidate.rhs, gold):
            return True, "candidate_rhs_matches_gold"
        return False, "gold_not_candidate_equation"

    if _expression_equal(candidate, gold):
        return True, "symbolic_equivalence"
    return False, "not_equivalent"


class _TreeNode:
    def __init__(self, label: str, children: list["_TreeNode"] | None = None):
        self.label = label
        self.children = children or []
        self.subtree_size = 0.0

    def get_children(self) -> list["_TreeNode"]:
        return self.children


def _distance_expression(expr: sympy.Basic) -> sympy.Basic:
    if isinstance(expr, Equality):
        return _safe_simplify(expr.rhs)
    return _safe_simplify(expr)


def _sympy_to_tree(expr: sympy.Basic) -> _TreeNode:
    number_types = (
        sympy.Integer,
        sympy.Rational,
        sympy.Float,
        sympy.core.numbers.Pi,
        sympy.core.numbers.Exp1,
        sympy.core.numbers.Infinity,
        sympy.core.numbers.NegativeInfinity,
        sympy.core.numbers.ImaginaryUnit,
    )
    if isinstance(expr, number_types):
        return _TreeNode(f"number_{expr}")
    if isinstance(expr, sympy.Symbol):
        return _TreeNode(f"symbol_{expr}")
    if isinstance(expr, (sympy.Add, sympy.Mul, sympy.Pow)):
        return _TreeNode(
            f"operator_{type(expr).__name__}",
            [_sympy_to_tree(argument) for argument in expr.args],
        )
    if isinstance(expr, sympy.Function):
        return _TreeNode(
            f"function_{expr.func.__name__}",
            [_sympy_to_tree(argument) for argument in expr.args],
        )
    raise ValueError(f"unsupported SymPy type: {type(expr).__name__}")


def _node_cost(node: _TreeNode) -> float:
    del node
    return 1.0


def _tree_size(node: _TreeNode) -> float:
    if node.children and node.subtree_size:
        return node.subtree_size
    total = 1.0 + sum(_tree_size(child) for child in node.children)
    node.subtree_size = total
    return total


def _tree_cost(node: _TreeNode) -> float:
    size = _tree_size(node)
    return min(size, 0.6 * (size - 5.0) + 5.0)


def _update_cost(left: _TreeNode, right: _TreeNode) -> float:
    if left.label == right.label:
        return 0.0
    return 1.0


def _eed_similarity(candidate: sympy.Basic, gold: sympy.Basic) -> float:
    """Return PHYBench's EED score normalized to [0, 1]."""
    try:
        candidate_tree = _sympy_to_tree(_distance_expression(candidate))
        gold_tree = _sympy_to_tree(_distance_expression(gold))
        distance = ext_distance(
            candidate_tree,
            gold_tree,
            get_children=lambda node: node.get_children(),
            single_insert_cost=_node_cost,
            insert_cost=_tree_cost,
            single_remove_cost=_node_cost,
            remove_cost=_tree_cost,
            update_cost=_update_cost,
        )
        gold_size = _tree_size(gold_tree)
        if distance == 0:
            return 1.0
        return max(0.0, 0.6 - distance / gold_size)
    except Exception:
        return 0.0


def score_answer(candidate_text: Any, gold_text: Any) -> Score:
    candidate_clean = clean_latex(candidate_text)
    gold_clean = clean_latex(gold_text)
    candidate_expr: sympy.Basic | None = None
    gold_expr: sympy.Basic | None = None
    candidate_error = ""
    gold_error = ""

    try:
        candidate_expr = parse_latex(candidate_clean)
    except Exception as exc:
        candidate_error = f"{type(exc).__name__}: {exc}"
    try:
        gold_expr = parse_latex(gold_clean)
    except Exception as exc:
        gold_error = f"{type(exc).__name__}: {exc}"

    candidate_ok = candidate_expr is not None
    gold_ok = gold_expr is not None
    candidate_canonical = _canonical(candidate_expr) if candidate_expr is not None else candidate_clean
    gold_canonical = _canonical(gold_expr) if gold_expr is not None else gold_clean

    if not gold_ok:
        return Score(0, 0.0, f"gold_parse_error: {gold_error}", candidate_canonical, gold_canonical, candidate_ok, False)
    if not candidate_ok:
        return Score(0, 0.0, f"candidate_parse_error: {candidate_error}", candidate_canonical, gold_canonical, False, True)

    equivalent, reason = _equivalent(candidate_expr, gold_expr)
    eed_fitness = 1.0 if equivalent else _eed_similarity(candidate_expr, gold_expr)
    return Score(int(equivalent), eed_fitness, reason, candidate_canonical, gold_canonical, True, True)


def canonical_answer(value: Any) -> str:
    cleaned = clean_latex(value)
    try:
        return _canonical(parse_latex(cleaned))
    except Exception:
        return re.sub(r"\s+", "", cleaned).lower()
