"""
header_calc.py — safe, sandboxed arithmetic evaluator for bulk trace-header
edits (SeiSee-style "CDP = TraceNumber * 2" expressions).

NEVER calls eval()/exec(). Expressions are parsed with the stdlib ``ast``
module and walked by hand against a strict whitelist of node types, binary/
unary/comparison operators, and a small set of vectorised NumPy functions —
anything else (attribute access, subscripting, comprehensions, lambdas,
imports, dunder access, ...) raises :class:`HeaderExprError` before a single
line of "user code" can run.

Overflow-safety note
---------------------
segyio always hands back trace-header fields as int32 NumPy arrays,
regardless of the field's true on-disk width (int16 fields are widened on
read). Plain NumPy *integer* arithmetic wraps silently on overflow — e.g.
``int32(3) * 1_000_000_000`` wraps to a negative number with no exception —
which would let a corrupting value sail past a naive range check. To avoid
that, every expression is evaluated internally in float64 (which saturates
to a detectable ``inf``/raises via ``errstate`` instead of wrapping), while
a separate boolean tracks the user's float-vs-integer *intent* (did the
expression use true division, sqrt, or a float literal?) so an
integer-only expression like ``CDP = TraceNumber * 2`` is correctly
recognised as integer-valued and is NOT blocked by the "no floats" rule.

This module only computes arrays — it never touches a file. Disk writes go
through ``io_segy.patch_trace_header_field``, which re-validates type/range
independently (defense in depth: a bug here can't corrupt a file).
"""
from __future__ import annotations

import ast
import operator
from typing import Dict, Optional, Tuple

import numpy as np

from .tasks import TopasCoreError


class HeaderExprError(TopasCoreError):
    """Raised for any unparseable, disallowed, or unsafe expression."""


# (numpy_op, force_float_result) — Div and Pow always yield a float *intent*
# even when both operands are integer-valued (matches Python/NumPy promotion
# for true division, and Pow is forced float here specifically so unbounded
# exponents can't quietly overflow an integer field — see module docstring).
_BIN_OPS = {
    ast.Add:      (operator.add,      False),
    ast.Sub:      (operator.sub,      False),
    ast.Mult:     (operator.mul,      False),
    ast.Div:      (operator.truediv,  True),
    ast.FloorDiv: (operator.floordiv, False),
    ast.Mod:      (operator.mod,      False),
    ast.Pow:      (operator.pow,      True),
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_COMPARE_OPS = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
# Small, explicit whitelist of vectorised NumPy functions, each tagged with
# whether its result is integer-valued by intent. "int" truncates toward
# zero — it is a controlled stand-in name, NOT Python's builtin.
_FUNCS = {
    "abs":   (np.abs,                                    None),
    "min":   (np.minimum,                                None),
    "max":   (np.maximum,                                None),
    "clip":  (np.clip,                                   None),
    "where": (np.where,                                  None),
    "round": (np.round,                                   False),
    "floor": (np.floor,                                   False),
    "ceil":  (np.ceil,                                    False),
    "sqrt":  (np.sqrt,                                    True),
    "int":   (lambda x: np.trunc(np.asarray(x, dtype=np.float64)), False),
}

_Val = Tuple[np.ndarray, bool]  # (float64 array, is_float_by_intent)


def _eval_node(node: ast.AST, variables: Dict[str, np.ndarray]) -> _Val:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, variables)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise HeaderExprError(f"Unsupported constant: {node.value!r}")
        return np.float64(node.value), isinstance(node.value, float)

    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise HeaderExprError(
                f"Unknown variable: '{node.id}'. Click Help for the list of "
                "available trace-header fields.")
        arr = np.asarray(variables[node.id])
        is_float = arr.dtype.kind == "f"
        return arr.astype(np.float64), is_float

    if isinstance(node, ast.BinOp):
        spec = _BIN_OPS.get(type(node.op))
        if spec is None:
            raise HeaderExprError(f"Operator not allowed: {type(node.op).__name__}")
        op, force_float = spec
        lval, lfloat = _eval_node(node.left, variables)
        rval, rfloat = _eval_node(node.right, variables)
        result = op(lval, rval)
        return result, (force_float or lfloat or rfloat)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise HeaderExprError(f"Unary operator not allowed: {type(node.op).__name__}")
        val, is_float = _eval_node(node.operand, variables)
        return op(val), is_float

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise HeaderExprError("Chained comparisons are not supported")
        op = _COMPARE_OPS.get(type(node.ops[0]))
        if op is None:
            raise HeaderExprError(f"Comparison not allowed: {type(node.ops[0]).__name__}")
        lval, _ = _eval_node(node.left, variables)
        rval, _ = _eval_node(node.comparators[0], variables)
        return op(lval, rval).astype(np.float64), False  # boolean 0/1 -> integer-flavored

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise HeaderExprError(
                "Only a small set of math functions is allowed: "
                f"{', '.join(sorted(_FUNCS))}")
        if node.keywords:
            raise HeaderExprError("Keyword arguments are not allowed")
        fn, forced_is_float = _FUNCS[node.func.id]
        evaluated = [_eval_node(a, variables) for a in node.args]
        args = [v for v, _ in evaluated]
        result = fn(*args)
        is_float = forced_is_float if forced_is_float is not None else any(f for _, f in evaluated)
        return np.asarray(result, dtype=np.float64), is_float

    raise HeaderExprError(f"Expression element not allowed: {type(node).__name__}")


def evaluate_header_expr(expr: str, variables: Dict[str, np.ndarray]) -> _Val:
    """Safely evaluate a single arithmetic expression against named NumPy
    array variables. Returns ``(result_float64_array, is_float_by_intent)``.
    See module docstring for the sandboxing and overflow-safety contract."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise HeaderExprError(f"Invalid expression syntax: {exc.msg}") from exc

    try:
        with np.errstate(all="raise"):
            result, is_float = _eval_node(tree, variables)
    except FloatingPointError as exc:
        raise HeaderExprError(f"Arithmetic overflow/invalid operation: {exc}") from exc
    except ZeroDivisionError as exc:
        raise HeaderExprError(f"Division by zero: {exc}") from exc

    result = np.asarray(result, dtype=np.float64)
    if result.size and not np.all(np.isfinite(result)):
        raise HeaderExprError(
            "Expression produced non-finite values (inf/NaN) — check for "
            "division by zero or an extreme exponent.")
    return result, is_float


def available_functions() -> Tuple[str, ...]:
    """Names of the whitelisted math functions usable in calculator
    expressions (the public face of the private ``_FUNCS`` table, for UI
    "Help" panels)."""
    return tuple(sorted(_FUNCS))


def parse_assignment(spec: str) -> Tuple[str, str]:
    """Split ``"CDP = TraceNumber * 2"`` into ``("CDP", "TraceNumber * 2")``.
    Raises HeaderExprError if there isn't exactly one top-level ``=``."""
    if spec.count("=") != 1:
        raise HeaderExprError(
            "Expression must be a single assignment, e.g. 'CDP = TraceNumber * 2'")
    target, _, rhs = spec.partition("=")
    target = target.strip()
    if not target.isidentifier():
        raise HeaderExprError(f"Invalid target field name: {target!r}")
    rhs = rhs.strip()
    if not rhs:
        raise HeaderExprError("Missing expression after '='")
    return target, rhs


def validate_header_result(
    field_name: str,
    result: np.ndarray,
    is_float: bool,
    int_range: Optional[Tuple[int, int]],
) -> Tuple[bool, Optional[np.ndarray], str]:
    """Validate a computed array against a trace-header field's integer type
    and byte-width range. Returns ``(ok, int64_values_or_None, message)``.

    Fail-closed: an expression whose RESULT IS FLOAT BY INTENT (used ``/``,
    ``sqrt``, a float literal, or ``**``) is rejected outright — even if the
    numbers happen to be numerically whole — unless the user explicitly
    wraps it in ``int(...)``. Unknown fields and any out-of-range value are
    also rejected. Nothing is written by this function; it only decides
    whether writing would be safe.
    """
    if is_float:
        return False, None, (
            f"Expression result is floating-point, but '{field_name}' is an "
            "integer SEG-Y field. Wrap the expression in int(...) to truncate "
            "explicitly, or use integer-only operators (e.g. // instead of /)."
        )
    arr = np.asarray(result, dtype=np.float64)
    if int_range is None:
        return False, None, (
            f"'{field_name}' is not a recognised fixed-width SEG-Y trace-header "
            "field — refusing to write for safety."
        )
    if arr.size == 0:
        return False, None, "Expression produced no values."
    lo, hi = int_range
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmin < lo or vmax > hi:
        return False, None, (
            f"Result range [{vmin:.0f}, {vmax:.0f}] overflows '{field_name}' "
            f"(valid range [{lo}, {hi}]) — refusing to write to avoid "
            "corrupting the file. Clip or rescale the expression first."
        )
    return True, np.round(arr).astype(np.int64), ""
