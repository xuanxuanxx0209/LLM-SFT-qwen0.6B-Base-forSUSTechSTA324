from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction


_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_HASH_RE = re.compile(r"####\s*(.+)")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?%?")


def _strip_wrappers(text: str) -> str:
    value = (text or "").strip()
    value = value.replace("$", "").replace("￥", "").replace(",", "")
    value = value.strip().strip(".").strip()
    if value.startswith("{") and value.endswith("}"):
        value = value[1:-1].strip()
    if value.startswith("(") and value.endswith(")"):
        value = value[1:-1].strip()
    return value


def clean_answer(text: str) -> str:
    value = _strip_wrappers(text)
    if not value:
        return ""
    return re.sub(r"\s+", "", value)


def extract_prediction(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    hash_matches = _HASH_RE.findall(raw)
    if hash_matches:
        return clean_answer(hash_matches[-1])

    boxed_matches = _BOXED_RE.findall(raw)
    if boxed_matches:
        return clean_answer(boxed_matches[-1])

    numeric_matches = _NUMBER_RE.findall(raw)
    if numeric_matches:
        return clean_answer(numeric_matches[-1])

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return clean_answer(lines[-1] if lines else raw)


def _to_decimal(value: str) -> Decimal | None:
    normalized = clean_answer(value)
    if not normalized:
        return None
    is_percent = normalized.endswith("%")
    if is_percent:
        normalized = normalized[:-1]
    try:
        if "/" in normalized and normalized.count("/") == 1:
            fraction = Fraction(normalized)
            number = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        else:
            number = Decimal(normalized)
        if is_percent:
            number = number / Decimal("100")
        return number.normalize()
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None


def answers_match(prediction: str, gold: str) -> bool:
    pred_clean = clean_answer(prediction)
    gold_clean = clean_answer(gold)
    if not pred_clean or not gold_clean:
        return False
    if pred_clean == gold_clean:
        return True

    pred_num = _to_decimal(pred_clean)
    gold_num = _to_decimal(gold_clean)
    if pred_num is None or gold_num is None:
        return False

    with localcontext() as ctx:
        ctx.prec = 28
        return abs(pred_num - gold_num) <= Decimal("1e-9")
