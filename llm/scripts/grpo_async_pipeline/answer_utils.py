#!/usr/bin/env python3
"""Answer extraction helpers for math-style exact-match GRPO."""

from __future__ import annotations

import json
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path


FINAL_PATTERNS = [
    re.compile(r"####\s*([^\n]+)"),
    re.compile(r"Final Answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
    re.compile(r"The answer is\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"Answer\s*[:：]\s*([^\n]+)", re.IGNORECASE),
]
OPTION_KEYWORD_PATTERNS = [
    re.compile(
        r"(?:correct(?: answer| choice)?|answer|choice|option|corresponds to)\s*(?:is\s*)?(?:option\s*)?"
        r"[\(\[]?\s*([A-E])\s*[\)\]]?",
        re.IGNORECASE,
    ),
]
INLINE_OPTION_PATTERN = re.compile(r"^[^A-Za-z0-9]*([A-E])[^A-Za-z0-9]*$")
XML_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
FRACTION_PATTERN = re.compile(r"-?\d+\s*/\s*-?\d+")
WHITESPACE_PATTERN = re.compile(r"\s+")
PLACEHOLDER_BITS = (
    "<answer>",
    "#####",
    "uetype",
    "</user>",
    "</system>",
    "assistant",
    "#user",
)
BOX_MACROS = ("\\boxed", "\\fbox")
WRAPPER_MACROS = ("\\text", "\\mathrm", "\\mathbf", "\\operatorname")
NUMERIC_AFTER_EQUALS_PATTERN = re.compile(
    r"=\s*(-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?)\b"
)
LABEL_NOISE_WORDS = {
    "answer",
    "answers",
    "because",
    "correct",
    "equation",
    "figure",
    "given",
    "hence",
    "integer",
    "numbers",
    "option",
    "options",
    "product",
    "question",
    "so",
    "speed",
    "therefore",
    "third",
    "thus",
    "or",
    "no",
    "where",
}
ALLOWED_MATH_WORDS = {
    "cos",
    "cosh",
    "cot",
    "csc",
    "exp",
    "gcd",
    "lcm",
    "ln",
    "log",
    "max",
    "min",
    "mod",
    "sec",
    "sin",
    "sinh",
    "tan",
    "tanh",
}
RELATION_SYMBOLS = ("=", "<", ">", "≤", "≥")
BALANCE_REPLACEMENTS = {
    "\\(": "(",
    "\\)": ")",
    "\\[": "[",
    "\\]": "]",
    "\\{": "{",
    "\\}": "}",
}
BRACKET_PAIRS = {
    ")": "(",
    "]": "[",
    "}": "{",
}
MULTI_ASSIGNMENT_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_']*\s*=")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _extract_outer_macro_argument(text: str, macro: str) -> str | None:
    prefix = f"{macro}{{"
    if not text.startswith(prefix):
        return None

    depth = 0
    start = len(macro)
    end = None
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = index
                break

    if end is None or end != len(text) - 1:
        return None
    return text[start + 1 : end]


def _unwrap_outer_delimiters(text: str) -> str:
    updated = text.strip()
    changed = True
    while changed and updated:
        changed = False
        stripped = updated.strip()

        for left, right in (("\\(", "\\)"), ("\\[", "\\]"), ("$", "$")):
            if stripped.startswith(left) and stripped.endswith(right) and len(stripped) > len(left) + len(right):
                updated = stripped[len(left) : -len(right)].strip()
                changed = True
                stripped = updated

        if stripped.startswith("(") and stripped.endswith(")") and len(stripped) > 2:
            inner = stripped[1:-1].strip()
            if inner:
                updated = inner
                changed = True
    return updated.strip()


def _unwrap_known_macros(text: str) -> str:
    updated = text.strip()
    changed = True
    while changed and updated:
        changed = False
        stripped = updated.strip()
        for macro in (*BOX_MACROS, *WRAPPER_MACROS):
            inner = _extract_outer_macro_argument(stripped, macro)
            if inner is not None:
                updated = inner.strip()
                changed = True
                break
    return updated.strip()


def _extract_macro_contents(text: str, macro: str) -> list[str]:
    matches: list[str] = []
    token = f"{macro}{{"
    index = 0
    while True:
        start = text.find(token, index)
        if start < 0:
            break

        depth = 0
        end = None
        brace_start = start + len(macro)
        for cursor in range(brace_start, len(text)):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = cursor
                    break

        if end is None:
            break

        matches.append(text[brace_start + 1 : end])
        index = end + 1
    return matches


def _candidate_scopes(text: str) -> list[str]:
    scopes: list[str] = []
    stripped = text.strip()
    if not stripped:
        return scopes

    if "</think>" in stripped:
        suffix = stripped.rsplit("</think>", 1)[-1].strip()
        if suffix:
            scopes.append(suffix)

    if "</answer>" in stripped and "<answer>" in stripped:
        for candidate in reversed(XML_ANSWER_PATTERN.findall(stripped)):
            candidate = candidate.strip()
            if candidate:
                scopes.append(candidate)

    scopes.append(stripped)

    unique_scopes: list[str] = []
    seen = set()
    for scope in scopes:
        if scope not in seen:
            unique_scopes.append(scope)
            seen.add(scope)
    return unique_scopes


def _gold_candidate_scopes(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    if "</think>" in stripped:
        suffix = stripped.rsplit("</think>", 1)[-1].strip()
        return [suffix] if suffix else []

    return [stripped]


def is_plausible_fragment(text: str | None) -> bool:
    if text is None:
        return False
    text = text.strip()
    if not text or re.search(r"[A-Za-z0-9]", text) is None:
        return False
    lowered = text.lower()
    return not any(bit in lowered for bit in PLACEHOLDER_BITS)


def _normalize_option(candidate: str) -> str | None:
    text = _unwrap_outer_delimiters(_unwrap_known_macros(candidate))
    if not text:
        return None

    matched = INLINE_OPTION_PATTERN.fullmatch(text)
    if matched:
        return matched.group(1).upper()

    letters = re.findall(r"[A-E]", text.upper())
    if len(letters) == 1 and re.sub(r"[^A-Za-z]", "", text).upper() == letters[0]:
        return letters[0]
    return None


def _extract_option(scope: str) -> str | None:
    normalized_scope = scope
    for source, target in BALANCE_REPLACEMENTS.items():
        normalized_scope = normalized_scope.replace(source, target)

    for pattern in OPTION_KEYWORD_PATTERNS:
        matches = pattern.findall(normalized_scope)
        if matches:
            return matches[-1].upper()

    lines = [line.strip() for line in normalized_scope.splitlines() if line.strip()]
    for line in reversed(lines[-6:]):
        option = _normalize_option(line)
        if option is not None:
            return option
    return None


def clean_answer(text: str | None) -> str:
    if text is None:
        return ""

    updated = unicodedata.normalize("NFKC", text).strip()
    updated = updated.replace(",", "")
    updated = updated.replace("$", "")
    updated = updated.replace("%", "")
    updated = updated.replace("\u00a0", " ")
    updated = WHITESPACE_PATTERN.sub(" ", updated)
    updated = _unwrap_outer_delimiters(_unwrap_known_macros(updated))
    updated = updated.strip(" .`*[]")
    updated = WHITESPACE_PATTERN.sub(" ", updated).strip()

    option = _normalize_option(updated)
    if option is not None:
        return option

    if updated.lower() == "none":
        return "None"

    fractions = FRACTION_PATTERN.findall(updated)
    if len(fractions) == 1 and len(NUMBER_PATTERN.findall(updated)) >= 2:
        return fractions[0].replace(" ", "")

    numbers = NUMBER_PATTERN.findall(updated)
    if len(numbers) == 1:
        return numbers[0]

    if len(numbers) >= 2 and len(set(numbers)) == 1:
        return numbers[0]

    if any(operator in updated for operator in ("=", "+", "-", "*", "/", "^", "<", ">")):
        updated = re.sub(r"\s*([=+\-*/^<>])\s*", r"\1", updated)

    return updated


def _extract_numeric_after_equals(text: str) -> str | None:
    matches = NUMERIC_AFTER_EQUALS_PATTERN.findall(unicodedata.normalize("NFKC", text))
    if not matches:
        return None
    return clean_answer(matches[-1])


def _has_invalid_escape_sequence(text: str) -> bool:
    for index, char in enumerate(text):
        if char != "\\" or index + 1 >= len(text):
            continue
        next_char = text[index + 1]
        if next_char.isalpha() or next_char in "()[]{}":
            continue
        return True
    return False


def _has_balanced_delimiters(text: str) -> bool:
    normalized = text
    for source, target in BALANCE_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)

    stack: list[str] = []
    for char in normalized:
        if char in "([{":
            stack.append(char)
        elif char in BRACKET_PAIRS:
            if not stack or stack[-1] != BRACKET_PAIRS[char]:
                return False
            stack.pop()
    return not stack


def _looks_like_number_list_artifact(text: str) -> bool:
    numbers = NUMBER_PATTERN.findall(text)
    if len(numbers) < 3:
        return False
    if re.search(r"[A-Za-z]", text):
        return False
    if any(operator in text for operator in "=<>+-*/^:"):
        return False
    return True


def _looks_like_exact_match_label(text: str) -> bool:
    cleaned = clean_answer(text)
    if not cleaned or len(cleaned) > 120:
        return False
    if re.search(r"[A-Za-z0-9]", cleaned) is None:
        return False
    if cleaned.startswith(("-\\(", "-(")):
        return False
    if any(token in cleaned for token in ("→", "=>", "&", ";")):
        return False
    if cleaned.count(":") >= 2:
        return False
    if _has_invalid_escape_sequence(cleaned):
        return False
    if not _has_balanced_delimiters(cleaned):
        return False
    if re.search(r"[A-Za-z0-9]\\[(){}\[\]]", cleaned):
        return False
    if cleaned.count("=") + cleaned.count("<") + cleaned.count(">") + cleaned.count("≤") + cleaned.count("≥") > 1:
        return False
    if len(MULTI_ASSIGNMENT_PATTERN.findall(cleaned)) >= 2:
        return False
    if "\\in" in cleaned and len(set(NUMBER_PATTERN.findall(cleaned))) >= 2:
        return False
    if _looks_like_number_list_artifact(cleaned):
        return False

    option = _normalize_option(cleaned)
    if option is not None:
        return True

    if FRACTION_PATTERN.fullmatch(cleaned.replace(" ", "")):
        return True
    if NUMBER_PATTERN.fullmatch(cleaned):
        return True

    words = re.findall(r"[A-Za-z]+", cleaned)
    informative_words = [word.lower() for word in words if len(word) > 1]
    for word in informative_words:
        if word in LABEL_NOISE_WORDS:
            return False
        if len(word) > 2 and word not in ALLOWED_MATH_WORDS:
            return False

    return True


def extract_prediction(text: str) -> str:
    for scope in _candidate_scopes(text):
        option = _extract_option(scope)
        if option is not None:
            return option

        for candidate in reversed(XML_ANSWER_PATTERN.findall(scope)):
            if is_plausible_fragment(candidate):
                cleaned = clean_answer(candidate)
                if cleaned:
                    return cleaned

        for macro in BOX_MACROS:
            candidates = _extract_macro_contents(scope, macro)
            for candidate in reversed(candidates):
                if is_plausible_fragment(candidate):
                    cleaned = clean_answer(candidate)
                    if cleaned:
                        return cleaned

        for pattern in FINAL_PATTERNS:
            matches = pattern.findall(scope)
            for candidate in reversed(matches):
                if is_plausible_fragment(candidate):
                    cleaned = clean_answer(candidate)
                    if cleaned:
                        return cleaned

        lines = [line.strip() for line in scope.splitlines() if line.strip()]
        for line in reversed(lines[-10:]):
            option = _normalize_option(line)
            if option is not None:
                return option

            for macro in BOX_MACROS:
                candidates = _extract_macro_contents(line, macro)
                for candidate in reversed(candidates):
                    if is_plausible_fragment(candidate):
                        cleaned = clean_answer(candidate)
                        if cleaned:
                            return cleaned

            if not is_plausible_fragment(line):
                continue

            fractions = FRACTION_PATTERN.findall(line)
            if len(fractions) == 1:
                return fractions[0].replace(" ", "")

            numeric_after_equals = _extract_numeric_after_equals(line)
            if numeric_after_equals:
                return numeric_after_equals

            numbers = NUMBER_PATTERN.findall(line)
            if len(numbers) == 1:
                return numbers[0]

            cleaned = clean_answer(line)
            if cleaned and len(cleaned) <= 160:
                return cleaned

    numbers = NUMBER_PATTERN.findall(text.replace(",", ""))
    if numbers:
        return clean_answer(numbers[-1])
    return clean_answer(text)


def extract_gold_label(text: str) -> str:
    for scope in _gold_candidate_scopes(text):
        option = _extract_option(scope)
        if option is not None:
            return option

        for candidate in reversed(XML_ANSWER_PATTERN.findall(scope)):
            if not is_plausible_fragment(candidate):
                continue
            cleaned = clean_answer(candidate)
            if _looks_like_exact_match_label(cleaned):
                return cleaned

        for macro in BOX_MACROS:
            candidates = _extract_macro_contents(scope, macro)
            for candidate in reversed(candidates):
                if not is_plausible_fragment(candidate):
                    continue
                cleaned = clean_answer(candidate)
                if _looks_like_exact_match_label(cleaned):
                    return cleaned

        for pattern in FINAL_PATTERNS:
            matches = pattern.findall(scope)
            for candidate in reversed(matches):
                if not is_plausible_fragment(candidate):
                    continue
                cleaned = clean_answer(candidate)
                if _looks_like_exact_match_label(cleaned):
                    return cleaned

        lines = [line.strip() for line in scope.splitlines() if line.strip()]
        for line in reversed(lines[-6:]):
            option = _normalize_option(line)
            if option is not None:
                return option

            for macro in BOX_MACROS:
                candidates = _extract_macro_contents(line, macro)
                for candidate in reversed(candidates):
                    if not is_plausible_fragment(candidate):
                        continue
                    cleaned = clean_answer(candidate)
                    if _looks_like_exact_match_label(cleaned):
                        return cleaned

            cleaned = clean_answer(line)
            if _looks_like_exact_match_label(cleaned):
                return cleaned

    return ""


def answers_match(prediction: str, gold: str) -> bool:
    pred = clean_answer(prediction)
    ref = clean_answer(gold)
    if pred == ref:
        return True

    try:
        return Decimal(pred) == Decimal(ref)
    except (InvalidOperation, ValueError):
        return False
