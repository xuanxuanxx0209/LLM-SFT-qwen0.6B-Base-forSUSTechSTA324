#!/usr/bin/env python3
"""Exact-match math reward for OpenRLHF single-turn GRPO."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from answer_utils import answers_match, clean_answer, extract_prediction


def reward_func(queries, prompts, labels):
    query = queries[0]
    label = labels[0]

    prediction = extract_prediction(query)
    gold = clean_answer(label)
    exact_match = float(gold != "" and answers_match(prediction, gold))

    return {
        "rewards": exact_match,
        "scores": exact_match,
        "extra_logs": {
            "exact_match": exact_match,
            "prediction_nonempty": float(prediction != ""),
        },
    }
