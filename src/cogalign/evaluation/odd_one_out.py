import json
from collections.abc import Iterable, Mapping
from glob import glob
from operator import itemgetter
from os import PathLike
from pathlib import Path
from string import punctuation
from typing import Any

_PUNCTUATION_TRANSLATION = str.maketrans("", "", punctuation + "\n")
Record = Mapping[str, Any]

def merge_infer_result(
    re_pattern: str | PathLike[str], *, tail_divisor: int | None = None
) -> list[Record]:
    if tail_divisor is not None and tail_divisor < 1:
        raise ValueError

    records = []
    for path in sorted(glob(str(re_pattern), recursive=True)):
        with Path(path).open(encoding="utf-8-sig") as file:
            shard = json.load(file)
        if not isinstance(shard, list):
            raise ValueError
        records.extend(shard)

    records.sort(key=itemgetter("id"))
    if tail_divisor is None:
        return records
    tail_size = len(records) // tail_divisor
    return records[len(records) - tail_size:]

def remove_punctuation(text: str) -> str:
    return text.translate(_PUNCTUATION_TRANSLATION)

def extract_ans(text: str, options: Iterable[str]) -> str | None:
    text = text.lower()
    answer_start = text.rfind("answer")
    answer_tokens = set(remove_punctuation(text[answer_start:]).split(" "))
    answer = None
    for option in options:
        option = option.lower()
        if option in answer_tokens:
            answer = option
    return answer

def build_correctness(records: Iterable[Record]) -> list[int]:
    correctness = []
    for record in records:
        label = record["label_choice"].lower()
        response = record["response"]
        options = record["option"]
        try:
            answer = extract_ans(response, options)
        except (AttributeError, TypeError):
            answer = None

        prediction = "none" if answer is None else answer.strip()
        correctness.append(int(prediction == label))
    return correctness

def summarize_accuracy(correctness_array: Iterable[int]) -> dict[str, float | int | None]:
    correctness = list(correctness_array)
    total = len(correctness)
    true_num = int(sum(correctness))
    return {
        "accuracy": true_num / total if total else None,
        "true_num": true_num,
        "false_num": total - true_num,
        "total": total,
    }

def calculate_bootstrap_ci(
    correctness_array: Iterable[int | float],
    n_iterations: int = 1000,
    alpha: float = 0.95,
    seed: int | None = 42,
) -> tuple[float, float, float, list[float]]:
    if n_iterations < 1:
        raise ValueError
    if not 0.0 <= alpha <= 1.0:
        raise ValueError
    values = list(correctness_array)
    if not values:
        raise ValueError

    import numpy as np

    correctness = np.asarray(values)
    random = np.random.RandomState(seed)
    scores = []
    for _ in range(n_iterations):
        sample = random.choice(correctness, size=len(correctness), replace=True)
        scores.append(float(np.mean(sample)))

    tail = (1.0 - alpha) / 2.0
    lower, upper = np.percentile(scores, [tail * 100, (alpha + tail) * 100])
    return float(np.mean(correctness)), float(lower), float(upper), scores
