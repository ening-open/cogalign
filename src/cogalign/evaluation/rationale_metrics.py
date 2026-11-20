import json
from collections.abc import Iterable, Mapping
from glob import glob
from operator import itemgetter
from os import PathLike
from pathlib import Path
from statistics import mean, pstdev, variance
from typing import Any

JsonPath = str | PathLike[str]
Record = Mapping[str, Any]
RecordSource = Iterable[Record] | JsonPath

def read_json(path: JsonPath) -> Any:
    with Path(path).open(encoding="utf-8-sig") as file:
        return json.load(file)

def save_json(data: Any, path: JsonPath) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")

def _read_records(source: RecordSource) -> list[Record]:
    if isinstance(source, (str, PathLike)):
        return _read_record_file(source)
    return list(source)

def _read_record_file(path: JsonPath) -> list[Record]:
    records = read_json(path)
    if not isinstance(records, list):
        raise ValueError
    return records

def merge_json_parts(pattern: JsonPath) -> list[Record]:
    records = []
    for path in sorted(glob(str(pattern), recursive=True)):
        records.extend(_read_record_file(path))
    return records

def _align_records(
    left: RecordSource, right: RecordSource
) -> list[tuple[Record, Record]]:
    left_records = sorted(_read_records(left), key=itemgetter("id"))
    right_records = sorted(_read_records(right), key=itemgetter("id"))
    if len(left_records) != len(right_records):
        raise ValueError

    pairs = list(zip(left_records, right_records))
    for left_record, right_record in pairs:
        left_id = left_record["id"]
        right_id = right_record["id"]
        if left_id != right_id:
            raise ValueError
    return pairs

def _extract_rationale(record: Record) -> str:
    _, separator, rationale = record["response"].partition(":")
    if not separator:
        raise ValueError
    return rationale.strip()

def _paired_rationales(
    ds_records: RecordSource, qwen_records: RecordSource
) -> Iterable[tuple[Record, str, str]]:
    for ds_record, qwen_record in _align_records(ds_records, qwen_records):
        yield qwen_record, _extract_rationale(ds_record), _extract_rationale(qwen_record)

def load_embedding_model(
    model_path: JsonPath,
    *,
    device: str | None = None,
    **kwargs: Any,
):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), **kwargs)
    return model if device is None else model.to(device)

def compare_pair_similarity(
    ds_records: RecordSource,
    qwen_records: RecordSource,
    model,
    *,
    output_path: JsonPath | None = None,
) -> list[dict[str, Any]]:
    results = []
    for qwen_record, ds_text, qwen_text in _paired_rationales(ds_records, qwen_records):
        query_embedding = model.encode(qwen_text)
        document_embedding = model.encode(ds_text)
        results.append({
            "id": qwen_record["id"],
            "sim": float(model.similarity(query_embedding, document_embedding)),
            "dim_name": qwen_record.get("dim_name"),
        })

    if output_path is not None:
        save_json(results, output_path)
    return results

def count_pair_tokens(
    ds_records: RecordSource,
    qwen_records: RecordSource,
    tokenizer,
    *,
    add_special_tokens: bool = False,
    output_path: JsonPath | None = None,
) -> list[dict[str, Any]]:
    results = []
    for qwen_record, ds_text, qwen_text in _paired_rationales(ds_records, qwen_records):
        qwen_ids = tokenizer.encode(
            qwen_text, add_special_tokens=add_special_tokens, truncation=False
        )
        ds_ids = tokenizer.encode(
            ds_text, add_special_tokens=add_special_tokens, truncation=False
        )
        results.append({
            "id": qwen_record["id"],
            "qwen_token": len(qwen_ids),
            "ds_token": len(ds_ids),
        })

    if output_path is not None:
        save_json(results, output_path)
    return results

def summarize_metrics(
    records: RecordSource,
    fields: str | Iterable[str],
    *,
    require_dimension: bool = False,
) -> dict[str, dict[str, float | int | None]]:
    selected = _read_records(records)
    if require_dimension:
        selected = [record for record in selected if record.get("dim_name") is not None]
    if isinstance(fields, str):
        fields = (fields,)

    summaries = {}
    for field in fields:
        values = [record[field] for record in selected]
        summaries[field] = {
            "mean": mean(values) if values else None,
            "std": pstdev(values) if values else None,
            "min": min(values, default=None),
            "max": max(values, default=None),
            "sum": sum(values),
            "sample_variance": variance(values) if len(values) > 1 else None,
        }
    return summaries

def summarize_judgments(
    raw_records: RecordSource, judgment_records: RecordSource
) -> dict[str, Any]:
    pairs = _align_records(raw_records, judgment_records)
    counts = dict.fromkeys(("d", "f", "l", "n", "o", "u"), 0)
    for _, judgment in pairs:
        response = judgment["response"]
        if isinstance(response, str):
            response = json.loads(response)
        content = response["choices"][0]["message"]["content"]
        if isinstance(content, str):
            content = json.loads(content)

        for field in ("d", "f", "l", "n"):
            counts[field] += content[field]
        verdict = content["o"].lower()
        counts["o"] += verdict == "c"
        counts["u"] += verdict == "u"

    total = len(pairs)
    adjusted_total = total - counts["u"]
    return {
        "total": total,
        "counts": counts,
        "ratios": {
            field: count / total if total else None for field, count in counts.items()
        },
        "adjusted_o_ratio": counts["o"] / adjusted_total if adjusted_total else None,
    }
