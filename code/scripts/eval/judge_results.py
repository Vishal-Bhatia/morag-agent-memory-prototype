r"""Judge support-agent outputs and retrieval quality.

Run from the code/ directory:

    python .\scripts\judge_results.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from typing import Any

from scripts.common.cli_config import (
    add_config_args,
    load_experiment_and_model_configs,
    run_metadata,
)
from scripts.common.morag_utils import (
    call_json_model,
    load_json,
    load_prompt,
    openai_client_from_config,
    render_template,
    write_json,
)


ANSWER_SCORE_FIELDS = (
    "relevance",
    "specificity",
    "faithfulness",
    "groundedness",
    "tone",
    "overall_score",
)
RETRIEVAL_SCORE_FIELDS = ("hit_rate", "retrieval_quality", "memory_pollution_present")
RETRIEVAL_USEFULNESS_SCORE_FIELDS = ("context_usefulness", "context_specificity")
RETRIEVAL_USEFULNESS_BINARY_FIELDS = (
    "exact_answer_present",
    "noise_present",
    "misleading_context_present",
)


def compact_context(result: dict[str, Any]) -> str:
    if not result.get("retrieved_items"):
        return "None."
    return json.dumps(result["retrieved_items"], indent=2, ensure_ascii=False)


def build_answer_prompt(template: str, result: dict[str, Any], judge_model: str) -> str:
    return render_template(
        template,
        {
            "query_id": result["case_id"],
            "future_query": result["future_query"],
            "method": result["method"],
            "evidence_context": compact_context(result),
            "answer": result["answer"],
            "judge_model": judge_model,
        },
    )


def build_retrieval_prompt(template: str, result: dict[str, Any], judge_model: str) -> str:
    return render_template(
        template,
        {
            "query_id": result["case_id"],
            "future_query": result["future_query"],
            "expected_resolution_pattern": result["expected_resolution_pattern"],
            "retrieved_context": compact_context(result),
            "method": result["method"],
            "judge_model": judge_model,
        },
    )


def coerce_int(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_answer_judgment(
    payload: dict[str, Any],
    result: dict[str, Any],
    judge_model: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    normalized = {
        "query_id": result["case_id"],
        "method": result["method"],
        "judge_model": judge_model,
        "source_conversation_id": result["source_conversation_id"],
    }
    for key in ANSWER_SCORE_FIELDS:
        normalized[key] = max(1, min(5, coerce_int(payload, key, 1)))
    normalized["critical_failure"] = 1 if coerce_int(payload, "critical_failure", 0) else 0
    normalized["rationale"] = str(payload.get("rationale", "")).strip()
    normalized["usage"] = usage
    return normalized


def normalize_retrieval_judgment(
    payload: dict[str, Any],
    result: dict[str, Any],
    judge_model: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    normalized = {
        "query_id": result["case_id"],
        "method": result["method"],
        "judge_model": judge_model,
        "source_conversation_id": result["source_conversation_id"],
    }
    normalized["hit_rate"] = 1 if coerce_int(payload, "hit_rate", 0) else 0
    normalized["retrieval_quality"] = max(1, min(5, coerce_int(payload, "retrieval_quality", 1)))
    normalized["memory_pollution_present"] = (
        1 if coerce_int(payload, "memory_pollution_present", 0) else 0
    )
    normalized["rationale"] = str(payload.get("rationale", "")).strip()
    normalized["usage"] = usage
    return normalized


def normalize_retrieval_usefulness_judgment(
    payload: dict[str, Any],
    result: dict[str, Any],
    judge_model: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    normalized = {
        "query_id": result["case_id"],
        "method": result["method"],
        "judge_model": judge_model,
        "source_conversation_id": result["source_conversation_id"],
    }
    for key in RETRIEVAL_USEFULNESS_SCORE_FIELDS:
        normalized[key] = max(1, min(5, coerce_int(payload, key, 1)))
    for key in RETRIEVAL_USEFULNESS_BINARY_FIELDS:
        normalized[key] = 1 if coerce_int(payload, key, 0) else 0
    normalized["rationale"] = str(payload.get("rationale", "")).strip()
    normalized["usage"] = usage
    return normalized


def summarize_usage(items: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for item in items:
        for key, value in item.get("usage", {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def mean(values: list[float]) -> float:
    return round(statistics.mean(values), 3) if values else 0.0


def summarize_answer_judgments(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = f"{item['judge_model']}::{item['method']}"
        groups.setdefault(key, []).append(item)

    summary: dict[str, Any] = {}
    for key, rows in groups.items():
        judge_model, method = key.split("::", maxsplit=1)
        summary.setdefault(judge_model, {})[method] = {
            field: mean([row[field] for row in rows]) for field in ANSWER_SCORE_FIELDS
        } | {
            "critical_failure_rate": mean([row["critical_failure"] for row in rows]),
            "count": len(rows),
        }
    return summary


def summarize_retrieval_judgments(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = f"{item['judge_model']}::{item['method']}"
        groups.setdefault(key, []).append(item)

    summary: dict[str, Any] = {}
    for key, rows in groups.items():
        judge_model, method = key.split("::", maxsplit=1)
        summary.setdefault(judge_model, {})[method] = {
            "hit_rate": mean([row["hit_rate"] for row in rows]),
            "retrieval_quality": mean([row["retrieval_quality"] for row in rows]),
            "memory_pollution_rate": mean([row["memory_pollution_present"] for row in rows]),
            "count": len(rows),
        }
    return summary


def summarize_retrieval_usefulness_judgments(items: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        key = f"{item['judge_model']}::{item['method']}"
        groups.setdefault(key, []).append(item)

    summary: dict[str, Any] = {}
    for key, rows in groups.items():
        judge_model, method = key.split("::", maxsplit=1)
        summary.setdefault(judge_model, {})[method] = {
            "context_usefulness": mean([row["context_usefulness"] for row in rows]),
            "context_specificity": mean([row["context_specificity"] for row in rows]),
            "exact_answer_rate": mean([row["exact_answer_present"] for row in rows]),
            "noise_rate": mean([row["noise_present"] for row in rows]),
            "misleading_context_rate": mean([row["misleading_context_present"] for row in rows]),
            "count": len(rows),
        }
    return summary


def parse_judge_models(value: str | None, model_config: dict[str, Any]) -> list[str]:
    def unique_nonempty(models: list[str]) -> list[str]:
        output: list[str] = []
        for model in models:
            clean = str(model).strip()
            if clean and clean not in output:
                output.append(clean)
        return output

    if value:
        return unique_nonempty(value.split(","))
    return unique_nonempty(
        [
            model_config["models"].get("judge_primary_model", ""),
            model_config["models"].get("judge_secondary_model", ""),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--judge-models")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-answer", action="store_true")
    parser.add_argument("--skip-retrieval", action="store_true")
    parser.add_argument("--skip-exact-retrieval", action="store_true")
    parser.add_argument("--skip-usefulness-retrieval", action="store_true")
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    input_path = args.input or pathlib.Path(config["outputs"]["run_results_file"])
    output_path = args.output or pathlib.Path(config["outputs"]["judge_results_file"])
    judge_models = parse_judge_models(args.judge_models, model_config)
    temperature = float(model_config["generation"]["temperature"])

    run_results = load_json(input_path)["results"]
    if args.limit is not None:
        case_ids = []
        selected = []
        for result in run_results:
            if result["case_id"] not in case_ids:
                case_ids.append(result["case_id"])
            if len(case_ids) <= args.limit:
                selected.append(result)
        run_results = selected

    answer_template = load_prompt(pathlib.Path(config["prompts"]["answer_judge"]))
    retrieval_template = load_prompt(pathlib.Path(config["prompts"]["retrieval_judge"]))
    retrieval_usefulness_template = load_prompt(
        pathlib.Path(config["prompts"]["retrieval_usefulness_judge"])
    )
    client = openai_client_from_config(model_config)

    answer_judgments: list[dict[str, Any]] = []
    retrieval_judgments: list[dict[str, Any]] = []
    retrieval_usefulness_judgments: list[dict[str, Any]] = []
    if not args.skip_answer:
        total_answer_calls = len(run_results) * len(judge_models)
        answer_call = 0
        for judge_model in judge_models:
            for result in run_results:
                answer_call += 1
                print(
                    f"[answer {answer_call}/{total_answer_calls}] "
                    f"{judge_model} {result['method']} {result['case_id']}"
                )
                payload, usage = call_json_model(
                    client=client,
                    model=judge_model,
                    prompt=build_answer_prompt(answer_template, result, judge_model),
                    temperature=temperature,
                )
                answer_judgments.append(
                    normalize_answer_judgment(payload, result, judge_model, usage)
                )

    retrieval_candidates = [
        result for result in run_results if result["method"] in {"reference_rag", "morag"}
    ]
    if not args.skip_retrieval and not args.skip_exact_retrieval:
        total_retrieval_calls = len(retrieval_candidates) * len(judge_models)
        retrieval_call = 0
        for judge_model in judge_models:
            for result in retrieval_candidates:
                retrieval_call += 1
                print(
                    f"[retrieval {retrieval_call}/{total_retrieval_calls}] "
                    f"{judge_model} {result['method']} {result['case_id']}"
                )
                payload, usage = call_json_model(
                    client=client,
                    model=judge_model,
                    prompt=build_retrieval_prompt(retrieval_template, result, judge_model),
                    temperature=temperature,
                )
                retrieval_judgments.append(
                    normalize_retrieval_judgment(payload, result, judge_model, usage)
                )

    if not args.skip_retrieval and not args.skip_usefulness_retrieval:
        total_usefulness_calls = len(retrieval_candidates) * len(judge_models)
        usefulness_call = 0
        for judge_model in judge_models:
            for result in retrieval_candidates:
                usefulness_call += 1
                print(
                    f"[retrieval-usefulness {usefulness_call}/{total_usefulness_calls}] "
                    f"{judge_model} {result['method']} {result['case_id']}"
                )
                payload, usage = call_json_model(
                    client=client,
                    model=judge_model,
                    prompt=build_retrieval_prompt(
                        retrieval_usefulness_template,
                        result,
                        judge_model,
                    ),
                    temperature=temperature,
                )
                retrieval_usefulness_judgments.append(
                    normalize_retrieval_usefulness_judgment(
                        payload,
                        result,
                        judge_model,
                        usage,
                    )
                )

    output = {
        "run": run_metadata(config),
        "source_file": str(input_path),
        "judge_models": judge_models,
        "answer_judgment_count": len(answer_judgments),
        "retrieval_judgment_count": len(retrieval_judgments),
        "retrieval_usefulness_judgment_count": len(retrieval_usefulness_judgments),
        "answer_usage_totals": summarize_usage(answer_judgments),
        "retrieval_usage_totals": summarize_usage(retrieval_judgments),
        "retrieval_usefulness_usage_totals": summarize_usage(
            retrieval_usefulness_judgments
        ),
        "answer_summary": summarize_answer_judgments(answer_judgments),
        "retrieval_summary": summarize_retrieval_judgments(retrieval_judgments),
        "retrieval_usefulness_summary": summarize_retrieval_usefulness_judgments(
            retrieval_usefulness_judgments
        ),
        "answer_judgments": answer_judgments,
        "retrieval_judgments": retrieval_judgments,
        "retrieval_usefulness_judgments": retrieval_usefulness_judgments,
    }
    write_json(output_path, output)

    print(f"Wrote judge results to {output_path}")
    print(f"Answer judgments: {len(answer_judgments)}")
    print(f"Retrieval judgments: {len(retrieval_judgments)}")
    print(f"Retrieval usefulness judgments: {len(retrieval_usefulness_judgments)}")
    print(f"Answer usage totals: {output['answer_usage_totals']}")
    print(f"Retrieval usage totals: {output['retrieval_usage_totals']}")
    print(
        "Retrieval usefulness usage totals: "
        f"{output['retrieval_usefulness_usage_totals']}"
    )


if __name__ == "__main__":
    main()
