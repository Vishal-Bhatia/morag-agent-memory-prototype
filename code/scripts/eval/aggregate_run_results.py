r"""Aggregate judge-result summaries across datasets and runs.

Run from the code/ directory:

    python -m scripts.eval.aggregate_run_results
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from scripts.common.morag_utils import load_json, write_json


DEFAULT_RUNS_ROOT = pathlib.Path("data/runs")
DEFAULT_OUTPUT = pathlib.Path("data/results/all_runs_summary.json")


def infer_run_from_path(path: pathlib.Path, runs_root: pathlib.Path) -> dict[str, Any]:
    try:
        relative = path.relative_to(runs_root)
        parts = relative.parts
        if len(parts) >= 3:
            return {
                "dataset_slug": parts[0],
                "run_id": parts[1],
                "output_root": str(runs_root / parts[0] / parts[1]),
            }
    except ValueError:
        pass
    return {
        "dataset_slug": None,
        "run_id": None,
        "output_root": None,
    }


def flatten_answer_summary(
    result: dict[str, Any],
    source_file: pathlib.Path,
    runs_root: pathlib.Path,
) -> list[dict[str, Any]]:
    run = result.get("run") or infer_run_from_path(source_file, runs_root)
    rows: list[dict[str, Any]] = []
    for judge_model, methods in result.get("answer_summary", {}).items():
        for method, metrics in methods.items():
            rows.append(
                {
                    "summary_type": "answer",
                    "source_file": str(source_file),
                    "dataset_slug": run.get("dataset_slug"),
                    "dataset_source": run.get("dataset_source"),
                    "run_id": run.get("run_id"),
                    "output_root": run.get("output_root"),
                    "judge_model": judge_model,
                    "method": method,
                    **metrics,
                }
            )
    return rows


def flatten_retrieval_summary(
    result: dict[str, Any],
    source_file: pathlib.Path,
    runs_root: pathlib.Path,
) -> list[dict[str, Any]]:
    run = result.get("run") or infer_run_from_path(source_file, runs_root)
    rows: list[dict[str, Any]] = []
    for judge_model, methods in result.get("retrieval_summary", {}).items():
        for method, metrics in methods.items():
            rows.append(
                {
                    "summary_type": "retrieval",
                    "source_file": str(source_file),
                    "dataset_slug": run.get("dataset_slug"),
                    "dataset_source": run.get("dataset_source"),
                    "run_id": run.get("run_id"),
                    "output_root": run.get("output_root"),
                    "judge_model": judge_model,
                    "method": method,
                    **metrics,
                }
            )
    return rows


def aggregate_runs(runs_root: pathlib.Path) -> dict[str, Any]:
    judge_files = sorted(runs_root.glob("*/**/results/judge_results.json"))
    answer_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    run_index: list[dict[str, Any]] = []

    for judge_file in judge_files:
        result = load_json(judge_file)
        run = result.get("run") or infer_run_from_path(judge_file, runs_root)
        run_index.append(
            {
                **run,
                "judge_results_file": str(judge_file),
                "answer_judgment_count": result.get("answer_judgment_count", 0),
                "retrieval_judgment_count": result.get("retrieval_judgment_count", 0),
                "judge_models": result.get("judge_models", []),
                "answer_usage_totals": result.get("answer_usage_totals", {}),
                "retrieval_usage_totals": result.get("retrieval_usage_totals", {}),
            }
        )
        answer_rows.extend(flatten_answer_summary(result, judge_file, runs_root))
        retrieval_rows.extend(flatten_retrieval_summary(result, judge_file, runs_root))

    return {
        "runs_root": str(runs_root),
        "run_count": len(run_index),
        "answer_summary_row_count": len(answer_rows),
        "retrieval_summary_row_count": len(retrieval_rows),
        "runs": run_index,
        "answer_summary_rows": answer_rows,
        "retrieval_summary_rows": retrieval_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=pathlib.Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = aggregate_runs(args.runs_root)
    write_json(args.output, output)
    print(f"Aggregated {output['run_count']} runs")
    print(f"Answer summary rows: {output['answer_summary_row_count']}")
    print(f"Retrieval summary rows: {output['retrieval_summary_row_count']}")
    print(f"Wrote aggregate results to {args.output}")


if __name__ == "__main__":
    main()
