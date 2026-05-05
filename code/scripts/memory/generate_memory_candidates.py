r"""Generate moRAG memory candidates from experience cases.

Run a small pilot first:

    python .\scripts\generate_memory_candidates.py --limit 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from datetime import date, timedelta
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


def iso_after_days(start_date: date, days: int) -> str:
    return (start_date + timedelta(days=days)).isoformat()


def expiry_label_to_date(value: Any, run_date: date, default_review_days: int) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", stripped):
            return stripped

        match = re.match(r"^review_after_(\d+)_days$", stripped)
        if match:
            return iso_after_days(run_date, int(match.group(1)))

    return iso_after_days(run_date, default_review_days)


def build_prompt(
    template: str,
    case: dict[str, Any],
    run_date: date,
    default_review_days: int,
    short_review_days: int,
) -> str:
    return render_template(
        template,
        {
            "conversation": case["past_conversation_text"],
            "metadata": json.dumps(case["metadata"], indent=2),
            "current_date": run_date.isoformat(),
            "default_expires_on": iso_after_days(run_date, default_review_days),
            "short_expires_on": iso_after_days(run_date, short_review_days),
        },
    )


def normalize_candidate(
    candidate: dict[str, Any],
    source_conversation_id: str,
    case_id: str,
    case_metadata: dict[str, Any],
    ordinal: int,
    memory_id_prefix: str,
    run_date: date,
    default_review_days: int,
) -> dict[str, Any]:
    normalized = dict(candidate)
    model_memory_id = normalized.get("memory_id")
    if model_memory_id:
        normalized["model_memory_id"] = model_memory_id
    normalized["source_conversation_id"] = source_conversation_id
    normalized["source_case_id"] = case_id
    normalized["product_category"] = case_metadata.get("product_category")
    normalized["product_subcategory"] = case_metadata.get("product_subcategory")
    normalized["issue_type"] = case_metadata.get("issue_type")
    normalized["issue_severity"] = case_metadata.get("issue_severity")
    normalized["memory_id"] = f"{memory_id_prefix}_{case_id}_{ordinal:02d}"
    normalized.setdefault("status", "pending")
    normalized.setdefault("retrieval_tags", [])
    normalized["expires_on"] = expiry_label_to_date(
        normalized.get("expires_on", normalized.get("expiry")),
        run_date=run_date,
        default_review_days=default_review_days,
    )
    normalized.pop("expiry", None)
    return normalized


def generate_for_case(
    client: Any,
    model: str,
    temperature: float,
    prompt_template: str,
    case: dict[str, Any],
    memory_id_prefix: str,
    run_date: date,
    default_review_days: int,
    short_review_days: int,
) -> dict[str, Any]:
    prompt = build_prompt(
        template=prompt_template,
        case=case,
        run_date=run_date,
        default_review_days=default_review_days,
        short_review_days=short_review_days,
    )
    payload, usage = call_json_model(
        client=client,
        model=model,
        prompt=prompt,
        temperature=temperature,
    )
    candidates = payload.get("memory_candidates", [])
    normalized_candidates = [
        normalize_candidate(
            candidate=candidate,
            source_conversation_id=case["source_conversation_id"],
            case_id=case["case_id"],
            case_metadata=case["metadata"],
            ordinal=index + 1,
            memory_id_prefix=memory_id_prefix,
            run_date=run_date,
            default_review_days=default_review_days,
        )
        for index, candidate in enumerate(candidates)
    ]
    return {
        "source_case_id": case["case_id"],
        "source_conversation_id": case["source_conversation_id"],
        "candidate_count": len(normalized_candidates),
        "memory_candidates": normalized_candidates,
        "usage": usage,
    }


def summarize_usage(results: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for result in results:
        for key, value in result.get("usage", {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def generate_memory_candidates(
    config: dict[str, Any],
    model_config: dict[str, Any],
    prompt_path: pathlib.Path | None = None,
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    client = openai_client_from_config(model_config)

    input_path = input_path or pathlib.Path(config["dataset"]["experience_cases_file"])
    output_path = output_path or pathlib.Path(config["outputs"]["morag_candidates_file"])
    prompt_path = prompt_path or pathlib.Path(config["prompts"]["memory_extractor"])
    model = model_config["models"]["memory_extractor_model"]
    memory_id_prefix = config["morag"]["memory_id_prefix"]
    default_review_days = int(config["morag"]["default_review_days"])
    short_review_days = int(config["morag"]["short_review_days"])
    temperature = float(model_config["generation"]["temperature"])
    prompt_template = load_prompt(prompt_path)
    run_date = date.today()

    cases = load_json(input_path)
    if limit is not None:
        cases = cases[:limit]

    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] extracting memories for {case['case_id']}")
        results.append(
            generate_for_case(
                client=client,
                model=model,
                temperature=temperature,
                prompt_template=prompt_template,
                case=case,
                memory_id_prefix=memory_id_prefix,
                run_date=run_date,
                default_review_days=default_review_days,
                short_review_days=short_review_days,
            )
        )

    output = {
        "run": run_metadata(config),
        "model": model,
        "source_file": str(input_path),
        "run_date": run_date.isoformat(),
        "default_review_days": default_review_days,
        "short_review_days": short_review_days,
        "case_count": len(results),
        "total_candidate_count": sum(result["candidate_count"] for result in results),
        "usage_totals": summarize_usage(results),
        "results": results,
    }
    write_json(output_path, output)

    print(f"Wrote {output['total_candidate_count']} candidates from {len(results)} cases to {output_path}")
    print(f"Usage totals: {output['usage_totals']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--prompt", type=pathlib.Path)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    generate_memory_candidates(
        config=config,
        model_config=model_config,
        prompt_path=args.prompt,
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
