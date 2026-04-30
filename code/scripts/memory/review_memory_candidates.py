r"""Safety-review moRAG memory candidates.

Run from the code/ directory:

    python .\scripts\review_memory_candidates.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
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


def flatten_candidates(
    candidate_batches: list[dict[str, Any]],
    memory_id_prefix: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for batch in candidate_batches:
        source_case_id = batch["source_case_id"]
        for index, candidate in enumerate(batch["memory_candidates"], start=1):
            normalized = dict(candidate)
            model_memory_id = normalized.get("memory_id")
            if model_memory_id:
                normalized["model_memory_id"] = model_memory_id
            normalized["memory_id"] = f"{memory_id_prefix}_{source_case_id}_{index:02d}"
            normalized.setdefault("source_case_id", source_case_id)
            normalized.setdefault("source_conversation_id", batch["source_conversation_id"])
            candidates.append(normalized)
    return candidates


def build_prompt(template: str, candidates: list[dict[str, Any]]) -> str:
    return render_template(
        template,
        {
            "memory_candidates": json.dumps(
                {"memory_candidates": candidates},
                indent=2,
            )
        },
    )


def review_batch(
    client: Any,
    model: str,
    temperature: float,
    prompt_template: str,
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt = build_prompt(prompt_template, candidates)
    payload, usage = call_json_model(
        client=client,
        model=model,
        prompt=prompt,
        temperature=temperature,
    )
    return payload.get("reviewed_memories", []), usage


def merge_review(
    candidate: dict[str, Any],
    review: dict[str, Any],
    status_after_pass: str,
    status_after_fail: str,
) -> dict[str, Any]:
    decision = review.get("decision", "rejected")
    final_note = review.get("final_note", "")

    if decision in {"approved", "edited"}:
        status = status_after_pass
        indexed_note = final_note or candidate.get("note", "")
    else:
        status = status_after_fail
        indexed_note = ""

    return {
        **candidate,
        "review_decision": decision,
        "status": status,
        "original_note": candidate.get("note", ""),
        "note": indexed_note,
        "review_reason": review.get("reason", ""),
        "safety_flags": review.get("safety_flags", []),
    }


def chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def summarize_usage(usages: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def review_memory_candidates(
    config: dict[str, Any],
    model_config: dict[str, Any],
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    client = openai_client_from_config(model_config)

    input_path = input_path or pathlib.Path(config["outputs"]["morag_candidates_file"])
    output_path = output_path or pathlib.Path(config["outputs"]["approved_memories_file"])
    prompt_path = pathlib.Path(config["prompts"]["memory_safety_reviewer"])
    model = model_config["models"]["safety_reviewer_model"]
    temperature = float(model_config["generation"]["temperature"])
    batch_size = batch_size or int(config["morag"]["safety_review_batch_size"])
    memory_id_prefix = config["morag"]["memory_id_prefix"]

    candidate_payload = load_json(input_path)
    candidates = flatten_candidates(
        candidate_batches=candidate_payload["results"],
        memory_id_prefix=memory_id_prefix,
    )
    candidates_by_id = {candidate["memory_id"]: candidate for candidate in candidates}
    prompt_template = load_prompt(prompt_path)

    reviewed: list[dict[str, Any]] = []
    usages: list[dict[str, Any]] = []
    batches = chunked(candidates, batch_size)

    for index, batch in enumerate(batches, start=1):
        print(f"[{index}/{len(batches)}] reviewing {len(batch)} candidates")
        reviews, usage = review_batch(
            client=client,
            model=model,
            temperature=temperature,
            prompt_template=prompt_template,
            candidates=batch,
        )
        usages.append(usage)

        for review in reviews:
            memory_id = review.get("memory_id")
            if memory_id not in candidates_by_id:
                continue
            reviewed.append(
                merge_review(
                    candidate=candidates_by_id[memory_id],
                    review=review,
                    status_after_pass=config["morag"]["memory_status_after_pass"],
                    status_after_fail=config["morag"]["memory_status_after_fail"],
                )
            )

    reviewed_ids = {memory["memory_id"] for memory in reviewed}
    missing_ids = [
        candidate["memory_id"]
        for candidate in candidates
        if candidate["memory_id"] not in reviewed_ids
    ]

    output = {
        "run": run_metadata(config),
        "model": model,
        "source_file": str(input_path),
        "candidate_count": len(candidates),
        "reviewed_count": len(reviewed),
        "missing_review_ids": missing_ids,
        "usage_totals": summarize_usage(usages),
        "approved_or_edited_count": sum(
            1 for memory in reviewed if memory["review_decision"] in {"approved", "edited"}
        ),
        "rejected_count": sum(1 for memory in reviewed if memory["review_decision"] == "rejected"),
        "memories": reviewed,
    }
    write_json(output_path, output)

    print(f"Reviewed {len(reviewed)} / {len(candidates)} candidates")
    print(f"Approved/edited: {output['approved_or_edited_count']}")
    print(f"Rejected: {output['rejected_count']}")
    print(f"Usage totals: {output['usage_totals']}")
    print(f"Wrote reviewed memories to {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    review_memory_candidates(
        config=config,
        model_config=model_config,
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
