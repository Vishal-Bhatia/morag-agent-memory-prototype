"""Generate policy-like reference RAG guidance documents.

Reference guidance is built from experience + reference-background cases and is
intended to resemble centralized support documentation, not past-conversation
summaries.

Run from the code/ directory:

    python -m scripts.stores.generate_reference_guidance --limit-pairs 3
"""

from __future__ import annotations

import argparse
import collections
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


def guidance_scope(pair: tuple[str, str], cases: list[dict[str, Any]]) -> dict[str, Any]:
    category, subcategory = pair
    return {
        "product_category": category,
        "product_subcategory": subcategory,
        "case_count": len(cases),
    }


def source_case_summary(case: dict[str, Any]) -> dict[str, Any]:
    metadata = case["metadata"]
    return {
        "case_id": case["case_id"],
        "product_name": metadata.get("product_name"),
        "issue_type": metadata.get("issue_type"),
        "issue_severity": metadata.get("issue_severity"),
        "issue_description": metadata.get("issue_description"),
        "resolution_path_description": metadata.get("resolution_path_description"),
        "outcome": metadata.get("outcome"),
        "customer_sentiment": metadata.get("customer_sentiment"),
        "customer_satisfaction": metadata.get("customer_satisfaction"),
    }


def group_cases_by_pair(
    cases: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for case in cases:
        metadata = case["metadata"]
        pair = (
            str(metadata.get("product_category")),
            str(metadata.get("product_subcategory")),
        )
        grouped[pair].append(case)
    return dict(sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])))


def build_prompt(
    template: str,
    pair: tuple[str, str],
    cases: list[dict[str, Any]],
    max_cases_per_document: int,
) -> str:
    source_cases = [source_case_summary(case) for case in cases[:max_cases_per_document]]
    return render_template(
        template,
        {
            "scope": json.dumps(guidance_scope(pair, cases), indent=2),
            "source_cases": json.dumps(source_cases, indent=2),
        },
    )


def normalize_document(
    payload: dict[str, Any],
    pair: tuple[str, str],
    cases: list[dict[str, Any]],
    ordinal: int,
    run_id_prefix: str,
) -> dict[str, Any]:
    category, subcategory = pair
    source_case_ids = [case["case_id"] for case in cases]
    document = dict(payload)
    document["document_id"] = f"{run_id_prefix}_{ordinal:04d}"
    document["product_category"] = category
    document["product_subcategory"] = subcategory
    document["source_case_count"] = len(cases)
    document["source_case_ids"] = document.get("source_case_ids") or source_case_ids
    document.setdefault("confidence", "medium")
    return document


def summarize_usage(results: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for result in results:
        for key, value in result.get("usage", {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def generate_reference_guidance(
    config: dict[str, Any],
    model_config: dict[str, Any],
    prompt_path: pathlib.Path | None = None,
    experience_input_path: pathlib.Path | None = None,
    reference_input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
    limit_pairs: int | None = None,
    max_cases_per_document: int | None = None,
) -> dict[str, Any]:
    client = openai_client_from_config(model_config)
    prompt_path = prompt_path or pathlib.Path(config["prompts"]["reference_guidance"])
    experience_input_path = experience_input_path or pathlib.Path(
        config["dataset"]["experience_cases_file"]
    )
    reference_input_path = reference_input_path or pathlib.Path(
        config["dataset"]["reference_background_cases_file"]
    )
    output_path = output_path or pathlib.Path(config["outputs"]["reference_guidance_file"])
    model = model_config["models"]["reference_guidance_model"]
    temperature = float(model_config["generation"]["temperature"])
    template = load_prompt(prompt_path)
    reference_pipeline_config = config.get("pipeline", {}).get("reference_layer", {})
    if limit_pairs is None:
        configured_limit = int(reference_pipeline_config.get("reference_guidance_limit_pairs", 0))
        limit_pairs = configured_limit or None
    max_cases_per_document = int(
        max_cases_per_document
        if max_cases_per_document is not None
        else reference_pipeline_config.get("reference_guidance_max_cases_per_document", 12)
    )

    cases = load_json(experience_input_path) + load_json(reference_input_path)
    grouped = group_cases_by_pair(cases)
    pairs = list(grouped)
    if limit_pairs is not None:
        pairs = pairs[:limit_pairs]

    results: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs, start=1):
        pair_cases = grouped[pair]
        print(f"[{index}/{len(pairs)}] generating reference guidance for {pair[0]} / {pair[1]} ({len(pair_cases)} cases)")
        prompt = build_prompt(
            template=template,
            pair=pair,
            cases=pair_cases,
            max_cases_per_document=max_cases_per_document,
        )
        payload, usage = call_json_model(
            client=client,
            model=model,
            prompt=prompt,
            temperature=temperature,
        )
        document = normalize_document(
            payload=payload,
            pair=pair,
            cases=pair_cases,
            ordinal=index,
            run_id_prefix=config["retrieval"]["reference_rag_record_id_prefix"],
        )
        documents.append(document)
        results.append({"pair": pair, "usage": usage, "document_id": document["document_id"]})

    output = {
        "run": run_metadata(config),
        "model": model,
        "source_files": [str(experience_input_path), str(reference_input_path)],
        "document_count": len(documents),
        "max_cases_per_document": max_cases_per_document,
        "usage_totals": summarize_usage(results),
        "documents": documents,
    }
    write_json(output_path, output)
    print(f"Wrote {len(documents)} reference guidance documents to {output_path}")
    print(f"Usage totals: {output['usage_totals']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--prompt", type=pathlib.Path)
    parser.add_argument("--experience-input", type=pathlib.Path)
    parser.add_argument("--reference-input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--max-cases-per-document", type=int, default=12)
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    generate_reference_guidance(
        config=config,
        model_config=model_config,
        prompt_path=args.prompt,
        experience_input_path=args.experience_input,
        reference_input_path=args.reference_input,
        output_path=args.output,
        limit_pairs=args.limit_pairs,
        max_cases_per_document=args.max_cases_per_document,
    )


if __name__ == "__main__":
    main()
