r"""Build past-conversation fallback RAG records.

Run from the code/ directory:

    python -m scripts.stores.build_raw_rag_records
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from scripts.common.cli_config import add_config_args, load_experiment_config
from scripts.common.morag_utils import load_json, write_json


def format_metadata(metadata: dict[str, Any]) -> str:
    fields = [
        ("Product", metadata.get("product_name")),
        ("Product description", metadata.get("product_description")),
        ("Issue type", metadata.get("issue_type")),
        ("Issue severity", metadata.get("issue_severity")),
        ("Issue description", metadata.get("issue_description")),
        ("Resolution path", metadata.get("resolution_path_description")),
        ("Outcome", metadata.get("outcome")),
        ("Customer sentiment", metadata.get("customer_sentiment")),
        ("Customer satisfaction", metadata.get("customer_satisfaction")),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def build_search_text(case: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            format_metadata(case["metadata"]),
            "Conversation:",
            case["past_conversation_text"],
        ]
    )


def build_record(
    case: dict[str, Any],
    ordinal: int,
    record_id_prefix: str,
    retrieval_unit: str,
) -> dict[str, Any]:
    metadata = case["metadata"]
    return {
        "record_id": f"{record_id_prefix}_{ordinal:03d}",
        "source_case_id": case["case_id"],
        "source_conversation_id": case["source_conversation_id"],
        "source_row_idx": case["source_row_idx"],
        "retrieval_unit": retrieval_unit,
        "search_text": build_search_text(case),
        "display_text": case["past_conversation_text"],
        "metadata": {
            "product_name": metadata.get("product_name"),
            "product_category": metadata.get("product_category"),
            "product_subcategory": metadata.get("product_subcategory"),
            "issue_type": metadata.get("issue_type"),
            "issue_severity": metadata.get("issue_severity"),
            "outcome": metadata.get("outcome"),
            "resolution_path_description": metadata.get("resolution_path_description"),
        },
    }


def build_raw_rag_records(
    config: dict[str, Any],
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    output_path = output_path or pathlib.Path(config["outputs"]["raw_rag_records_file"])
    record_id_prefix = config["retrieval"]["raw_rag_record_id_prefix"]
    retrieval_unit = config["retrieval"]["raw_rag_unit"]

    if input_path is not None:
        cases = load_json(input_path)
    else:
        cases = load_json(pathlib.Path(config["dataset"]["experience_cases_file"]))
        reference_background_path = config["dataset"].get("reference_background_cases_file")
        if reference_background_path:
            cases += load_json(pathlib.Path(reference_background_path))

    records = [
        build_record(
            case=case,
            ordinal=index + 1,
            record_id_prefix=record_id_prefix,
            retrieval_unit=retrieval_unit,
        )
        for index, case in enumerate(cases)
    ]

    write_json(output_path, records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.dataset_config, args.run_id)
    output_path = args.output or pathlib.Path(config["outputs"]["raw_rag_records_file"])
    records = build_raw_rag_records(
        config=config,
        input_path=args.input,
        output_path=output_path,
    )
    print(f"Wrote {len(records)} raw RAG records to {output_path}")


if __name__ == "__main__":
    main()
