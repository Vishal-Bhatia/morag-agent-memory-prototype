r"""Build searchable moRAG memory records from deduped approved memories.

Run from the code/ directory:

    python .\scripts\build_morag_memory_records.py
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from scripts.common.cli_config import add_config_args, load_experiment_config
from scripts.common.morag_utils import load_json, write_json


def build_search_text(memory: dict[str, Any]) -> str:
    fields = [
        ("Product category", memory.get("product_category")),
        ("Product subcategory", memory.get("product_subcategory")),
        ("Issue type", memory.get("issue_type")),
        ("Memory type", memory.get("memory_type")),
        ("Note", memory.get("note")),
        ("Evidence", memory.get("evidence_excerpt")),
        ("Tags", ", ".join(memory.get("retrieval_tags", []))),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)


def build_record(memory: dict[str, Any], ordinal: int, retrieval_unit: str) -> dict[str, Any]:
    return {
        "record_id": memory["memory_id"],
        "source_memory_id": memory["memory_id"],
        "source_case_id": memory.get("source_case_id"),
        "source_conversation_id": memory.get("source_conversation_id"),
        "source_conversation_ids": memory.get("source_conversation_ids", []),
        "retrieval_unit": retrieval_unit,
        "search_text": build_search_text(memory),
        "display_text": memory["note"],
        "metadata": {
            "product_category": memory.get("product_category"),
            "product_subcategory": memory.get("product_subcategory"),
            "issue_type": memory.get("issue_type"),
            "issue_severity": memory.get("issue_severity"),
            "memory_type": memory.get("memory_type"),
            "confidence": memory.get("confidence"),
            "expires_on": memory.get("expires_on"),
            "review_decision": memory.get("review_decision"),
            "safety_flags": memory.get("safety_flags", []),
            "duplicate_count": memory.get("duplicate_count", 0),
            "ordinal": ordinal,
        },
    }


def build_morag_memory_records(
    config: dict[str, Any],
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    input_path = input_path or pathlib.Path(config["outputs"]["deduped_memories_file"])
    output_path = output_path or pathlib.Path(config["outputs"]["morag_records_file"])
    retrieval_unit = config["retrieval"]["morag_unit"]

    payload = load_json(input_path)
    memories = payload["memories"]
    records = [
        build_record(memory=memory, ordinal=index + 1, retrieval_unit=retrieval_unit)
        for index, memory in enumerate(memories)
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
    output_path = args.output or pathlib.Path(config["outputs"]["morag_records_file"])
    records = build_morag_memory_records(
        config=config,
        input_path=args.input,
        output_path=output_path,
    )
    print(f"Wrote {len(records)} moRAG memory records to {output_path}")


if __name__ == "__main__":
    main()
