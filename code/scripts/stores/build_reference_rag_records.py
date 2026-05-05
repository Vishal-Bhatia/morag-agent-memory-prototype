"""Build reference RAG records from policy-like guidance documents."""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from scripts.common.cli_config import add_config_args, load_experiment_config
from scripts.common.morag_utils import load_json, write_json


def bullet_section(title: str, values: list[str]) -> str:
    if not values:
        return ""
    return title + ":\n" + "\n".join(f"- {value}" for value in values)


def build_guidance_text(document: dict[str, Any]) -> str:
    sections = [
        f"Title: {document.get('document_title')}",
        f"Product category: {document.get('product_category')}",
        f"Product subcategory: {document.get('product_subcategory')}",
        f"Scope: {document.get('scope')}",
        bullet_section("Common customer needs", document.get("common_customer_needs", [])),
        bullet_section("Recommended response policy", document.get("recommended_response_policy", [])),
        bullet_section("Required clarifications", document.get("required_clarifications", [])),
        bullet_section("Do not assume", document.get("do_not_assume", [])),
        bullet_section("Safety or escalation triggers", document.get("safety_or_escalation_triggers", [])),
        bullet_section("Tone guidance", document.get("tone_guidance", [])),
    ]
    return "\n\n".join(section for section in sections if section)


def build_record(
    document: dict[str, Any],
    retrieval_unit: str,
) -> dict[str, Any]:
    guidance_text = build_guidance_text(document)
    return {
        "record_id": document["document_id"],
        "source_document_id": document["document_id"],
        "retrieval_unit": retrieval_unit,
        "search_text": guidance_text,
        "display_text": guidance_text,
        "metadata": {
            "product_category": document.get("product_category"),
            "product_subcategory": document.get("product_subcategory"),
            "source_case_count": document.get("source_case_count"),
            "confidence": document.get("confidence"),
            "source_case_ids": document.get("source_case_ids", []),
        },
    }


def build_reference_rag_records(
    config: dict[str, Any],
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    input_path = input_path or pathlib.Path(config["outputs"]["reference_guidance_file"])
    output_path = output_path or pathlib.Path(config["outputs"]["reference_rag_records_file"])
    retrieval_unit = config["retrieval"]["reference_rag_unit"]

    guidance_payload = load_json(input_path)
    documents = guidance_payload.get("documents", guidance_payload)
    records = [
        build_record(document=document, retrieval_unit=retrieval_unit)
        for document in documents
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
    output_path = args.output or pathlib.Path(config["outputs"]["reference_rag_records_file"])
    records = build_reference_rag_records(
        config=config,
        input_path=args.input,
        output_path=output_path,
    )
    print(f"Wrote {len(records)} reference RAG records to {output_path}")


if __name__ == "__main__":
    main()
