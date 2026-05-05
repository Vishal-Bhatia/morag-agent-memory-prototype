r"""Preview retrieval for a configured record store.

Run from the code/ directory:

    python -m scripts.eval.preview_retrieval --store reference_rag
    python -m scripts.eval.preview_retrieval --store raw_rag
    python -m scripts.eval.preview_retrieval --store morag
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from scripts.common.cli_config import (
    add_config_args,
    load_experiment_and_model_configs,
    run_metadata,
)
from scripts.common.morag_utils import load_json, write_json
from scripts.common.vector_index_utils import load_faiss_backend, retrieve_records
from scripts.stores.build_vector_index import STORE_OUTPUT_KEYS, output_path as store_output_path


def format_raw_rag_record(record: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "rank": rank,
        "score": score,
        "record_id": record["record_id"],
        "source_case_id": record["source_case_id"],
        "source_conversation_id": record["source_conversation_id"],
        "source_row_idx": record["source_row_idx"],
        "issue_type": metadata.get("issue_type"),
        "issue_severity": metadata.get("issue_severity"),
        "outcome": metadata.get("outcome"),
        "resolution_path_description": metadata.get("resolution_path_description"),
    }


def format_reference_rag_record(record: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "rank": rank,
        "score": score,
        "record_id": record["record_id"],
        "product_category": metadata.get("product_category"),
        "product_subcategory": metadata.get("product_subcategory"),
        "source_case_count": metadata.get("source_case_count"),
        "guidance": record["display_text"],
    }


def format_morag_record(record: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "rank": rank,
        "score": score,
        "record_id": record["record_id"],
        "source_case_id": record.get("source_case_id"),
        "source_conversation_id": record.get("source_conversation_id"),
        "memory_type": metadata.get("memory_type"),
        "confidence": metadata.get("confidence"),
        "note": record["display_text"],
        "safety_flags": metadata.get("safety_flags", []),
    }


def format_retrieved_record(store: str) -> Any:
    if store == "reference_rag":
        return format_reference_rag_record
    if store == "raw_rag":
        return format_raw_rag_record
    if store == "morag":
        return format_morag_record
    raise ValueError(f"Unsupported store: {store}")


def preview_item_key(store: str) -> str:
    if store == "reference_rag":
        return "retrieved_guidance"
    if store == "raw_rag":
        return "retrieved_records"
    if store == "morag":
        return "retrieved_memories"
    raise ValueError(f"Unsupported store: {store}")


def preview_output_path(config: dict[str, Any], store: str, override: pathlib.Path | None) -> pathlib.Path:
    if override is not None:
        return override
    outputs = config["outputs"]
    if store == "reference_rag":
        return pathlib.Path(outputs["reference_rag_retrieval_preview_file"])
    if store == "raw_rag":
        return pathlib.Path(outputs["raw_rag_retrieval_preview_file"])
    if store == "morag":
        return pathlib.Path(outputs["morag_retrieval_preview_file"])
    raise ValueError(f"Unsupported store: {store}")


def preview_case(
    eval_case: dict[str, Any],
    store: str,
    records: list[dict[str, Any]],
    vector_metadata: dict[str, Any],
    model_config: dict[str, Any],
    top_k: int,
    faiss_backend: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    retrieved = retrieve_records(
        query=eval_case["future_query"],
        records=records,
        vector_metadata=vector_metadata,
        vector_config=vector_metadata["vector_config"],
        embedding_model=model_config["embeddings"]["model"],
        top_k=top_k,
        formatter=format_retrieved_record(store),
        faiss_backend=faiss_backend,
    )
    return {
        "eval_case_id": eval_case["case_id"],
        "source_conversation_id": eval_case["source_conversation_id"],
        "source_row_idx": eval_case["source_row_idx"],
        "future_query": eval_case["future_query"],
        "expected_resolution_pattern": eval_case["expected_resolution_pattern"],
        preview_item_key(store): retrieved,
    }


def print_preview(previews: list[dict[str, Any]], store: str, max_cases: int) -> None:
    item_key = preview_item_key(store)
    for item in previews[:max_cases]:
        print()
        print(f"{item['eval_case_id']}: {item['future_query']}")
        print(f"Expected: {item['expected_resolution_pattern']}")
        for record in item[item_key]:
            if store == "reference_rag":
                print(
                    f"  {record['rank']}. {record['record_id']} "
                    f"score={record['score']:.4f} "
                    f"pair={record['product_category']}/{record['product_subcategory']}"
                )
            elif store == "raw_rag":
                print(
                    f"  {record['rank']}. {record['record_id']} "
                    f"score={record['score']:.4f} "
                    f"issue={record['issue_type']}"
                )
            else:
                print(
                    f"  {record['rank']}. {record['record_id']} "
                    f"score={record['score']:.4f} "
                    f"type={record['memory_type']} "
                    f"note={record['note'][:120]}"
                )


def build_retrieval_preview(
    config: dict[str, Any],
    model_config: dict[str, Any],
    store: str,
    output_path: pathlib.Path | None = None,
    max_cases: int = 5,
) -> list[dict[str, Any]]:
    outputs = config["outputs"]
    output_path = preview_output_path(config, store, output_path)
    eval_cases = load_json(pathlib.Path(config["dataset"]["eval_cases_file"]))
    records = load_json(output_path_for_records(outputs, store))
    vector_metadata = load_json(store_output_path(outputs, store, "metadata"))
    top_k = int(config["retrieval"]["top_k"])

    faiss_backend = None
    if vector_metadata["backend"] == config["vector_index"]["faiss_backend_name"]:
        faiss_backend = load_faiss_backend(
            vector_metadata=vector_metadata,
            embedding_model=model_config["embeddings"]["model"],
        )

    previews = [
        preview_case(
            eval_case=eval_case,
            store=store,
            records=records,
            vector_metadata=vector_metadata,
            model_config=model_config,
            top_k=top_k,
            faiss_backend=faiss_backend,
        )
        for eval_case in eval_cases
    ]

    output = {
        "run": run_metadata(config),
        "store": store,
        "backend": vector_metadata["backend"],
        "preview_count": len(previews),
        "previews": previews,
    }
    write_json(output_path, output)
    print(f"Wrote {len(previews)} {STORE_OUTPUT_KEYS[store]['label']} retrieval previews to {output_path}")
    print(f"Backend: {vector_metadata['backend']}")
    print_preview(previews, store=store, max_cases=max_cases)
    return previews


def output_path_for_records(outputs: dict[str, Any], store: str) -> pathlib.Path:
    return store_output_path(outputs, store, "records")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=sorted(STORE_OUTPUT_KEYS), required=True)
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    build_retrieval_preview(
        config=config,
        model_config=model_config,
        store=args.store,
        output_path=args.output,
        max_cases=args.max_cases,
    )


if __name__ == "__main__":
    main()
