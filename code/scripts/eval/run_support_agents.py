r"""Run support-agent answers for no-memory, reference RAG, and moRAG.

Run from the code/ directory:

    python .\scripts\run_support_agents.py --limit 2
"""

from __future__ import annotations

import argparse
import time
import pathlib
from typing import Any

from scripts.common.cli_config import (
    add_config_args,
    load_experiment_and_model_configs,
    run_metadata,
)
from scripts.common.conversation_utils import render_turns
from scripts.common.morag_utils import (
    call_text_model,
    load_json,
    load_prompt,
    openai_client_from_config,
    render_template,
    write_json,
)
from scripts.common.vector_index_utils import load_faiss_backend, retrieve_records


def usage_totals(results: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for result in results:
        for key, value in result.get("usage", {}).items():
            if isinstance(value, int):
                totals[key] = totals.get(key, 0) + value
    return totals


def current_conversation_for_case(eval_case: dict[str, Any]) -> str:
    return render_turns([eval_case["future_query_turn"]])


def format_raw_retrieval(record: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "rank": rank,
        "score": score,
        "record_id": record["record_id"],
        "source_case_id": record["source_case_id"],
        "source_conversation_id": record["source_conversation_id"],
        "issue_type": metadata.get("issue_type"),
        "outcome": metadata.get("outcome"),
        "resolution_path_description": metadata.get("resolution_path_description"),
        "conversation": record["display_text"],
    }


def format_reference_retrieval(record: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "rank": rank,
        "score": score,
        "record_id": record["record_id"],
        "product_category": metadata.get("product_category"),
        "product_subcategory": metadata.get("product_subcategory"),
        "source_case_count": metadata.get("source_case_count"),
        "source_case_ids": metadata.get("source_case_ids", []),
        "guidance": record["display_text"],
    }


def format_morag_retrieval(record: dict[str, Any], rank: int, score: float) -> dict[str, Any]:
    metadata = record["metadata"]
    return {
        "rank": rank,
        "score": score,
        "record_id": record["record_id"],
        "source_case_id": record.get("source_case_id"),
        "source_conversation_id": record.get("source_conversation_id"),
        "memory_type": metadata.get("memory_type"),
        "confidence": metadata.get("confidence"),
        "safety_flags": metadata.get("safety_flags", []),
        "note": record["display_text"],
    }


def render_raw_context(retrieved_records: list[dict[str, Any]]) -> str:
    if not retrieved_records:
        return "None."

    blocks = []
    for item in retrieved_records:
        blocks.append(
            "\n".join(
                [
                    f"Retrieved raw conversation {item['rank']}",
                    f"score: {item['score']:.4f}",
                    f"record_id: {item['record_id']}",
                    f"issue_type: {item.get('issue_type')}",
                    f"outcome: {item.get('outcome')}",
                    f"resolution_path: {item.get('resolution_path_description')}",
                    "turns:",
                    item["conversation"],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def render_reference_context(retrieved_records: list[dict[str, Any]]) -> str:
    if not retrieved_records:
        return "None."

    blocks = []
    for item in retrieved_records:
        blocks.append(
            "\n".join(
                [
                    f"Retrieved reference guidance {item['rank']}",
                    f"score: {item['score']:.4f}",
                    f"record_id: {item['record_id']}",
                    f"product_category: {item.get('product_category')}",
                    f"product_subcategory: {item.get('product_subcategory')}",
                    f"source_case_count: {item.get('source_case_count')}",
                    "guidance:",
                    item["guidance"],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def render_morag_context(retrieved_memories: list[dict[str, Any]]) -> str:
    if not retrieved_memories:
        return "None."

    blocks = []
    for item in retrieved_memories:
        blocks.append(
            "\n".join(
                [
                    f"Approved memory {item['rank']}",
                    f"score: {item['score']:.4f}",
                    f"record_id: {item['record_id']}",
                    f"memory_type: {item.get('memory_type')}",
                    f"confidence: {item.get('confidence')}",
                    f"safety_flags: {item.get('safety_flags', [])}",
                    f"note: {item['note']}",
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_prompt(
    base_prompt: str,
    method_prompt: str,
    current_conversation: str,
    retrieved_context: str,
) -> str:
    return render_template(
        "\n\n".join([base_prompt, method_prompt]),
        {
            "current_conversation": current_conversation,
            "future_query": current_conversation,
            "retrieved_context": retrieved_context,
        },
    )


def retrieve_for_method(
    method: str,
    query: str,
    stores: dict[str, Any],
    model_config: dict[str, Any],
    top_k: int,
) -> tuple[str, list[dict[str, Any]]]:
    if method == "no_memory":
        return "None.", []

    if method == "raw_rag":
        retrieved = retrieve_records(
            query=query,
            records=stores["raw_records"],
            vector_metadata=stores["raw_metadata"],
            vector_config=stores["raw_metadata"]["vector_config"],
            embedding_model=model_config["embeddings"]["model"],
            top_k=top_k,
            formatter=format_raw_retrieval,
            faiss_backend=stores.get("raw_faiss_backend"),
        )
        return render_raw_context(retrieved), retrieved

    if method == "reference_rag":
        retrieved = retrieve_records(
            query=query,
            records=stores["reference_records"],
            vector_metadata=stores["reference_metadata"],
            vector_config=stores["reference_metadata"]["vector_config"],
            embedding_model=model_config["embeddings"]["model"],
            top_k=top_k,
            formatter=format_reference_retrieval,
            faiss_backend=stores.get("reference_faiss_backend"),
        )
        return render_reference_context(retrieved), retrieved

    if method == "morag":
        retrieved = retrieve_records(
            query=query,
            records=stores["morag_records"],
            vector_metadata=stores["morag_metadata"],
            vector_config=stores["morag_metadata"]["vector_config"],
            embedding_model=model_config["embeddings"]["model"],
            top_k=top_k,
            formatter=format_morag_retrieval,
            faiss_backend=stores.get("morag_faiss_backend"),
        )
        return render_morag_context(retrieved), retrieved

    raise ValueError(f"Unsupported method: {method}")


def load_retrieval_stores(
    config: dict[str, Any],
    model_config: dict[str, Any],
    methods: list[str],
) -> dict[str, Any]:
    outputs = config["outputs"]
    vector_config = config["vector_index"]
    stores: dict[str, Any] = {}

    if "raw_rag" in methods:
        stores["raw_records"] = load_json(pathlib.Path(outputs["raw_rag_records_file"]))
        stores["raw_metadata"] = load_json(pathlib.Path(outputs["raw_rag_vector_metadata_file"]))
        if stores["raw_metadata"]["backend"] == vector_config["faiss_backend_name"]:
            stores["raw_faiss_backend"] = load_faiss_backend(
                vector_metadata=stores["raw_metadata"],
                embedding_model=model_config["embeddings"]["model"],
            )

    if "reference_rag" in methods:
        stores["reference_records"] = load_json(pathlib.Path(outputs["reference_rag_records_file"]))
        stores["reference_metadata"] = load_json(
            pathlib.Path(outputs["reference_rag_vector_metadata_file"])
        )
        if stores["reference_metadata"]["backend"] == vector_config["faiss_backend_name"]:
            stores["reference_faiss_backend"] = load_faiss_backend(
                vector_metadata=stores["reference_metadata"],
                embedding_model=model_config["embeddings"]["model"],
            )

    if "morag" in methods:
        stores["morag_records"] = load_json(pathlib.Path(outputs["morag_records_file"]))
        stores["morag_metadata"] = load_json(pathlib.Path(outputs["morag_vector_metadata_file"]))
        if stores["morag_metadata"]["backend"] == vector_config["faiss_backend_name"]:
            stores["morag_faiss_backend"] = load_faiss_backend(
                vector_metadata=stores["morag_metadata"],
                embedding_model=model_config["embeddings"]["model"],
            )

    return stores


def run_case_method(
    client: Any,
    eval_case: dict[str, Any],
    method: str,
    base_prompt: str,
    method_prompt: str,
    stores: dict[str, Any],
    config: dict[str, Any],
    model_config: dict[str, Any],
) -> dict[str, Any]:
    current_conversation = current_conversation_for_case(eval_case)
    retrieved_context, retrieved_items = retrieve_for_method(
        method=method,
        query=eval_case["future_query"],
        stores=stores,
        model_config=model_config,
        top_k=int(config["retrieval"]["top_k"]),
    )
    prompt = build_prompt(
        base_prompt=base_prompt,
        method_prompt=method_prompt,
        current_conversation=current_conversation,
        retrieved_context=retrieved_context,
    )

    started = time.perf_counter()
    answer, usage = call_text_model(
        client=client,
        model=model_config["models"]["answer_model"],
        prompt=prompt,
        temperature=float(model_config["generation"]["temperature"]),
    )
    elapsed_ms = round((time.perf_counter() - started) * 1000)

    return {
        "case_id": eval_case["case_id"],
        "method": method,
        "source_conversation_id": eval_case["source_conversation_id"],
        "source_row_idx": eval_case["source_row_idx"],
        "current_conversation": current_conversation,
        "future_query": eval_case["future_query"],
        "expected_resolution_pattern": eval_case["expected_resolution_pattern"],
        "reference_metadata": eval_case["metadata"],
        "retrieved_items": retrieved_items,
        "answer": answer,
        "elapsed_ms": elapsed_ms,
        "usage": usage,
    }


def parse_methods(value: str | None, configured_methods: list[str]) -> list[str]:
    if value is None:
        return configured_methods
    return [method.strip() for method in value.split(",") if method.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--methods")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    methods = parse_methods(args.methods, config["evaluation"]["methods"])
    output_path = args.output or pathlib.Path(config["outputs"]["run_results_file"])

    client = openai_client_from_config(model_config)
    eval_cases = load_json(pathlib.Path(config["dataset"]["eval_cases_file"]))
    if args.limit is not None:
        eval_cases = eval_cases[: args.limit]

    stores = load_retrieval_stores(config, model_config, methods)
    prompt_paths = config["prompts"]
    base_prompt = load_prompt(pathlib.Path(prompt_paths["base_support_agent"]))
    method_prompts = {
        "no_memory": load_prompt(pathlib.Path(prompt_paths["no_memory_agent"])),
        "raw_rag": load_prompt(pathlib.Path(prompt_paths["raw_rag_agent"])),
        "reference_rag": load_prompt(pathlib.Path(prompt_paths["reference_rag_agent"])),
        "morag": load_prompt(pathlib.Path(prompt_paths["morag_agent"])),
    }

    results: list[dict[str, Any]] = []
    total = len(eval_cases) * len(methods)
    completed = 0
    for eval_case in eval_cases:
        for method in methods:
            completed += 1
            print(f"[{completed}/{total}] {method} {eval_case['case_id']}")
            results.append(
                run_case_method(
                    client=client,
                    eval_case=eval_case,
                    method=method,
                    base_prompt=base_prompt,
                    method_prompt=method_prompts[method],
                    stores=stores,
                    config=config,
                    model_config=model_config,
                )
            )

    output = {
        "run": run_metadata(config),
        "answer_model": model_config["models"]["answer_model"],
        "methods": methods,
        "case_count": len(eval_cases),
        "result_count": len(results),
        "usage_totals": usage_totals(results),
        "results": results,
    }
    write_json(output_path, output)
    print(f"Wrote {len(results)} agent responses to {output_path}")
    print(f"Usage totals: {output['usage_totals']}")


if __name__ == "__main__":
    main()
