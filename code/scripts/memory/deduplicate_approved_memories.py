r"""Deduplicate safety-reviewed moRAG memories before indexing.

Run from the code/ directory:

    python .\scripts\deduplicate_approved_memories.py
"""

from __future__ import annotations

import argparse
import pathlib
import re
import string
from collections import defaultdict
from typing import Any

import numpy as np

from scripts.common.cli_config import (
    add_config_args,
    load_experiment_and_model_configs,
    run_metadata,
)
from scripts.common.morag_utils import load_json, write_json
from scripts.common.vector_index_utils import load_embedding_model


def normalize_note(note: str) -> str:
    translation = str.maketrans("", "", string.punctuation)
    normalized = note.casefold().translate(translation)
    return re.sub(r"\s+", " ", normalized).strip()


def approved_memories(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        memory
        for memory in payload["memories"]
        if memory.get("status") == "approved" and memory.get("note")
    ]


def scope_key(memory: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(memory.get(field) for field in fields)


def confidence_rank(memory: dict[str, Any]) -> int:
    ranks = {"high": 3, "medium": 2, "low": 1}
    return ranks.get(str(memory.get("confidence", "")).casefold(), 0)


def choose_canonical(memories: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        memories,
        key=lambda memory: (
            confidence_rank(memory),
            len(memory.get("evidence_excerpt", "")),
            -len(memory.get("note", "")),
        ),
        reverse=True,
    )[0]


def merge_group(group: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    canonical = dict(choose_canonical(group))
    duplicate_ids = [
        memory["memory_id"]
        for memory in group
        if memory["memory_id"] != canonical["memory_id"]
    ]
    source_conversation_ids = sorted(
        {
            memory.get("source_conversation_id")
            for memory in group
            if memory.get("source_conversation_id")
        }
    )
    source_case_ids = sorted(
        {memory.get("source_case_id") for memory in group if memory.get("source_case_id")}
    )
    retrieval_tags = sorted(
        {
            tag
            for memory in group
            for tag in memory.get("retrieval_tags", [])
            if isinstance(tag, str)
        }
    )

    canonical["dedup_status"] = "canonical"
    canonical["dedup_reason"] = reason
    canonical["duplicate_memory_ids"] = duplicate_ids
    canonical["duplicate_count"] = len(duplicate_ids)
    canonical["source_conversation_ids"] = source_conversation_ids
    canonical["source_case_ids"] = source_case_ids
    canonical["evidence_count"] = len(source_conversation_ids)
    canonical["retrieval_tags"] = retrieval_tags
    return canonical


def exact_dedup_groups(memories: list[dict[str, Any]], scope_fields: list[str]) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for memory in memories:
        key = (*scope_key(memory, scope_fields), normalize_note(memory["note"]))
        groups[key].append(memory)
    return list(groups.values())


def semantic_groups(
    memories: list[dict[str, Any]],
    model_name: str,
    threshold: float,
    scope_fields: list[str],
) -> list[list[dict[str, Any]]]:
    if len(memories) <= 1:
        return [memories]

    grouped_by_scope: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for memory in memories:
        grouped_by_scope[scope_key(memory, scope_fields)].append(memory)

    model = load_embedding_model(model_name)
    output_groups: list[list[dict[str, Any]]] = []

    for scoped_memories in grouped_by_scope.values():
        if len(scoped_memories) <= 1:
            output_groups.append(scoped_memories)
            continue

        notes = [memory["note"] for memory in scoped_memories]
        embeddings = model.encode(notes, normalize_embeddings=True)
        embeddings = np.asarray(embeddings, dtype="float32")
        similarity = embeddings @ embeddings.T

        visited: set[int] = set()
        for start_index in range(len(scoped_memories)):
            if start_index in visited:
                continue
            stack = [start_index]
            group_indices: set[int] = set()

            while stack:
                current = stack.pop()
                if current in group_indices:
                    continue
                group_indices.add(current)
                neighbors = np.where(similarity[current] >= threshold)[0]
                for neighbor in neighbors:
                    if int(neighbor) not in group_indices:
                        stack.append(int(neighbor))

            visited.update(group_indices)
            output_groups.append([scoped_memories[index] for index in sorted(group_indices)])

    return output_groups


def deduplicate_approved_memories(
    config: dict[str, Any],
    model_config: dict[str, Any],
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    input_path = input_path or pathlib.Path(config["outputs"]["approved_memories_file"])
    output_path = output_path or pathlib.Path(config["outputs"]["deduped_memories_file"])
    threshold = float(config["morag"]["dedup_similarity_threshold"])
    scope_fields = list(config["morag"]["dedup_scope_fields"])
    embedding_model = model_config["embeddings"]["model"]

    payload = load_json(input_path)
    memories = approved_memories(payload)

    exact_groups = exact_dedup_groups(memories, scope_fields)
    exact_canonical = [
        merge_group(group, reason="exact_normalized_note")
        if len(group) > 1
        else {**group[0], "dedup_status": "canonical", "duplicate_memory_ids": [], "duplicate_count": 0}
        for group in exact_groups
    ]

    semantic_grouped = semantic_groups(
        memories=exact_canonical,
        model_name=embedding_model,
        threshold=threshold,
        scope_fields=scope_fields,
    )
    deduped = [
        merge_group(group, reason=f"semantic_similarity_gte_{threshold}")
        if len(group) > 1
        else group[0]
        for group in semantic_grouped
    ]

    output = {
        "run": run_metadata(config),
        "source_file": str(input_path),
        "input_memory_count": len(memories),
        "deduped_memory_count": len(deduped),
        "removed_duplicate_count": len(memories) - len(deduped),
        "dedup_similarity_threshold": threshold,
        "dedup_scope_fields": scope_fields,
        "embedding_model": embedding_model,
        "memories": deduped,
    }
    write_json(output_path, output)

    print(f"Input approved memories: {len(memories)}")
    print(f"Deduped memories: {len(deduped)}")
    print(f"Removed duplicates: {len(memories) - len(deduped)}")
    print(f"Wrote deduped memories to {output_path}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    deduplicate_approved_memories(
        config=config,
        model_config=model_config,
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
