"""Pipeline 3: build the moRAG layer.

Run from the code/ directory:

    python -m scripts.pipelines.build_morag_layer
"""

from __future__ import annotations

import argparse

from scripts.common.cli_config import add_config_args, load_experiment_and_model_configs
from scripts.memory.deduplicate_approved_memories import deduplicate_approved_memories
from scripts.memory.generate_memory_candidates import generate_memory_candidates
from scripts.memory.review_memory_candidates import review_memory_candidates
from scripts.stores.build_store import build_records_for_store
from scripts.stores.build_vector_index import build_store_vector_index, output_path


def config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-batch-size", type=int)
    parser.add_argument("--force-backend", choices=["auto", "tfidf", "faiss"])
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-review", action="store_true")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--skip-store", action="store_true")
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    pipeline_config = config.get("pipeline", {})
    morag_config = pipeline_config.get("morag_layer", {})
    force_backend = args.force_backend or pipeline_config.get("default_vector_backend", "auto")
    configured_limit = int(morag_config.get("memory_candidate_limit", 0))
    limit = args.limit if args.limit is not None else configured_limit or None

    if (
        not args.skip_generation
        and config_bool(morag_config.get("generate_memory_candidates"), True)
    ):
        generate_memory_candidates(
            config=config,
            model_config=model_config,
            limit=limit,
        )

    if (
        not args.skip_review
        and config_bool(morag_config.get("review_memory_candidates"), True)
    ):
        review_memory_candidates(
            config=config,
            model_config=model_config,
            batch_size=args.review_batch_size,
        )

    if (
        not args.skip_dedup
        and config_bool(morag_config.get("deduplicate_memories"), True)
    ):
        deduplicate_approved_memories(
            config=config,
            model_config=model_config,
        )

    if not args.skip_store and config_bool(morag_config.get("build_morag_store"), True):
        records_path = output_path(config["outputs"], "morag", "records")
        records = build_records_for_store(
            config=config,
            store="morag",
            output_records_path=records_path,
        )
        print(f"Wrote {len(records)} moRAG records to {records_path}")
        metadata = build_store_vector_index(
            config=config,
            model_config=model_config,
            store="morag",
            records_path=records_path,
            force_backend=force_backend,
        )
        print(f"Built moRAG vector index for {metadata['record_count']} records")


if __name__ == "__main__":
    main()
