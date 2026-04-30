r"""Run the moRAG memory-build pipeline.

This is a convenience orchestrator for the full memory path:

    generate candidates -> safety review -> deduplicate -> build moRAG store

Run from the code/ directory:

    python .\scripts\build_morag_pipeline.py --force-backend faiss
"""

from __future__ import annotations

import argparse

from scripts.common.cli_config import add_config_args, load_experiment_and_model_configs
from scripts.memory.deduplicate_approved_memories import deduplicate_approved_memories
from scripts.memory.generate_memory_candidates import generate_memory_candidates
from scripts.memory.review_memory_candidates import review_memory_candidates
from scripts.stores.build_store import build_records_for_store
from scripts.stores.build_vector_index import build_store_vector_index, output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--review-batch-size", type=int)
    parser.add_argument("--force-backend", choices=["auto", "tfidf", "faiss"], default="auto")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Start from the configured memory candidates file.",
    )
    parser.add_argument(
        "--skip-review",
        action="store_true",
        help="Start from the configured approved memories file.",
    )
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Start from the configured deduped memories file.",
    )
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )

    if not args.skip_generation:
        generate_memory_candidates(
            config=config,
            model_config=model_config,
            limit=args.limit,
        )

    if not args.skip_review:
        review_memory_candidates(
            config=config,
            model_config=model_config,
            batch_size=args.review_batch_size,
        )

    if not args.skip_dedup:
        deduplicate_approved_memories(
            config=config,
            model_config=model_config,
        )

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
        force_backend=args.force_backend,
    )
    print(f"Built moRAG vector index for {metadata['record_count']} records")
    print(f"Backend: {metadata['backend']}")
    print(f"Wrote metadata to {output_path(config['outputs'], 'morag', 'metadata')}")


if __name__ == "__main__":
    main()
