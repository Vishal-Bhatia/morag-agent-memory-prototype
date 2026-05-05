"""Pipeline 2: build the reference layer.

This creates the policy-like reference RAG and the separate past-conversation
fallback store.

Run from the code/ directory:

    python -m scripts.pipelines.build_reference_layer
"""

from __future__ import annotations

import argparse

from scripts.common.cli_config import add_config_args, load_experiment_and_model_configs
from scripts.stores.build_store import build_records_for_store
from scripts.stores.build_vector_index import build_store_vector_index, output_path
from scripts.stores.generate_reference_guidance import generate_reference_guidance


def config_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def build_and_index_store(
    config: dict,
    model_config: dict,
    store: str,
    force_backend: str,
) -> None:
    records_path = output_path(config["outputs"], store, "records")
    records = build_records_for_store(
        config=config,
        store=store,
        output_records_path=records_path,
    )
    print(f"Wrote {len(records)} {store} records to {records_path}")
    metadata = build_store_vector_index(
        config=config,
        model_config=model_config,
        store=store,
        records_path=records_path,
        force_backend=force_backend,
    )
    print(f"Built {store} vector index for {metadata['record_count']} records")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--force-backend", choices=["auto", "tfidf", "faiss"])
    parser.add_argument("--limit-pairs", type=int)
    parser.add_argument("--max-cases-per-document", type=int)
    parser.add_argument("--skip-guidance", action="store_true")
    parser.add_argument("--skip-reference-store", action="store_true")
    parser.add_argument("--skip-past-convo-store", action="store_true")
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    pipeline_config = config.get("pipeline", {})
    reference_config = pipeline_config.get("reference_layer", {})
    force_backend = args.force_backend or pipeline_config.get("default_vector_backend", "auto")

    if (
        not args.skip_guidance
        and config_bool(reference_config.get("generate_reference_guidance"), True)
    ):
        generate_reference_guidance(
            config=config,
            model_config=model_config,
            limit_pairs=args.limit_pairs,
            max_cases_per_document=args.max_cases_per_document,
        )

    if (
        not args.skip_reference_store
        and config_bool(reference_config.get("build_reference_rag_store"), True)
    ):
        build_and_index_store(
            config=config,
            model_config=model_config,
            store="reference_rag",
            force_backend=force_backend,
        )

    if (
        not args.skip_past_convo_store
        and config_bool(reference_config.get("build_past_conversation_fallback_store"), True)
    ):
        build_and_index_store(
            config=config,
            model_config=model_config,
            store="raw_rag",
            force_backend=force_backend,
        )


if __name__ == "__main__":
    main()
