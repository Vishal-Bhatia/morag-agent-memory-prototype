r"""Build records and vector index for one retrieval store.

This is the convenience entry point for stores that have a record-building step
followed by indexing. The lower-level record and vector scripts remain useful
for debugging each stage independently.

Run from the code/ directory:

    python -m scripts.stores.build_store --store reference_rag --force-backend faiss
    python -m scripts.stores.build_store --store raw_rag --force-backend faiss
    python -m scripts.stores.build_store --store morag --force-backend faiss
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from scripts.common.cli_config import add_config_args, load_experiment_and_model_configs
from scripts.stores.build_morag_memory_records import build_morag_memory_records
from scripts.stores.build_reference_rag_records import build_reference_rag_records
from scripts.stores.build_raw_rag_records import build_raw_rag_records
from scripts.stores.build_vector_index import STORE_OUTPUT_KEYS, build_store_vector_index, output_path


def build_records_for_store(
    config: dict[str, Any],
    store: str,
    input_path: pathlib.Path | None = None,
    output_records_path: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    if store == "reference_rag":
        return build_reference_rag_records(
            config=config,
            input_path=input_path,
            output_path=output_records_path,
        )
    if store == "raw_rag":
        return build_raw_rag_records(
            config=config,
            input_path=input_path,
            output_path=output_records_path,
        )
    if store == "morag":
        return build_morag_memory_records(
            config=config,
            input_path=input_path,
            output_path=output_records_path,
        )
    raise ValueError(f"Unsupported store: {store}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=sorted(STORE_OUTPUT_KEYS), required=True)
    add_config_args(parser, include_model_config=True)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--records", type=pathlib.Path)
    parser.add_argument("--force-backend", choices=["auto", "tfidf", "faiss"], default="auto")
    args = parser.parse_args()

    config, model_config = load_experiment_and_model_configs(
        args.config,
        args.model_config,
        args.dataset_config,
        args.run_id,
    )
    records_path = args.records or output_path(config["outputs"], args.store, "records")
    metadata_path = output_path(config["outputs"], args.store, "metadata")
    label = STORE_OUTPUT_KEYS[args.store]["label"]

    records = build_records_for_store(
        config=config,
        store=args.store,
        input_path=args.input,
        output_records_path=records_path,
    )
    print(f"Wrote {len(records)} {label} records to {records_path}")

    metadata = build_store_vector_index(
        config=config,
        model_config=model_config,
        store=args.store,
        records_path=records_path,
        force_backend=args.force_backend,
    )
    print(f"Built {label} vector index for {metadata['record_count']} records")
    print(f"Backend: {metadata['backend']}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
