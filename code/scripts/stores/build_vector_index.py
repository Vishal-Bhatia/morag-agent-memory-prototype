r"""Build local embeddings and a vector index for a configured record store.

The preferred path is SentenceTransformers + FAISS when both are installed.
The fallback path uses scikit-learn TF-IDF vectors + NearestNeighbors.

Run from the code/ directory:

    python .\scripts\build_vector_index.py --store raw_rag --force-backend faiss
    python .\scripts\build_vector_index.py --store morag --force-backend faiss
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
from scripts.common.morag_utils import load_json
from scripts.common.vector_index_utils import build_vector_index


STORE_OUTPUT_KEYS = {
    "reference_rag": {
        "label": "reference RAG",
        "records": "reference_rag_records_file",
        "embeddings": "reference_rag_embeddings_file",
        "vectorizer": "reference_rag_vectorizer_file",
        "index": "reference_rag_vector_index_file",
        "metadata": "reference_rag_vector_metadata_file",
    },
    "raw_rag": {
        "label": "past-conversation fallback RAG",
        "records": "raw_rag_records_file",
        "embeddings": "raw_rag_embeddings_file",
        "vectorizer": "raw_rag_vectorizer_file",
        "index": "raw_rag_vector_index_file",
        "metadata": "raw_rag_vector_metadata_file",
    },
    "morag": {
        "label": "moRAG",
        "records": "morag_records_file",
        "embeddings": "morag_embeddings_file",
        "vectorizer": "morag_vectorizer_file",
        "index": "morag_vector_index_file",
        "metadata": "morag_vector_metadata_file",
    },
}


def output_path(outputs: dict[str, Any], store: str, key: str) -> pathlib.Path:
    output_key = STORE_OUTPUT_KEYS[store][key]
    return pathlib.Path(outputs[output_key])


def build_store_vector_index(
    config: dict[str, Any],
    model_config: dict[str, Any],
    store: str,
    records_path: pathlib.Path | None = None,
    force_backend: str = "auto",
) -> dict[str, Any]:
    outputs = config["outputs"]
    records_path = records_path or output_path(outputs, store, "records")
    metadata_path = output_path(outputs, store, "metadata")
    records = load_json(records_path)

    return build_vector_index(
        records=records,
        records_path=records_path,
        embeddings_path=output_path(outputs, store, "embeddings"),
        vectorizer_path=output_path(outputs, store, "vectorizer"),
        index_path=output_path(outputs, store, "index"),
        metadata_path=metadata_path,
        model_name=model_config["embeddings"]["model"],
        vector_config=config["vector_index"],
        force_backend=force_backend,
        run_metadata=run_metadata(config),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=sorted(STORE_OUTPUT_KEYS), required=True)
    add_config_args(parser, include_model_config=True)
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
    metadata = build_store_vector_index(
        config=config,
        model_config=model_config,
        store=args.store,
        records_path=records_path,
        force_backend=args.force_backend,
    )

    label = STORE_OUTPUT_KEYS[args.store]["label"]
    print(f"Built {label} vector index for {metadata['record_count']} records")
    print(f"Backend: {metadata['backend']}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
