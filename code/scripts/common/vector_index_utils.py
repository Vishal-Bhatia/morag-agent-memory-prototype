"""Shared vector-index helpers for raw RAG and moRAG stores."""

from __future__ import annotations

import os
import pathlib
import pickle
from typing import Any

import numpy as np

from scripts.common.morag_utils import has_module, write_json


def should_use_faiss(force_backend: str) -> bool:
    use_faiss = (
        force_backend in {"auto", "faiss"}
        and has_module("sentence_transformers")
        and has_module("faiss")
    )
    if force_backend == "faiss" and not use_faiss:
        raise RuntimeError("FAISS backend requested, but sentence_transformers/faiss is unavailable")
    return use_faiss


def load_embedding_model(model_name: str) -> Any:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=True)


def build_tfidf_index(
    records: list[dict[str, Any]],
    embeddings_path: pathlib.Path,
    vectorizer_path: pathlib.Path,
    index_path: pathlib.Path,
    vector_config: dict[str, Any],
) -> dict[str, Any]:
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    texts = [record["search_text"] for record in records]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(
            int(vector_config["tfidf_ngram_min"]),
            int(vector_config["tfidf_ngram_max"]),
        ),
        max_features=int(vector_config["tfidf_max_features"]),
    )
    embeddings = vectorizer.fit_transform(texts)
    index = NearestNeighbors(metric="cosine", algorithm="brute")
    index.fit(embeddings)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(embeddings_path, embeddings)
    with vectorizer_path.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
    with index_path.open("wb") as index_file:
        pickle.dump(index, index_file)

    return {
        "backend": vector_config["tfidf_backend_name"],
        "embedding_shape": list(embeddings.shape),
        "metric": "cosine",
        "vectorizer_file": str(vectorizer_path),
        "vector_index_file": str(index_path),
    }


def build_faiss_index(
    records: list[dict[str, Any]],
    embeddings_path: pathlib.Path,
    index_path: pathlib.Path,
    model_name: str,
    vector_config: dict[str, Any],
) -> dict[str, Any]:
    import faiss

    texts = [record["search_text"] for record in records]
    model = load_embedding_model(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    faiss.write_index(index, str(index_path))

    return {
        "backend": vector_config["faiss_backend_name"],
        "embedding_model": model_name,
        "embedding_shape": list(embeddings.shape),
        "metric": "inner_product_on_normalized_vectors",
        "vector_index_file": str(index_path),
    }


def build_vector_index(
    records: list[dict[str, Any]],
    records_path: pathlib.Path,
    embeddings_path: pathlib.Path,
    vectorizer_path: pathlib.Path,
    index_path: pathlib.Path,
    metadata_path: pathlib.Path,
    model_name: str,
    vector_config: dict[str, Any],
    force_backend: str,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if should_use_faiss(force_backend):
        backend_metadata = build_faiss_index(
            records=records,
            embeddings_path=embeddings_path.with_suffix(".npy"),
            index_path=index_path.with_suffix(".faiss"),
            model_name=model_name,
            vector_config=vector_config,
        )
    else:
        backend_metadata = build_tfidf_index(
            records=records,
            embeddings_path=embeddings_path,
            vectorizer_path=vectorizer_path,
            index_path=index_path,
            vector_config=vector_config,
        )

    metadata = {
        "run": run_metadata,
        "records_file": str(records_path),
        "record_count": len(records),
        "record_ids": [record["record_id"] for record in records],
        "vector_config": vector_config,
        **backend_metadata,
    }
    write_json(metadata_path, metadata)
    return metadata


def load_faiss_backend(vector_metadata: dict[str, Any], embedding_model: str) -> tuple[Any, Any]:
    import faiss

    index = faiss.read_index(str(pathlib.Path(vector_metadata["vector_index_file"])))
    model = load_embedding_model(embedding_model)
    return index, model


def retrieve_with_faiss(
    query: str,
    records: list[dict[str, Any]],
    index: Any,
    model: Any,
    top_k: int,
    formatter: Any,
) -> list[dict[str, Any]]:
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.asarray(query_embedding, dtype="float32")
    scores, indices = index.search(query_embedding, top_k)

    return [
        formatter(
            record=records[int(record_index)],
            rank=rank + 1,
            score=float(scores[0][rank]),
        )
        for rank, record_index in enumerate(indices[0])
        if int(record_index) >= 0
    ]


def retrieve_with_tfidf(
    query: str,
    records: list[dict[str, Any]],
    vectorizer_path: pathlib.Path,
    index_path: pathlib.Path,
    top_k: int,
    formatter: Any,
) -> list[dict[str, Any]]:
    with vectorizer_path.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    with index_path.open("rb") as index_file:
        index = pickle.load(index_file)

    query_vector = vectorizer.transform([query])
    distances, indices = index.kneighbors(query_vector, n_neighbors=top_k)

    return [
        formatter(
            record=records[int(record_index)],
            rank=rank + 1,
            score=1.0 - float(distances[0][rank]),
        )
        for rank, record_index in enumerate(indices[0])
    ]


def retrieve_records(
    query: str,
    records: list[dict[str, Any]],
    vector_metadata: dict[str, Any],
    vector_config: dict[str, Any],
    embedding_model: str,
    top_k: int,
    formatter: Any,
    faiss_backend: tuple[Any, Any] | None = None,
) -> list[dict[str, Any]]:
    backend = vector_metadata["backend"]
    if backend == vector_config["faiss_backend_name"]:
        if faiss_backend is None:
            faiss_backend = load_faiss_backend(vector_metadata, embedding_model)
        index, model = faiss_backend
        return retrieve_with_faiss(query, records, index, model, top_k, formatter)

    if backend == vector_config["tfidf_backend_name"]:
        return retrieve_with_tfidf(
            query=query,
            records=records,
            vectorizer_path=pathlib.Path(vector_metadata["vectorizer_file"]),
            index_path=pathlib.Path(vector_metadata["vector_index_file"]),
            top_k=top_k,
            formatter=formatter,
        )

    raise ValueError(f"Unsupported retrieval backend: {backend}")
