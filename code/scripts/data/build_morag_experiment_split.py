"""Build the larger annotated moRAG experiment split.

The split has three non-overlapping pools:

- experience: used to create moRAG and also available to reference RAG
- reference_background: available only to reference RAG
- eval: held out for measurement

Run from the code/ directory:

    python -m scripts.data.build_morag_experiment_split --run-id auto
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import random
from typing import Any

from scripts.common.cli_config import add_config_args, load_experiment_config, run_metadata
from scripts.common.conversation_utils import first_customer_turn, normalize_turns, render_turns
from scripts.common.morag_utils import load_json, write_json


def category_pair(record: dict[str, Any], category_field: str, subcategory_field: str) -> str:
    return f"{record.get(category_field)}::{record.get(subcategory_field)}"


def is_other_record(record: dict[str, Any], category_field: str, subcategory_field: str) -> bool:
    category = str(record.get(category_field, ""))
    subcategory = str(record.get(subcategory_field, ""))
    return category == "other_toys_and_related" or subcategory.startswith("other_")


def select_balanced_records(
    records: list[dict[str, Any]],
    count: int,
    category_field: str,
    subcategory_field: str,
    rng: random.Random,
    max_pair_share: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in records:
        grouped[category_pair(record, category_field, subcategory_field)].append(record)

    for rows in grouped.values():
        rng.shuffle(rows)

    sorted_pairs = sorted(grouped, key=lambda pair: len(grouped[pair]), reverse=True)
    max_per_pair = max(1, int(count * max_pair_share))
    selected: list[dict[str, Any]] = []

    while len(selected) < count:
        added_this_round = False
        for pair in sorted_pairs:
            rows = grouped[pair]
            already_selected_for_pair = sum(
                1 for item in selected if category_pair(item, category_field, subcategory_field) == pair
            )
            if rows and already_selected_for_pair < max_per_pair:
                selected.append(rows.pop())
                added_this_round = True
                if len(selected) == count:
                    break
        if not added_this_round:
            break

    if len(selected) < count:
        raise ValueError(f"Could only select {len(selected)} records; needed {count}")

    rng.shuffle(selected)
    return selected


def split_records(
    popular_records: list[dict[str, Any]],
    other_records: list[dict[str, Any]],
    experience_popular_count: int,
    reference_popular_count: int,
    eval_popular_count: int,
    experience_other_count: int,
    reference_other_count: int,
    eval_other_count: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng.shuffle(popular_records)
    rng.shuffle(other_records)

    experience = (
        popular_records[:experience_popular_count]
        + other_records[:experience_other_count]
    )
    reference_start = experience_popular_count
    reference_other_start = experience_other_count
    reference_background = (
        popular_records[reference_start : reference_start + reference_popular_count]
        + other_records[reference_other_start : reference_other_start + reference_other_count]
    )
    eval_start = reference_start + reference_popular_count
    eval_other_start = reference_other_start + reference_other_count
    eval_records = (
        popular_records[eval_start : eval_start + eval_popular_count]
        + other_records[eval_other_start : eval_other_start + eval_other_count]
    )

    rng.shuffle(experience)
    rng.shuffle(reference_background)
    rng.shuffle(eval_records)
    return experience, reference_background, eval_records


def base_metadata(
    record: dict[str, Any],
    product_category_field: str,
    product_subcategory_field: str,
) -> dict[str, Any]:
    return {
        "product_id": record.get("product_id"),
        "product_name": record.get("product_name"),
        "product_description": record.get("product_description"),
        "product_category": record.get(product_category_field),
        "product_subcategory": record.get(product_subcategory_field),
        "issue_type": record.get("issue_type"),
        "issue_severity": record.get("issue_severity"),
        "issue_description": record.get("issue_description"),
        "resolution_path_description": record.get("resolution_path_description"),
        "outcome": record.get("outcome"),
        "customer_sentiment": record.get("customer_sentiment"),
        "customer_satisfaction": record.get("customer_satisfaction"),
    }


def build_context_case(
    record: dict[str, Any],
    ordinal: int,
    case_id_prefix: str,
    split_name: str,
    past_conversation_field: str,
    conversation_config: dict[str, Any],
    product_category_field: str,
    product_subcategory_field: str,
) -> dict[str, Any]:
    past_turns = normalize_turns(record[past_conversation_field], conversation_config)
    return {
        "case_id": f"{case_id_prefix}_{ordinal:04d}",
        "source_conversation_id": record["conversation_id"],
        "source_row_idx": record["source_row_idx"],
        "split": split_name,
        "past_conversation_turns": past_turns,
        "past_conversation_text": render_turns(past_turns),
        "metadata": base_metadata(record, product_category_field, product_subcategory_field),
    }


def build_eval_case(
    record: dict[str, Any],
    ordinal: int,
    case_id_prefix: str,
    past_conversation_field: str,
    future_dialogue_field: str,
    conversation_config: dict[str, Any],
    product_category_field: str,
    product_subcategory_field: str,
) -> dict[str, Any]:
    past_turns = normalize_turns(record[past_conversation_field], conversation_config)
    future_turns = normalize_turns(record[future_dialogue_field], conversation_config)
    first_turn = first_customer_turn(future_turns)

    if first_turn is None:
        raise ValueError(f"No customer turn found for {record['conversation_id']}")

    return {
        "case_id": f"{case_id_prefix}_{ordinal:04d}",
        "source_conversation_id": record["conversation_id"],
        "source_row_idx": record["source_row_idx"],
        "split": "eval",
        "future_query": first_turn["text"],
        "future_query_turn": first_turn,
        "future_dialogue_turns_reference_only": future_turns,
        "past_conversation_turns_reference_only": past_turns,
        "expected_resolution_pattern": record.get("resolution_path_description"),
        "metadata": base_metadata(record, product_category_field, product_subcategory_field),
    }


def counts_by_pair(
    records: list[dict[str, Any]],
    category_field: str,
    subcategory_field: str,
) -> dict[str, int]:
    counter = collections.Counter(
        category_pair(record, category_field, subcategory_field) for record in records
    )
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def build_morag_experiment_split(
    config: dict[str, Any],
    input_path: pathlib.Path | None = None,
    sample_output: pathlib.Path | None = None,
    experience_output: pathlib.Path | None = None,
    reference_output: pathlib.Path | None = None,
    eval_output: pathlib.Path | None = None,
    metadata_output: pathlib.Path | None = None,
    seed: int | None = None,
    max_pair_share: float | None = None,
) -> dict[str, Any]:
    pipeline_config = config.get("pipeline", {}).get("experiment_data", {})
    seed = int(seed if seed is not None else pipeline_config.get("seed", 42))
    max_pair_share = float(
        max_pair_share
        if max_pair_share is not None
        else pipeline_config.get("max_pair_share", 0.15)
    )
    rng = random.Random(seed)
    dataset_config = config["dataset"]
    conversation_config = config.get("conversation", {})

    category_field = dataset_config["product_category_field"]
    subcategory_field = dataset_config["product_subcategory_field"]
    input_path = input_path or pathlib.Path(dataset_config["annotated_pool_file"])
    sample_output = sample_output or pathlib.Path(dataset_config["experiment_sample_file"])
    experience_output = experience_output or pathlib.Path(
        dataset_config["experience_cases_file"]
    )
    reference_output = reference_output or pathlib.Path(
        dataset_config["reference_background_cases_file"]
    )
    eval_output = eval_output or pathlib.Path(dataset_config["eval_cases_file"])
    metadata_output = metadata_output or pathlib.Path(
        dataset_config["split_metadata_file"]
    )

    experience_count = int(dataset_config["experience_case_count"])
    reference_count = int(dataset_config["reference_background_case_count"])
    eval_count = int(dataset_config["eval_case_count"])
    experience_popular_count = int(dataset_config["experience_popular_case_count"])
    reference_popular_count = int(dataset_config["reference_background_popular_case_count"])
    eval_popular_count = int(dataset_config["eval_popular_case_count"])
    experience_other_count = experience_count - experience_popular_count
    reference_other_count = reference_count - reference_popular_count
    eval_other_count = eval_count - eval_popular_count
    popular_count = experience_popular_count + reference_popular_count + eval_popular_count
    other_count = experience_other_count + reference_other_count + eval_other_count

    records = load_json(input_path)
    popular_pool = [
        record
        for record in records
        if not is_other_record(record, category_field, subcategory_field)
    ]
    other_pool = [
        record
        for record in records
        if is_other_record(record, category_field, subcategory_field)
    ]

    selected_popular = select_balanced_records(
        records=popular_pool,
        count=popular_count,
        category_field=category_field,
        subcategory_field=subcategory_field,
        rng=rng,
        max_pair_share=max_pair_share,
    )
    selected_other = select_balanced_records(
        records=other_pool,
        count=other_count,
        category_field=category_field,
        subcategory_field=subcategory_field,
        rng=rng,
        max_pair_share=max_pair_share,
    )

    experience_records, reference_records, eval_records = split_records(
        popular_records=selected_popular,
        other_records=selected_other,
        experience_popular_count=experience_popular_count,
        reference_popular_count=reference_popular_count,
        eval_popular_count=eval_popular_count,
        experience_other_count=experience_other_count,
        reference_other_count=reference_other_count,
        eval_other_count=eval_other_count,
        rng=rng,
    )
    experiment_sample = experience_records + reference_records + eval_records

    past_conversation_field = dataset_config["past_conversation_field"]
    future_dialogue_field = dataset_config["future_dialogue_field"]
    experience_case_id_prefix = dataset_config["experience_case_id_prefix"]
    reference_case_id_prefix = dataset_config["reference_background_case_id_prefix"]
    eval_case_id_prefix = dataset_config["eval_case_id_prefix"]

    experience_cases = [
        build_context_case(
            record=record,
            ordinal=index + 1,
            case_id_prefix=experience_case_id_prefix,
            split_name="experience",
            past_conversation_field=past_conversation_field,
            conversation_config=conversation_config,
            product_category_field=category_field,
            product_subcategory_field=subcategory_field,
        )
        for index, record in enumerate(experience_records)
    ]
    reference_cases = [
        build_context_case(
            record=record,
            ordinal=index + 1,
            case_id_prefix=reference_case_id_prefix,
            split_name="reference_background",
            past_conversation_field=past_conversation_field,
            conversation_config=conversation_config,
            product_category_field=category_field,
            product_subcategory_field=subcategory_field,
        )
        for index, record in enumerate(reference_records)
    ]
    eval_cases = [
        build_eval_case(
            record=record,
            ordinal=index + 1,
            case_id_prefix=eval_case_id_prefix,
            past_conversation_field=past_conversation_field,
            future_dialogue_field=future_dialogue_field,
            conversation_config=conversation_config,
            product_category_field=category_field,
            product_subcategory_field=subcategory_field,
        )
        for index, record in enumerate(eval_records)
    ]

    metadata = {
        "run": run_metadata(config),
        "source_file": str(input_path),
        "seed": seed,
        "split_strategy": dataset_config["split_strategy"],
        "max_pair_share": max_pair_share,
        "counts": {
            "experiment_sample": len(experiment_sample),
            "popular": len(selected_popular),
            "other_or_long_tail": len(selected_other),
            "experience": len(experience_cases),
            "reference_background": len(reference_cases),
            "eval": len(eval_cases),
        },
        "composition": {
            "experience": {
                "popular": experience_popular_count,
                "other_or_long_tail": experience_other_count,
            },
            "reference_background": {
                "popular": reference_popular_count,
                "other_or_long_tail": reference_other_count,
            },
            "eval": {
                "popular": eval_popular_count,
                "other_or_long_tail": eval_other_count,
            },
        },
        "pair_counts": {
            "sample": counts_by_pair(experiment_sample, category_field, subcategory_field),
            "experience": counts_by_pair(experience_records, category_field, subcategory_field),
            "reference_background": counts_by_pair(
                reference_records, category_field, subcategory_field
            ),
            "eval": counts_by_pair(eval_records, category_field, subcategory_field),
        },
        "leakage_policy": {
            "morag_source": "experience only",
            "reference_rag_source": "experience + reference_background",
            "eval_source": "held out from morag and reference_rag",
        },
        "conversation_config": conversation_config,
    }

    write_json(sample_output, experiment_sample)
    write_json(experience_output, experience_cases)
    write_json(reference_output, reference_cases)
    write_json(eval_output, eval_cases)
    write_json(metadata_output, metadata)

    print(f"Wrote {len(experiment_sample):,} sampled records to {sample_output}")
    print(f"Wrote {len(experience_cases):,} experience cases to {experience_output}")
    print(f"Wrote {len(reference_cases):,} reference cases to {reference_output}")
    print(f"Wrote {len(eval_cases):,} eval cases to {eval_output}")
    print(f"Wrote split metadata to {metadata_output}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--sample-output", type=pathlib.Path)
    parser.add_argument("--experience-output", type=pathlib.Path)
    parser.add_argument("--reference-output", type=pathlib.Path)
    parser.add_argument("--eval-output", type=pathlib.Path)
    parser.add_argument("--metadata-output", type=pathlib.Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-pair-share", type=float)
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.dataset_config, args.run_id)
    build_morag_experiment_split(
        config=config,
        input_path=args.input,
        sample_output=args.sample_output,
        experience_output=args.experience_output,
        reference_output=args.reference_output,
        eval_output=args.eval_output,
        metadata_output=args.metadata_output,
        seed=args.seed,
        max_pair_share=args.max_pair_share,
    )


if __name__ == "__main__":
    main()
