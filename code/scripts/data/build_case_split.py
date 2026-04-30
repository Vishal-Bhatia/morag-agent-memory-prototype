r"""Build the sequential moRAG experience/eval split.

Run from the code/ directory:

    python .\scripts\build_case_split.py
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

from scripts.common.cli_config import add_config_args, load_experiment_config, run_metadata
from scripts.common.conversation_utils import first_customer_turn, normalize_turns, render_turns
from scripts.common.morag_utils import load_json, write_json


def base_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_name": record.get("product_name"),
        "product_description": record.get("product_description"),
        "issue_type": record.get("issue_type"),
        "issue_severity": record.get("issue_severity"),
        "issue_description": record.get("issue_description"),
        "resolution_path_description": record.get("resolution_path_description"),
        "outcome": record.get("outcome"),
        "customer_sentiment": record.get("customer_sentiment"),
        "customer_satisfaction": record.get("customer_satisfaction"),
    }


def build_experience_case(
    record: dict[str, Any],
    ordinal: int,
    case_id_prefix: str,
    past_conversation_field: str,
    conversation_config: dict[str, Any],
) -> dict[str, Any]:
    past_turns = normalize_turns(record[past_conversation_field], conversation_config)
    return {
        "case_id": f"{case_id_prefix}_{ordinal:03d}",
        "source_conversation_id": record["conversation_id"],
        "source_row_idx": record["source_row_idx"],
        "split": "experience",
        "past_conversation_turns": past_turns,
        "past_conversation_text": render_turns(past_turns),
        "metadata": base_metadata(record),
    }


def build_eval_case(
    record: dict[str, Any],
    ordinal: int,
    case_id_prefix: str,
    past_conversation_field: str,
    future_dialogue_field: str,
    conversation_config: dict[str, Any],
) -> dict[str, Any]:
    past_turns = normalize_turns(record[past_conversation_field], conversation_config)
    future_turns = normalize_turns(record[future_dialogue_field], conversation_config)
    first_turn = first_customer_turn(future_turns)

    if first_turn is None:
        raise ValueError(f"No customer turn found for {record['conversation_id']}")

    return {
        "case_id": f"{case_id_prefix}_{ordinal:03d}",
        "source_conversation_id": record["conversation_id"],
        "source_row_idx": record["source_row_idx"],
        "split": "eval",
        "future_query": first_turn["text"],
        "future_query_turn": first_turn,
        "future_dialogue_turns_reference_only": future_turns,
        "past_conversation_turns_reference_only": past_turns,
        "expected_resolution_pattern": record.get("resolution_path_description"),
        "metadata": base_metadata(record),
    }


def configured_path(value: str) -> pathlib.Path:
    return pathlib.Path(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--experience-output", type=pathlib.Path)
    parser.add_argument("--eval-output", type=pathlib.Path)
    parser.add_argument("--metadata-output", type=pathlib.Path)
    parser.add_argument("--experience-count", type=int)
    parser.add_argument("--eval-count", type=int)
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.dataset_config, args.run_id)
    dataset_config = config["dataset"]
    conversation_config = config.get("conversation", {})

    input_path = args.input or configured_path(dataset_config["sample_file"])
    experience_output = args.experience_output or configured_path(
        dataset_config["experience_cases_file"]
    )
    eval_output = args.eval_output or configured_path(dataset_config["eval_cases_file"])
    metadata_output = args.metadata_output or configured_path(
        dataset_config["split_metadata_file"]
    )
    experience_count = args.experience_count or int(dataset_config["experience_case_count"])
    eval_count = args.eval_count or int(dataset_config["eval_case_count"])
    split_strategy = dataset_config.get("split_strategy", "first_n_experience_next_m_eval")
    experience_case_id_prefix = dataset_config["experience_case_id_prefix"]
    eval_case_id_prefix = dataset_config["eval_case_id_prefix"]
    past_conversation_field = dataset_config["past_conversation_field"]
    future_dialogue_field = dataset_config["future_dialogue_field"]

    records = load_json(input_path)
    required = experience_count + eval_count
    if len(records) < required:
        raise ValueError(f"Need at least {required} records, found {len(records)}")

    experience_records = records[:experience_count]
    eval_records = records[experience_count:required]

    experience_cases = [
        build_experience_case(
            record=record,
            ordinal=index + 1,
            case_id_prefix=experience_case_id_prefix,
            past_conversation_field=past_conversation_field,
            conversation_config=conversation_config,
        )
        for index, record in enumerate(experience_records)
    ]
    eval_cases = [
        build_eval_case(
            record=record,
            ordinal=index + 1,
            case_id_prefix=eval_case_id_prefix,
            past_conversation_field=past_conversation_field,
            future_dialogue_field=future_dialogue_field,
            conversation_config=conversation_config,
        )
        for index, record in enumerate(eval_records)
    ]

    metadata = {
        "run": run_metadata(config),
        "source_file": str(input_path),
        "split_strategy": split_strategy,
        "experience_count": len(experience_cases),
        "eval_count": len(eval_cases),
        "experience_source_row_idx_range": [
            experience_cases[0]["source_row_idx"],
            experience_cases[-1]["source_row_idx"],
        ],
        "eval_source_row_idx_range": [
            eval_cases[0]["source_row_idx"],
            eval_cases[-1]["source_row_idx"],
        ],
        "live_interaction_mode": "single_turn",
        "future_query_source": "first_customer_turn_from_new_customer_support_dialogue",
        "past_experience_source": past_conversation_field,
        "conversation_config": conversation_config,
    }

    write_json(experience_output, experience_cases)
    write_json(eval_output, eval_cases)
    write_json(metadata_output, metadata)

    print(f"Wrote {len(experience_cases)} experience cases to {experience_output}")
    print(f"Wrote {len(eval_cases)} eval cases to {eval_output}")
    print(f"Wrote split metadata to {metadata_output}")


if __name__ == "__main__":
    main()
