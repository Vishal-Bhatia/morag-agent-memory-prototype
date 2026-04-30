"""Download a compact local sample from a Hugging Face dataset.

This uses the Hugging Face Dataset Viewer API instead of the full datasets
library so the prototype can be reproduced with only Python's standard library.

The default settings come from configs/experiment_config.toml. CLI arguments can
override those settings for quick experiments with another compatible dataset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import urllib.parse
import urllib.request

from scripts.common.cli_config import add_config_args, load_experiment_config

BASE_URL = "https://datasets-server.huggingface.co/rows"


def fetch_rows(
    dataset: str,
    dataset_config: str,
    split: str,
    offset: int,
    length: int,
) -> dict:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": dataset_config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    url = f"{BASE_URL}?{params}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def compact_row(
    row_idx: int,
    row: dict,
    keep_columns: list[str],
    conversation_id_prefix: str,
) -> dict:
    compact = {
        "conversation_id": f"{conversation_id_prefix}_{row_idx:05d}",
        "source_row_idx": row_idx,
    }
    compact.update({column: row.get(column) for column in keep_columns})
    return compact


def download_sample(
    dataset: str,
    dataset_config: str,
    split: str,
    sample_size: int,
    batch_size: int,
    keep_columns: list[str],
    conversation_id_prefix: str,
) -> list[dict]:
    records: list[dict] = []
    offset = 0

    while len(records) < sample_size:
        length = min(batch_size, sample_size - len(records))
        payload = fetch_rows(
            dataset=dataset,
            dataset_config=dataset_config,
            split=split,
            offset=offset,
            length=length,
        )

        rows = payload.get("rows", [])
        if not rows:
            break

        for item in rows:
            records.append(
                compact_row(
                    row_idx=item["row_idx"],
                    row=item["row"],
                    keep_columns=keep_columns,
                    conversation_id_prefix=conversation_id_prefix,
                )
            )

        offset += len(rows)

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--dataset")
    parser.add_argument("--hf-config")
    parser.add_argument("--split")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--conversation-id-prefix")
    parser.add_argument("--keep-column", action="append", dest="keep_columns")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.dataset_config, args.run_id)
    dataset_config = config["dataset"]

    source_dataset = args.dataset or dataset_config["source"]
    source_config = args.hf_config or dataset_config["hf_config"]
    source_split = args.split or dataset_config["hf_split"]
    sample_size = args.sample_size or int(dataset_config["sample_size"])
    batch_size = args.batch_size or int(dataset_config["download_batch_size"])
    keep_columns = args.keep_columns or list(dataset_config["download_keep_columns"])
    output_path = args.output or pathlib.Path(dataset_config["sample_file"])
    conversation_id_prefix = (
        args.conversation_id_prefix
        or dataset_config.get("conversation_id_prefix")
        or f"{source_dataset.replace('/', '_')}_{source_split}"
    )

    records = download_sample(
        dataset=source_dataset,
        dataset_config=source_config,
        split=source_split,
        sample_size=sample_size,
        batch_size=min(batch_size, 100),
        keep_columns=keep_columns,
        conversation_id_prefix=conversation_id_prefix,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(f"Dataset: {source_dataset}")
    print(f"Config/split: {source_config}/{source_split}")
    print(f"Columns: {', '.join(keep_columns)}")
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()
