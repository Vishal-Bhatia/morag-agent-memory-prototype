"""Build a clean JSON pool from the annotated Sutro workbook.

The annotated Excel file is treated as read-only. This script writes a
machine-readable JSON file under the configured run directory.

Run from the code/ directory:

    python -m scripts.data.build_annotated_pool --run-id auto
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import pandas as pd

from scripts.common.cli_config import add_config_args, load_experiment_config, run_metadata
from scripts.common.morag_utils import write_json


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def compact_record(
    row_idx: int,
    row: dict[str, Any],
    keep_columns: list[str],
    conversation_id_prefix: str,
    product_category_field: str,
    product_subcategory_field: str,
) -> dict[str, Any]:
    compact = {
        "conversation_id": f"{conversation_id_prefix}_{row_idx:05d}",
        "source_row_idx": row_idx,
        "product_id": clean_scalar(row.get("product_id")),
        product_category_field: clean_scalar(row.get(product_category_field)),
        product_subcategory_field: clean_scalar(row.get(product_subcategory_field)),
    }
    compact.update({column: clean_scalar(row.get(column)) for column in keep_columns})
    return compact


def build_annotated_pool(
    config: dict[str, Any],
    input_path: pathlib.Path | None = None,
    output_path: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    dataset_config = config["dataset"]

    input_path = input_path or pathlib.Path(dataset_config["annotated_workbook_file"])
    output_path = output_path or pathlib.Path(dataset_config["annotated_pool_file"])
    sheet_name = dataset_config.get("annotated_workbook_sheet", "data")
    validity_field = dataset_config["annotation_validity_field"]
    validity_value = str(dataset_config["annotation_validity_value"]).lower()
    product_category_field = dataset_config["product_category_field"]
    product_subcategory_field = dataset_config["product_subcategory_field"]
    keep_columns = list(dataset_config["download_keep_columns"])
    conversation_id_prefix = dataset_config["conversation_id_prefix"]

    dataframe = pd.read_excel(input_path, sheet_name=sheet_name)
    dataframe = dataframe[
        dataframe[validity_field].astype(str).str.lower().eq(validity_value)
    ].reset_index(drop=False)
    dataframe = dataframe.rename(columns={"index": "source_row_idx"})

    required_columns = {
        "product_id",
        product_category_field,
        product_subcategory_field,
        *keep_columns,
    }
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Annotated workbook missing columns: {', '.join(missing_columns)}")

    records = [
        compact_record(
            row_idx=int(row["source_row_idx"]),
            row=row.to_dict(),
            keep_columns=keep_columns,
            conversation_id_prefix=conversation_id_prefix,
            product_category_field=product_category_field,
            product_subcategory_field=product_subcategory_field,
        )
        for _, row in dataframe.iterrows()
    ]

    metadata = {
        "run": run_metadata(config),
        "source_file": str(input_path),
        "source_sheet": sheet_name,
        "validity_field": validity_field,
        "validity_value": validity_value,
        "record_count": len(records),
        "product_category_field": product_category_field,
        "product_subcategory_field": product_subcategory_field,
    }

    write_json(output_path, records)
    write_json(output_path.with_name(output_path.stem + "_metadata.json"), metadata)

    print(f"Read annotated workbook: {input_path}")
    print(f"Wrote {len(records):,} valid records to {output_path}")
    print(f"Wrote metadata to {output_path.with_name(output_path.stem + '_metadata.json')}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--input", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.dataset_config, args.run_id)
    build_annotated_pool(
        config=config,
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
