"""Shared CLI/config defaults for prototype scripts."""

from __future__ import annotations

import argparse
import pathlib
from datetime import datetime
from typing import Any

from scripts.common.morag_utils import load_toml


DEFAULT_CONFIG = pathlib.Path("configs/experiment_config.toml")
DEFAULT_MODEL_CONFIG = pathlib.Path("configs/model_config.toml")


def add_config_args(parser: argparse.ArgumentParser, include_model_config: bool = False) -> None:
    parser.add_argument("--config", type=pathlib.Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-config", type=pathlib.Path)
    parser.add_argument("--run-id")
    if include_model_config:
        parser.add_argument("--model-config", type=pathlib.Path, default=DEFAULT_MODEL_CONFIG)


def resolve_run_id(value: str | None) -> str:
    if not value or value == "latest":
        return "latest"
    if value == "auto":
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return value


def render_value(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**replacements)
    if isinstance(value, list):
        return [render_value(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: render_value(item, replacements) for key, item in value.items()}
    return value


def dataset_config_path(
    experiment_config: dict[str, Any],
    override: pathlib.Path | None,
) -> pathlib.Path:
    if override is not None:
        return override
    return pathlib.Path(experiment_config["dataset_config"]["default"])


def load_experiment_config(
    path: pathlib.Path = DEFAULT_CONFIG,
    dataset_config: pathlib.Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    config = load_toml(path)
    resolved_dataset_config_path = dataset_config_path(config, dataset_config)
    dataset = load_toml(resolved_dataset_config_path)
    conversation = dataset.pop("conversation", {})
    dataset_slug = dataset["slug"]

    run_config = dict(config.get("run", {}))
    run_config["run_id"] = resolve_run_id(run_id or run_config.get("run_id"))
    replacements = {
        "dataset_slug": dataset_slug,
        "run_id": run_config["run_id"],
    }
    run_config["output_root"] = render_value(
        run_config.get("output_root", "data/runs/{dataset_slug}/{run_id}"),
        replacements,
    )
    replacements["output_root"] = run_config["output_root"]

    config["dataset"] = render_value(dataset, replacements)
    config["conversation"] = render_value(conversation, replacements)
    config["run"] = run_config
    config["outputs"] = render_value(config["outputs"], replacements)
    config["_config_paths"] = {
        "experiment_config": str(path),
        "dataset_config": str(resolved_dataset_config_path),
    }
    return config


def run_metadata(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    run_config = config["run"]
    return {
        "dataset_slug": dataset["slug"],
        "dataset_source": dataset.get("source"),
        "run_id": run_config["run_id"],
        "output_root": run_config["output_root"],
        "experiment_config": config.get("_config_paths", {}).get("experiment_config"),
        "dataset_config": config.get("_config_paths", {}).get("dataset_config"),
    }


def load_experiment_and_model_configs(
    config_path: pathlib.Path = DEFAULT_CONFIG,
    model_config_path: pathlib.Path = DEFAULT_MODEL_CONFIG,
    dataset_config_path: pathlib.Path | None = None,
    run_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        load_experiment_config(
            path=config_path,
            dataset_config=dataset_config_path,
            run_id=run_id,
        ),
        load_toml(model_config_path),
    )
