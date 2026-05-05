"""Pipeline 1: build the annotated pool and experiment split.

Run from the code/ directory:

    python -m scripts.pipelines.build_experiment_data
"""

from __future__ import annotations

import argparse

from scripts.common.cli_config import add_config_args, load_experiment_config
from scripts.data.build_annotated_pool import build_annotated_pool
from scripts.data.build_morag_experiment_split import build_morag_experiment_split


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_args(parser)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-pair-share", type=float)
    parser.add_argument("--skip-pool", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config, args.dataset_config, args.run_id)

    if not args.skip_pool:
        build_annotated_pool(config=config)

    if not args.skip_split:
        build_morag_experiment_split(
            config=config,
            seed=args.seed,
            max_pair_share=args.max_pair_share,
        )


if __name__ == "__main__":
    main()
