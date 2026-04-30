# moRAG Prototype Code

Run all commands from this `code/` directory.

The paths in `configs/experiment_config.toml` are relative to this directory.

## Current Status

The first run should be treated as a pipeline and measurement sanity check, not
as proof that moRAG improves answer quality. On the initial Sutro 75/25 split:

- `no_memory` had the best judged answer quality.
- `morag` performed better than `raw_rag` on answer quality, but not better than `no_memory`.
- `morag` retrieved cleaner context than `raw_rag`.
- `morag` used far fewer answer-generation tokens than `raw_rag`.
- memory generation and safety review cost more tokens than the small eval saved.

Key token figures:

```text
Answer generation:
no_memory: 11,057 tokens
raw_rag:   79,095 tokens
morag:     26,340 tokens

moRAG setup/review:
candidate generation: 185,665 tokens
safety review:         94,245 tokens
total:                279,910 tokens
```

Use this code as a prototype scaffold. Stronger claims need a better recurring-issue dataset and a rolling support-queue simulation.

## Current Layout

- `configs/`: model, experiment, and dataset configuration.
- `prompts/`: agent, memory extraction, safety review, and judge prompts.
- `data/`: local dataset samples and future generated artifacts.
- `scripts/`: repeatable utility scripts grouped by pipeline stage.
- `morag_eval_rubric.md`: evaluation design and safety policy.

Script groups:

- `scripts/common/`: shared config, prompt, conversation, and vector helpers.
- `scripts/data/`: dataset download and case split creation.
- `scripts/stores/`: raw RAG/moRAG record and vector-index builders.
- `scripts/memory/`: moRAG memory generation, review, deduplication, and pipeline orchestration.
- `scripts/eval/`: retrieval previews, answer generation, and judging.

## Datasets And Runs

Dataset-specific settings live in `configs/datasets/`. The default dataset is
selected in `configs/experiment_config.toml`:

```toml
[dataset_config]
default = "configs/datasets/sutro_customer_support.toml"
```

Use `--dataset-config` to run the same experiment against another compatible
dataset config.

All run outputs are scoped by dataset and run id:

```text
data/runs/{dataset_slug}/{run_id}/
```

Use stable run ids when comparing experiments:

```powershell
python -m scripts.data.build_case_split --run-id 2026-04-30_sutro_baseline
python -m scripts.memory.build_morag_pipeline --run-id 2026-04-30_sutro_baseline --force-backend faiss
python -m scripts.eval.run_support_agents --run-id 2026-04-30_sutro_baseline
python -m scripts.eval.judge_results --run-id 2026-04-30_sutro_baseline
```

Use `--run-id auto` for a timestamped ad hoc run. The default run id is
`latest`, which is convenient for local iteration but will overwrite prior
artifacts for the same dataset.

## Regenerate Dataset Sample

```powershell
python -m scripts.data.download_hf_sample
```

The downloader reads dataset source, split, sample size, selected columns, and
output path from `configs/datasets/sutro_customer_support.toml`. CLI flags such
as `--dataset-config`, `--run-id`, `--dataset`, `--split`, `--sample-size`,
`--keep-column`, and `--output` can override the config for quick trials.

## Build Retrieval Stores

```powershell
python -m scripts.stores.build_store --store raw_rag --force-backend faiss
python -m scripts.stores.build_store --store morag --force-backend faiss
```

Use `scripts.stores.build_raw_rag_records`,
`scripts.stores.build_morag_memory_records`, or
`scripts.stores.build_vector_index` directly when debugging one stage at a time.

## Build moRAG Memory Pipeline

```powershell
python -m scripts.memory.build_morag_pipeline --force-backend faiss
```

For a smaller pilot run:

```powershell
python -m scripts.memory.build_morag_pipeline --limit 15 --force-backend faiss
```

## Preview Retrieval

```powershell
python -m scripts.eval.preview_retrieval --store raw_rag
python -m scripts.eval.preview_retrieval --store morag
```

## Evaluate Answers

```powershell
python -m scripts.eval.run_support_agents
python -m scripts.eval.judge_results
```

## Aggregate Results

Compile all completed run summaries into one JSON file:

```powershell
python -m scripts.eval.aggregate_run_results
```

Default output:

```text
data/results/all_runs_summary.json
```

The aggregate file contains:

- `runs`: one row per discovered run.
- `answer_summary_rows`: flattened answer-quality metrics for graphing.
- `retrieval_summary_rows`: flattened retrieval metrics for graphing.

## Secrets

The OpenAI API key is read from the environment variable configured in `configs/model_config.toml`.

Current expected variable:

```text
OPENAI_API_KEY
```
