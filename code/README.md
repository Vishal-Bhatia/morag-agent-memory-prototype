# moRAG Prototype Code

Run all commands from this `code/` directory. Paths in `configs/experiment_config.toml` are relative to this directory.

## Current Workflow

The current workflow is organized into three high-level pipelines:

```powershell
python -m scripts.pipelines.build_experiment_data
python -m scripts.pipelines.build_reference_layer --force-backend faiss
python -m scripts.pipelines.build_morag_layer --force-backend faiss
```

Then run answer generation and judging:

```powershell
python -m scripts.eval.run_support_agents --methods no_memory,reference_rag,morag
python -m scripts.eval.judge_results --limit 150
```

The current default judge is `gpt-5.4-nano`, configured in `configs/model_config.toml`.

## Current Status

The latest Sutro run should be treated as a directional prototype result:

- `morag` had the best judged answer quality.
- `morag` used far fewer answer-generation tokens than `reference_rag`.
- Retrieval quality remained weak because the dataset is more product-specific than process-repeatable.
- The run does not prove moRAG is generally superior to RAG.

Key answer scores from the 150 judged cases:

| Method | Overall | Specificity | Groundedness |
|---|---:|---:|---:|
| no_memory | 3.287 | 2.260 | 2.353 |
| reference_rag | 3.720 | 3.113 | 3.033 |
| morag | 3.800 | 3.420 | 3.433 |

Answer-generation token use across 600 eval cases:

| Method | Total Tokens | Avg Tokens / Case |
|---|---:|---:|
| no_memory | 249,142 | 415 |
| reference_rag | 3,404,185 | 5,674 |
| morag | 657,758 | 1,096 |

Use this code as a prototype scaffold. Stronger claims need a better recurring-process dataset and a rolling support-queue simulation.

## Layout

- `configs/`: model, experiment, and dataset configuration.
- `prompts/`: agent, memory extraction, safety review, reference generation, and judge prompts.
- `data/`: local generated artifacts; ignored by Git.
- `scripts/`: repeatable utility scripts grouped by pipeline stage.
- `morag_eval_rubric.md`: evaluation design and safety policy.

Script groups:

- `scripts/common/`: shared config, prompt, conversation, and vector helpers.
- `scripts/data/`: dataset download and split creation.
- `scripts/stores/`: reference RAG, past-conversation fallback, moRAG record, and vector-index builders.
- `scripts/memory/`: moRAG memory generation, review, deduplication, and legacy memory-pipeline orchestration.
- `scripts/eval/`: retrieval previews, answer generation, judging, and aggregation.
- `scripts/pipelines/`: high-level pipeline entry points.

## Datasets And Runs

Dataset-specific settings live in `configs/datasets/`. The default dataset is selected in `configs/experiment_config.toml`:

```toml
[dataset_config]
default = "configs/datasets/sutro_customer_support.toml"
```

Use `--dataset-config` to run the same experiment against another compatible dataset config.

All run outputs are scoped by dataset and run id:

```text
data/runs/{dataset_slug}/{run_id}/
```

Use stable run ids when comparing experiments:

```powershell
python -m scripts.pipelines.build_experiment_data --run-id 2026-05-05_sutro_reference_morag
python -m scripts.pipelines.build_reference_layer --run-id 2026-05-05_sutro_reference_morag --force-backend faiss
python -m scripts.pipelines.build_morag_layer --run-id 2026-05-05_sutro_reference_morag --force-backend faiss
python -m scripts.eval.run_support_agents --run-id 2026-05-05_sutro_reference_morag --methods no_memory,reference_rag,morag
python -m scripts.eval.judge_results --run-id 2026-05-05_sutro_reference_morag --limit 150
```

Use `--run-id auto` for a timestamped ad hoc run. The default run id is `latest`, which is convenient for local iteration but will overwrite prior artifacts for the same dataset.

## Pipeline 1: Experiment Data

```powershell
python -m scripts.pipelines.build_experiment_data
```

This builds:

- annotated valid cases
- 1,200-case experiment sample
- 400 experience cases
- 200 reference-background cases
- 600 eval cases

## Pipeline 2: Reference Layer

```powershell
python -m scripts.pipelines.build_reference_layer --force-backend faiss
```

This builds:

- policy-like `reference_rag` guidance documents
- `reference_rag` vector index
- separate past-conversation fallback store

The fallback store is built for future layered retrieval experiments. The current answer run does not automatically use it.

## Pipeline 3: moRAG Layer

```powershell
python -m scripts.pipelines.build_morag_layer --force-backend faiss
```

This builds:

- memory candidates from experience cases
- safety-reviewed approved memories
- deduplicated memories
- moRAG vector index

For a smaller pilot run:

```powershell
python -m scripts.pipelines.build_morag_layer --limit 25 --force-backend faiss
```

## Evaluate Answers

```powershell
python -m scripts.eval.run_support_agents --methods no_memory,reference_rag,morag
python -m scripts.eval.judge_results --limit 150
```

The judge output includes:

- answer quality
- strict retrieval hit/quality
- broader retrieval usefulness

To run only retrieval usefulness on an existing answer file:

```powershell
python -m scripts.eval.judge_results --limit 150 --skip-answer --skip-exact-retrieval --output data/runs/sutro_customer_support/latest/results/retrieval_usefulness_results.json
```

## Preview Retrieval

```powershell
python -m scripts.eval.preview_retrieval --store reference_rag
python -m scripts.eval.preview_retrieval --store morag
```

The legacy `raw_rag` store remains available as a past-conversation fallback store:

```powershell
python -m scripts.eval.preview_retrieval --store raw_rag
```

## Aggregate Results

Compile completed run summaries into one JSON file:

```powershell
python -m scripts.eval.aggregate_run_results
```

Default output:

```text
data/results/all_runs_summary.json
```

The aggregate file contains:

- `runs`: one row per discovered result file.
- `answer_summary_rows`: flattened answer-quality metrics for graphing.
- `retrieval_summary_rows`: flattened strict retrieval metrics.
- `retrieval_usefulness_summary_rows`: flattened broader retrieval-usefulness metrics.

## Secrets

The OpenAI API key is read from the environment variable configured in `configs/model_config.toml`.

Current expected variable:

```text
OPENAI_API_KEY
```
