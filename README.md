# moRAG: Agent-Owned Retrieval Notes

This repository explores a practical idea: human support agents often create their own working notes to handle recurring issues faster. The question here is whether AI agents can benefit from a governed version of that behavior: **moRAG**, or "my-own-RAG".

The intent is modest. This is not presented as a frontier architecture or a broad benchmark claim. It is a prototype for testing whether an agent-maintained, safety-reviewed memory layer can sit alongside ordinary retrieval and improve answer behavior or context efficiency in support-style workflows.

## Current Status

This is an early prototype and directional experiment, not a production claim.

The current Sutro run is encouraging:

- moRAG produced the highest judged answer quality.
- moRAG used far fewer answer-time tokens than broad reference RAG.
- Even including memory extraction and review, moRAG used fewer tokens than the reference-RAG path in this run.
- Retrieval quality remained weak, largely because the dataset often required product/SKU-specific facts that did not repeat cleanly.

There is no evidence here to support universal supremacy of moRAG over RAG. The result should be read in the following context:

> moRAG looks promising as a compact operational memory layer over ordinary RAG, especially for support environments where workflows, escalation patterns, defect modes, and handling conventions repeat. It theoretically, and in a limited way empirically, improves overall quality and lowers token consumption versus broad reference RAG.

I am looking for collaborators, better datasets, or enterprise teams willing to test this under controlled conditions. I am open to NDAs and to running the experiment in a production-like environment where privacy, safety review, governance, and auditability are handled properly.

## What This Tests

The current experiment compares three isolated modes:

- `no_memory`: the agent responds without retrieved prior context.
- `reference_rag`: the agent receives broad, policy-like support guidance built from prior cases.
- `morag`: the agent receives approved, compact memory notes generated from prior cases.

The current dataset is:

- `sutro/synthetic-customer-support-dialogues-20k`

The current run uses an annotated Sutro split:

- 1,200 total sampled cases
- 400 experience cases for moRAG memory creation
- 200 reference-background cases for reference guidance
- 600 eval cases for answer generation
- 150 eval cases judged with `gpt-5.4-nano`

## Latest Results

Answer-quality summary from the 150 judged cases:

| Method | Overall | Specificity | Groundedness | Faithfulness | Critical Failure |
|---|---:|---:|---:|---:|---:|
| no_memory | 3.287 | 2.260 | 2.353 | 4.507 | 0.000 |
| reference_rag | 3.720 | 3.113 | 3.033 | 4.207 | 0.000 |
| morag | 3.800 | 3.420 | 3.433 | 4.080 | 0.000 |

Strict retrieval summary:

| Method | Hit Rate | Retrieval Quality | Pollution Rate |
|---|---:|---:|---:|
| reference_rag | 0.060 | 1.880 | 0.000 |
| morag | 0.173 | 2.213 | 0.007 |

Broader retrieval-usefulness summary:

| Method | Context Usefulness | Context Specificity | Exact Answer Rate | Noise Rate | Misleading Rate |
|---|---:|---:|---:|---:|---:|
| reference_rag | 2.147 | 1.467 | 0.000 | 0.180 | 0.000 |
| morag | 2.120 | 1.627 | 0.007 | 0.280 | 0.000 |

Answer-generation token use across 600 eval cases:

| Method | Total Tokens | Avg Tokens / Case |
|---|---:|---:|
| no_memory | 249,142 | 415 |
| reference_rag | 3,404,185 | 5,674 |
| morag | 657,758 | 1,096 |

Memory/reference lifecycle token use:

| Stage | Total Tokens |
|---|---:|
| reference guidance generation | 245,907 |
| moRAG memory extraction | 1,077,803 |
| moRAG memory safety review | 487,533 |

Build plus answer-generation comparison:

| Path | Approx. Total Tokens |
|---|---:|
| reference RAG | 3.65M |
| moRAG | 2.22M |

The result is positive but not final. Retrieval remained weak because while annotation gave us clean categories, many eval answers depended on exact product-specific facts. In the judged 150 cases, only 21 had exact product overlap with the experience/reference material. This is a hard setting for memory reuse, and may not be representative of enterprise support queues where process steps, escalation patterns, known issue handling, and policy interpretations often repeat more consistently than individual product facts.

## Interpretation

What the run supports:

- moRAG improved judged answer quality versus both no-memory and broad reference RAG.
- moRAG sharply reduced answer-time context cost versus broad reference RAG.
- Compact approved memories can be more token-efficient than long policy-style context.
- The idea remains worth testing on more realistic support data.

What the run does not prove:

- It does not prove moRAG is generally superior to RAG.
- It does not prove moRAG replaces a KB or reference RAG.
- It does not prove retrieval is solved.
- It does not yet test a layered production policy such as `moRAG -> reference RAG -> past conversation fallback`.

The next serious test should use a process-heavy support dataset with recurring workflows, not mostly SKU-specific facts.

## Repository Layout

```text
code/
  configs/
    datasets/
  prompts/
  scripts/
    common/
    data/
    stores/
    memory/
    eval/
    pipelines/
  data/
  requirements.txt
  README.md
```

Generated data and results are kept out of Git by default. Runs are organized as:

```text
code/data/runs/{dataset_slug}/{run_id}/
```

## Quick Start

Run commands from the `code/` directory:

```powershell
cd code
pip install -r requirements.txt
$env:OPENAI_API_KEY="..."
```

Build the annotated experiment split:

```powershell
python -m scripts.pipelines.build_experiment_data --run-id latest
```

Build the reference layer:

```powershell
python -m scripts.pipelines.build_reference_layer --force-backend faiss --run-id latest
```

Build the moRAG layer:

```powershell
python -m scripts.pipelines.build_morag_layer --force-backend faiss --run-id latest
```

Run answers:

```powershell
python -m scripts.eval.run_support_agents --methods no_memory,reference_rag,morag --run-id latest
```

Judge 150 cases with the default nano judge:

```powershell
python -m scripts.eval.judge_results --limit 150 --run-id latest
```

Aggregate completed runs:

```powershell
python -m scripts.eval.aggregate_run_results
```

More detailed commands are in [code/README.md](code/README.md).

## Safety Position

moRAG is not a place to store private account state. It should store governed operational lessons, not volatile or sensitive facts.

Do not store:

- addresses
- phone numbers
- email addresses
- account balances
- payment details
- authentication details
- current subscription/account status
- one customer's private situation as a general rule

Prefer procedural memory:

```text
For balance questions, fetch the current balance from the billing system before answering.
```

Not volatile memory:

```text
Customer has an outstanding balance of $84.20.
```

## Author Note

The author, Vishal Jitenra Bhatia, is interested in practical ways of making AI systems more efficient, especially where retrieval, memory, and workflow design can reduce repeated effort. The author has a moderate grasp of Python coding and an above-average understanding of neural networks, deep learning, machine learning, and AI systems.

This project was substantially built with AI coding assistance. In plain terms, it is "vibe coded", but not blindly accepted. The author reviewed the code, challenged design choices, tested pieces of the pipeline, and used the process to brainstorm safer and cleaner ways to structure the experiment.

Codex was the coding helper for this version of the repository.

Criticism, corrections, and suggestions are welcome. The author also apologizes in advance if something obvious has been missed.

## License

This project is released under the MIT License. That makes it free to use, copy, modify, merge, publish, distribute, sublicense, and sell, provided the copyright and license notice are included.

If the intent is to prevent enterprise or commercial use, MIT is not the right license. For now, this repository uses MIT to keep research and iteration simple while requiring attribution through the license notice.

## Dataset License Note

The code in this repository is MIT licensed.

Some example datasets used for experiments may have their own licenses. In particular, `Tobi-Bueck/customer-support-tickets` is licensed under CC-BY-NC-4.0 and is used here only as a research/prototype target.

The datasets are not redistributed in this repository. Generated artifacts derived from datasets are also excluded from Git.

Commercial or enterprise users should use their own appropriately licensed data and should review the license terms of any dataset they choose to run through this code.
