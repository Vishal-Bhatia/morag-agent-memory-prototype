# moRAG: A Small Prototype For Agent-Owned Retrieval Notes

This repository explores a practical idea: support agents often create their own working notes to handle recurring issues faster. The question here is whether AI agents can benefit from a governed version of that behavior: **moRAG**, or "my-own-RAG".

The intent is modest. This is not presented as a frontier architecture or a broad benchmark claim. It is a small experiment around whether an agent-maintained, safety-reviewed memory layer can sit alongside ordinary retrieval and improve efficiency, retrieval cleanliness, or answer behavior in specific settings.

## What This Tests

The prototype compares three modes:

- `no_memory`: the agent responds without retrieved prior context.
- `raw_rag`: the agent receives semantically similar prior conversations.
- `morag`: the agent receives approved memory notes generated from prior cases.

The current default dataset is:

- `sutro/synthetic-customer-support-dialogues-20k`

The first run showed an important limitation: the sample split had weak repeatability between the experience set and eval set, so it should not be used to claim broad answer-quality gains. The code is still useful for testing the mechanics of memory creation, review, indexing, retrieval, cost, and evaluation.

## Current Status

This is an early prototype. It is ready to inspect, run, criticize, and extend, but it should not be read as proof that moRAG improves answer quality in general.

The first run used 100 records from `sutro/synthetic-customer-support-dialogues-20k`:

- First 75 records: experience set used for raw RAG and memory generation.
- Next 25 records: eval set.
- Methods compared: `no_memory`, `raw_rag`, `morag`.
- Judges: `gpt-5.5` and `gpt-5.4`.

Memory pipeline output:

```text
Memory candidates generated: 246
Reviewed by safety reviewer: 246
Approved/edited: 239
Rejected: 7
After deduplication: 237 memories
```

Answer-quality summary:

```text
gpt-5.5 overall answer score
no_memory: 3.80
raw_rag:   3.00
morag:     3.24

gpt-5.4 overall answer score
no_memory: 3.68
raw_rag:   3.12
morag:     3.24
```

So, in this run, `no_memory` scored best on answer quality. The result does not support a quality-lift claim.

Retrieval-quality summary:

```text
gpt-5.5 retrieval
raw_rag hit rate: 0.12
morag hit rate:   0.28
raw_rag memory pollution rate: 0.32
morag memory pollution rate:   0.00

gpt-5.4 retrieval
raw_rag hit rate: 0.16
morag hit rate:   0.24
raw_rag memory pollution rate: 0.20
morag memory pollution rate:   0.00
```

The useful signal was that moRAG retrieved cleaner context than raw RAG and avoided judged retrieval pollution in this run.

Answer-generation token use:

```text
no_memory: 11,057 total tokens
raw_rag:   79,095 total tokens
morag:     26,340 total tokens
```

Average per answer:

```text
no_memory: ~442 tokens
raw_rag:   ~3,164 tokens
morag:     ~1,054 tokens
```

This shows that moRAG reduced answer-time token usage versus raw RAG by using compact approved notes instead of full prior conversations.

However, that answer-time comparison does **not** include memory lifecycle cost:

```text
Memory candidate generation: 185,665 tokens
Safety review:               94,245 tokens
Total moRAG setup/review:    279,910 tokens
```

In this small run, moRAG does not win on total lifecycle tokens. The savings only become meaningful if approved memories are reused across enough future cases. With this run's numbers, the rough break-even point is about 130 similar future answers.

The practical conclusion:

> This prototype validates the mechanics of governed memory creation, safety review, retrieval, and evaluation. It also shows cleaner retrieval and lower answer-time token use than raw RAG in the first run. It does not yet prove answer-quality improvement or total lifecycle cost efficiency.

The next step is a more lifelike rolling support-queue simulation on data with recurring issues.

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
  data/
  requirements.txt
  README.md
```

Generated data and results are kept out of Git by default. Runs are organized as:

```text
code/data/runs/{dataset_slug}/{run_id}/
```

This makes it easier to compare multiple datasets and repeated runs.

## Quick Start

From the `code/` directory:

```powershell
pip install -r requirements.txt
```

Set your OpenAI API key:

```powershell
$env:OPENAI_API_KEY="..."
```

Download the configured dataset sample:

```powershell
python -m scripts.data.download_hf_sample
```

Build the case split:

```powershell
python -m scripts.data.build_case_split --run-id latest
```

Build raw RAG:

```powershell
python -m scripts.stores.build_store --store raw_rag --force-backend faiss --run-id latest
```

Build moRAG:

```powershell
python -m scripts.memory.build_morag_pipeline --force-backend faiss --run-id latest
```

Run agent answers and judge them:

```powershell
python -m scripts.eval.run_support_agents --run-id latest
python -m scripts.eval.judge_results --run-id latest
```

Aggregate all completed runs:

```powershell
python -m scripts.eval.aggregate_run_results
```

More detailed commands are in [code/README.md](code/README.md).

## Author Note

The author is interested in practical ways of making AI systems more efficient, especially where retrieval, memory, and workflow design can reduce repeated effort. The author has a moderate grasp of Python coding and an above-average understanding of neural networks, deep learning, machine learning, and AI systems.

This project was substantially built with AI coding assistance. In plain terms, it is "vibe coded", but not blindly accepted. The author reviewed the code, challenged design choices, tested pieces of the pipeline, and used the process to brainstorm safer and cleaner ways to structure the experiment.

Codex was the coding helper for this version of the repository.

Criticism, corrections, and suggestions are welcome. The author also apologizes in advance if something obvious has been missed.

## License

This project is released under the MIT License. That makes it free to use, copy, modify, merge, publish, distribute, sublicense, and sell, provided the copyright and license notice are included.

If the intent is to prevent enterprise or commercial use, MIT is not the right license. For now, this repository uses MIT to keep research and iteration simple while requiring attribution through the license notice.
