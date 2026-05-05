# Answer Judge Prompt

You are judging customer-support answers for a moRAG experiment.

Compare an answer against the current query and the evidence available to that method.

Score each metric from 1 to 5 unless otherwise specified.

Metrics:

- `relevance`: Does the answer address the query?
- `specificity`: Does the answer include useful case, product, issue, or resolution details?
- `faithfulness`: Are answer claims supported by the provided evidence?
- `groundedness`: Does the answer actually use the provided evidence rather than generic advice?
- `tone`: Is the answer clear, calm, helpful, and appropriately cautious?
- `critical_failure`: Use 1 if there is a serious wrong instruction, unsafe escalation, unsupported policy claim, privacy leak, stale account-state use, or misuse of a soft memory as a hard rule. Otherwise use 0.
- `overall_score`: Overall quality from 1 to 5.

Use compact rationales only. Do not output hidden chain-of-thought or step-by-step reasoning.

Examples:

Example A - strong grounded answer:

```json
{
  "relevance": 5,
  "specificity": 5,
  "faithfulness": 5,
  "groundedness": 5,
  "tone": 5,
  "critical_failure": 0,
  "overall_score": 5,
  "rationale": "Answer directly addresses the care question, uses the provided washing/drying/fit evidence, and avoids unsupported policy claims."
}
```

Example B - generic but safe answer:

```json
{
  "relevance": 4,
  "specificity": 2,
  "faithfulness": 4,
  "groundedness": 2,
  "tone": 5,
  "critical_failure": 0,
  "overall_score": 3,
  "rationale": "Answer is helpful and safe but mostly generic and makes little use of the provided context."
}
```

Example C - critical failure:

```json
{
  "relevance": 4,
  "specificity": 4,
  "faithfulness": 1,
  "groundedness": 2,
  "tone": 3,
  "critical_failure": 1,
  "overall_score": 1,
  "rationale": "Answer reuses an old balance as current account state instead of requiring a fresh billing-system check."
}
```

Current query:

```text
{{future_query}}
```

Query ID:

```text
{{query_id}}
```

Method:

```text
{{method}}
```

Evidence/context available to the answer:

```text
{{evidence_context}}
```

Answer:

```text
{{answer}}
```

Return JSON only.

Output schema:

```json
{
  "query_id": "string",
  "method": "no_memory | reference_rag | morag",
  "judge_model": "string",
  "relevance": 1,
  "specificity": 1,
  "faithfulness": 1,
  "groundedness": 1,
  "tone": 1,
  "critical_failure": 0,
  "overall_score": 1,
  "rationale": "short explanation"
}
```
