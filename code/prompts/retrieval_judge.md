# Retrieval Judge Prompt

You judge whether retrieved context is useful for answering a future customer-support query.

Judge retrieval quality separately from answer quality.

Score `retrieval_quality` from 1 to 5.

Consider:

- relevance to the future query
- usefulness for producing a support answer
- lack of irrelevant noise
- source clarity and auditability
- whether retrieved context avoids unsafe private or volatile facts

For hit rate, decide whether at least one retrieved item contains an expected relevant resolution pattern, product handling note, or support workflow.

Use compact rationales only. Do not output hidden chain-of-thought or step-by-step reasoning.

Examples:

Example A - useful retrieval:

```json
{
  "hit_rate": 1,
  "retrieval_quality": 5,
  "memory_pollution_present": 0,
  "rationale": "Retrieved context contains the expected customization-form and bulk-order handling pattern with little irrelevant noise."
}
```

Example B - noisy retrieval:

```json
{
  "hit_rate": 1,
  "retrieval_quality": 3,
  "memory_pollution_present": 0,
  "rationale": "At least one item contains the expected resolution pattern, but other retrieved items are only loosely related."
}
```

Example C - polluted retrieval:

```json
{
  "hit_rate": 0,
  "retrieval_quality": 1,
  "memory_pollution_present": 1,
  "rationale": "Retrieved context includes private or volatile customer-specific facts and does not contain the expected handling pattern."
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

Expected resolution pattern:

```text
{{expected_resolution_pattern}}
```

Retrieved context:

```text
{{retrieved_context}}
```

Return JSON only.

Output schema:

```json
{
  "query_id": "string",
  "method": "raw_rag | morag",
  "judge_model": "string",
  "hit_rate": 0,
  "retrieval_quality": 1,
  "memory_pollution_present": 0,
  "rationale": "short explanation"
}
```
