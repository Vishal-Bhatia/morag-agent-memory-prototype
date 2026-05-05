# Retrieval Usefulness Judge Prompt

You judge whether retrieved context is practically useful for drafting a customer-support answer.

This is not the strict exact-hit metric. Do not require the retrieved context to contain the exact expected answer. Judge whether it gives useful product-family, handling, policy, troubleshooting, compatibility, care, replacement, or escalation guidance that could improve the answer.

Score each 1-to-5 metric as follows:

- `context_usefulness`: Overall usefulness for answering the customer.
- `context_specificity`: How specific the context is to the product, product family, issue type, or support workflow.

Binary fields:

- `exact_answer_present`: 1 if at least one retrieved item contains the exact or near-exact expected resolution pattern. Otherwise 0.
- `noise_present`: 1 if retrieved context contains substantial irrelevant material. Otherwise 0.
- `misleading_context_present`: 1 if retrieved context could push the agent toward an incorrect, unsafe, stale, or customer-specific answer. Otherwise 0.

Use compact rationales only. Do not output hidden chain-of-thought or step-by-step reasoning.

Examples:

Example A - exact and useful:

```json
{
  "context_usefulness": 5,
  "context_specificity": 5,
  "exact_answer_present": 1,
  "noise_present": 0,
  "misleading_context_present": 0,
  "rationale": "Retrieved context directly contains the expected compatibility handling pattern and is specific to the product family."
}
```

Example B - useful but not exact:

```json
{
  "context_usefulness": 4,
  "context_specificity": 3,
  "exact_answer_present": 0,
  "noise_present": 0,
  "misleading_context_present": 0,
  "rationale": "Retrieved context does not contain the exact product answer, but it gives relevant care and verification guidance for this product family."
}
```

Example C - weak/noisy:

```json
{
  "context_usefulness": 2,
  "context_specificity": 1,
  "exact_answer_present": 0,
  "noise_present": 1,
  "misleading_context_present": 0,
  "rationale": "Most retrieved items are from unrelated product families and provide little actionable support guidance."
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

Method:

```text
{{method}}
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
  "method": "reference_rag | morag",
  "judge_model": "string",
  "context_usefulness": 1,
  "context_specificity": 1,
  "exact_answer_present": 0,
  "noise_present": 0,
  "misleading_context_present": 0,
  "rationale": "short explanation"
}
```
