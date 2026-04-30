# moRAG Evaluation Rubric

## Dataset

Primary dataset:

- `sutro/synthetic-customer-support-dialogues-20k`
- Hugging Face: https://hf.co/datasets/sutro/synthetic-customer-support-dialogues-20k
- License: MIT

Reason for selection:

- Customer-support dialogue format fits the moRAG use case.
- Synthetic data reduces privacy risk for a public GitHub/LinkedIn demo.
- Dialogue-level fields can support memory extraction, raw RAG retrieval, and future-query evaluation.
- The dataset is large enough for experimentation but can be sampled down for a small demo.

## Evaluation Setup

Compare three methods:

1. No memory
   - Answer using only the current future query.

2. Raw conversation RAG
   - Retrieve relevant raw dialogue chunks or conversations.
   - Generate an answer using the retrieved raw context.

3. moRAG
   - Extract structured memory candidates from prior conversations.
   - Approve/edit/reject candidates.
   - Retrieve only approved memory notes.
   - Generate an answer using the retrieved approved memories.

## Memory Safety Policy

moRAG must be implemented as candidate memory with rejection gates. The agent may propose memory candidates, but it must not be allowed to save arbitrary facts as approved memory.

### Hard Exclusion Rules

The following should never become approved moRAG memory:

- addresses
- phone numbers
- email addresses
- account numbers
- payment details
- balances
- transaction amounts
- authentication details
- passwords, tokens, or OTPs
- government IDs
- sensitive health, medical, legal, or financial details unless explicitly in scope and approved
- raw personal identifiers unless necessary, permitted, and explicitly approved

If a candidate contains excluded information, it should be rejected or transformed into a safe procedural memory.

Example rejected memory:

```text
Customer shipping address is 42 Green Street.
```

Example safe procedural memory:

```text
Shipping addresses must be fetched from the authoritative account system at time of use.
```

### Authoritative Source Rules

moRAG should not store volatile facts that belong in systems of record.

Do not store:

- current balance
- open invoice amount
- subscription status
- delivery address
- current plan
- refund eligibility
- account owner
- active discount
- current shipment status

Instead, store procedural guidance about where the value should be checked.

Bad memory:

```text
Customer has an outstanding balance of $84.20.
```

Good memory:

```text
For balance questions, fetch the current balance from the billing system before answering.
```

Core rule:

```text
moRAG may remember the workflow, not volatile account state.
```

### Restricted Memory Types

For v1, automatically reject memory candidates with these types:

- customer_specific_fact
- account_state
- personal_detail
- financial_value
- authentication_detail
- government_identifier
- private_contact_detail

Allowed or reviewable memory types:

- process_rule
- reusable_solution
- failure_pattern
- escalation_pattern
- product_handling_note
- client_preference, only when non-sensitive and scoped
- workflow_learning
- prompt_learning

### Validation Rules

Daily KB validation is useful for memory candidates that are KB-verifiable, such as:

- product facts
- policy rules
- process rules
- troubleshooting steps
- escalation procedures

For sensitive or volatile values, the rule is stricter:

```text
Never validate and save volatile values. Always fetch them live from the authoritative source.
```

### Evaluation Impact

Unsafe memory behavior should affect both memory pollution and critical failure scoring.

Examples of memory pollution:

- storing an address
- storing a balance
- storing a phone number
- storing an account-specific status
- turning one customer's situation into a general rule
- using stale values from prior conversations

Examples of critical failures:

- answer uses a pre-saved balance instead of fetching current balance
- answer reveals or reuses a private address
- answer treats an old refund eligibility statement as current truth
- answer applies a client-specific preference outside its scope

Public positioning:

```text
moRAG is not a place to store private account state. It stores governed operational lessons, and volatile or sensitive facts must remain in authoritative systems.
```

## Response Quality Metrics

### Relevance

Measures whether the answer directly addresses the future query.

Scoring guidance:

- High: Directly answers the query and stays on task.
- Medium: Mostly answers the query but includes some drift.
- Low: Misses the main user need or answers a different question.

### Specificity

Measures whether the answer includes useful support-specific details such as issue type, product context, resolution pattern, escalation path, or known customer handling preference.

Scoring guidance:

- High: Includes concrete details that would help a support agent act.
- Medium: Includes some detail but remains partly generic.
- Low: Mostly generic advice.

### Faithfulness

Measures whether the answer is supported by the retrieved/source evidence and does not contradict it.

Scoring guidance:

- High: Claims are supported by the available evidence.
- Medium: Mostly supported, with minor unsupported assumptions.
- Low: Contains unsupported or contradictory claims.

### Groundedness

Measures whether the answer visibly uses retrieved context rather than producing plausible generic guidance.

Scoring guidance:

- High: Clearly incorporates relevant retrieved memories or raw dialogue evidence.
- Medium: Uses retrieved context weakly or partially.
- Low: Could have been written without the retrieved context.

### Critical Failure Rate

Tracks serious failures that should override otherwise good scores.

Critical failures include:

- Incorrect troubleshooting instruction.
- Unsafe escalation advice.
- Unsupported policy or refund claim.
- Misuse of customer-specific information.
- Treating a soft preference as a hard rule.
- Contradicting a known resolution path.

Report as:

- `0`: no critical failure
- `1`: critical failure present

### Tone

Measures whether the answer is appropriate for customer support.

Scoring guidance:

- High: Clear, calm, helpful, and not overconfident.
- Medium: Acceptable but slightly vague, rigid, or overly verbose.
- Low: Confusing, dismissive, overconfident, or poorly suited to support use.

## Retrieval Metrics

### moRAG Hit Rate

Measures whether moRAG retrieval returns at least one expected relevant approved memory for a future query.

Report as:

- `1`: at least one expected relevant memory retrieved
- `0`: no expected relevant memory retrieved

### Raw RAG Hit Rate

Measures whether raw conversation RAG returns at least one expected relevant raw dialogue chunk or conversation for a future query.

Report as:

- `1`: at least one expected relevant raw item retrieved
- `0`: no expected relevant raw item retrieved

### Retrieval Quality

Measures the usefulness of retrieved items, separate from simple hit rate.

Consider:

- Relevance to the future query.
- Lack of irrelevant noise.
- Usefulness for generating an answer.
- Source clarity and auditability.

Scoring guidance:

- High: Retrieved items are mostly relevant, compact, and useful.
- Medium: Some relevant items are present, but with noticeable noise.
- Low: Retrieved items are mostly irrelevant or too noisy to use.

### Memory Pollution Rate

Measures the percentage of approved or retrieved moRAG memories that are bad memories.

Bad memories include notes that are:

- Wrong.
- Stale.
- Overgeneralized.
- Irrelevant.
- Unsafe.
- Too vague to be useful.
- Unsupported by source evidence.

Formula:

```text
memory_pollution_rate = bad_memories / total_reviewed_or_retrieved_memories
```

## Efficacy Metrics

### Time to Output

Measures end-to-end latency for each method.

Recommended logging:

- retrieval latency
- answer generation latency
- total latency

### Token Consumption

Measures model token usage for each answer.

Recommended logging:

- answer generation input tokens
- answer generation output tokens
- total answer generation tokens

### Retrieved Context Size

Measures how much context is sent to the answer generator.

Recommended logging:

- number of retrieved items
- estimated retrieved-context tokens
- average retrieved item length

## LLM-as-Judge Setup

Use two judge models:

- Primary judge: `gpt-5.5`
- Secondary judge: `gpt-5.4`

Judge response quality and retrieval quality separately.

Recommended judge output schema:

```json
{
  "query_id": "string",
  "method": "no_memory | raw_rag | morag",
  "judge_model": "string",
  "relevance": 1,
  "specificity": 1,
  "faithfulness": 1,
  "groundedness": 1,
  "tone": 1,
  "critical_failure": 0,
  "retrieval_quality": 1,
  "overall_score": 1,
  "rationale": "short explanation"
}
```

Use a consistent numeric scale, such as 1-5, for all scored metrics.

## Public Reporting Priorities

For the GitHub README and LinkedIn post, lead with:

1. Answer specificity.
2. Faithfulness and groundedness.
3. Retrieval quality.
4. Token/context reduction.
5. Critical failure rate.

Avoid large claims such as "moRAG beats RAG."

Preferred claim:

```text
In support-style workflows, approved atomic memory notes can produce more precise, auditable future answers than raw chat-history retrieval.
```

## Cost and Latency Hypothesis

moRAG may reduce downstream retrieval and context costs by replacing noisy raw-history retrieval with compact approved memory notes.

This is a hypothesis to test, not an assumed result.

Expected benefits:

- Smaller retrieval index than raw chunked conversations.
- Shorter retrieved context.
- Lower answer-generation input tokens.
- Potentially faster retrieval.
- Cleaner evidence for audit and review.

Expected tradeoffs:

- Upfront memory extraction cost.
- Review and governance cost.
- Maintenance cost for expiry, validation, conflicts, and supersession.
- Possible need to retain raw logs for audit.
