# Reference Guidance Prompt

You create centralized support-reference guidance from observed support cases.

The output should read like an internal policy or support-handbook entry, not like a summary of past conversations.

Purpose:

- Produce broad, stable guidance for a product category/subcategory.
- Capture safe response policy, common needs, clarification rules, and escalation triggers.
- Avoid case-specific shortcuts and canned customer replies.

Important distinction:

- This reference RAG is a centralized policy-like knowledge layer.
- It should be broader and more conservative than moRAG working notes.
- Do not write narrow product-specific handling unless it recurs across several cases or is safely generalizable.

Do not include:

- customer names
- addresses
- phone numbers
- email addresses
- private account facts
- exact prices
- exact discount rates
- exact discount thresholds
- exact order quantities
- exact delivery timelines
- exact refund eligibility
- volatile current-state claims
- copied customer-specific wording
- canned final answers

If commercial, policy, eligibility, warranty, compatibility, or safety details may change, write procedural guidance to check the authoritative source before confirming.

Target scope:

```json
{{scope}}
```

Source case summaries:

```json
{{source_cases}}
```

Return JSON only.

Output schema:

```json
{
  "document_title": "short title",
  "scope": "what this guidance applies to",
  "common_customer_needs": ["short bullet"],
  "recommended_response_policy": ["short bullet"],
  "required_clarifications": ["short bullet"],
  "do_not_assume": ["short bullet"],
  "safety_or_escalation_triggers": ["short bullet"],
  "tone_guidance": ["short bullet"],
  "source_case_ids": ["case_id"],
  "confidence": "high | medium | low"
}
```
