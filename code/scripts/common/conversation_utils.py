"""Conversation-turn normalization helpers."""

from __future__ import annotations

import ast
import re
from typing import Any


AGENT_PREFIXES = (
    "customer support agent",
    "support agent",
    "agent",
)

CUSTOMER_PREFIXES = (
    "customer",
    "client",
    "user",
)

DEFAULT_CONVERSATION_CONFIG = {
    "agent_speaker_labels": list(AGENT_PREFIXES),
    "customer_speaker_labels": list(CUSTOMER_PREFIXES),
    "unknown_speaker_policy": "customer",
    "unlabeled_turn_policy": "alternate_customer_first",
}


def strip_speaker_prefix(text: str) -> tuple[str | None, str]:
    match = re.match(r"^\s*([^:]{1,80}):\s*(.*)$", text, flags=re.DOTALL)
    if not match:
        return None, text.strip()
    return match.group(1).strip(), match.group(2).strip()


def conversation_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONVERSATION_CONFIG)
    if overrides:
        config.update(overrides)

    if config["unknown_speaker_policy"] not in {"customer", "agent"}:
        raise ValueError(
            "unknown_speaker_policy must be one of: customer, agent"
        )
    if config["unlabeled_turn_policy"] not in {
        "alternate_customer_first",
        "alternate_agent_first",
        "customer",
        "agent",
    }:
        raise ValueError(
            "unlabeled_turn_policy must be one of: "
            "alternate_customer_first, alternate_agent_first, customer, agent"
        )

    return config


def classify_speaker(
    source_label: str | None,
    turn_index: int,
    config: dict[str, Any] | None = None,
) -> str:
    settings = conversation_config(config)
    agent_labels = [label.casefold() for label in settings["agent_speaker_labels"]]
    customer_labels = [label.casefold() for label in settings["customer_speaker_labels"]]

    if source_label:
        normalized = source_label.casefold()
        if any(label in normalized for label in agent_labels):
            return "agent"
        if any(label in normalized for label in customer_labels):
            return "customer"
        return settings["unknown_speaker_policy"]

    unlabeled_policy = settings["unlabeled_turn_policy"]
    if unlabeled_policy == "alternate_customer_first":
        return "customer" if turn_index % 2 == 0 else "agent"
    if unlabeled_policy == "alternate_agent_first":
        return "agent" if turn_index % 2 == 0 else "customer"
    if unlabeled_policy in {"customer", "agent"}:
        return unlabeled_policy
    raise ValueError(f"Unsupported unlabeled_turn_policy: {unlabeled_policy}")


def coerce_dialogue_turns(dialogue: Any) -> list[str]:
    if isinstance(dialogue, list):
        return [str(item) for item in dialogue]
    if isinstance(dialogue, tuple):
        return [str(item) for item in dialogue]

    text = str(dialogue).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            if len(parsed) > 1 or "\n" not in text:
                return [str(item) for item in parsed]
    except (SyntaxError, ValueError):
        pass

    if text.startswith("[") and text.endswith("]"):
        line_items = []
        for line in text[1:-1].splitlines():
            item = line.strip().rstrip(",").strip()
            if len(item) >= 2 and item[0] == item[-1] and item[0] in {"'", '"'}:
                line_items.append(item[1:-1])
        if line_items:
            return [item.replace("\\n", "\n") for item in line_items]

        quoted_items = [
            match.group(2)
            for match in re.finditer(r"(['\"])(.*?)(?<!\\)\1", text, flags=re.DOTALL)
        ]
        if quoted_items:
            return [item.replace("\\n", "\n") for item in quoted_items]

    return [line.strip() for line in text.splitlines() if line.strip()]


def normalize_turns(
    dialogue: Any,
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    customer_count = 0
    agent_count = 0

    for index, raw_text in enumerate(coerce_dialogue_turns(dialogue)):
        source_label, clean_text = strip_speaker_prefix(raw_text)
        speaker = classify_speaker(source_label, index, config)

        if speaker == "customer":
            customer_count += 1
            turn_label = f"cust-comment-{customer_count}"
        else:
            agent_count += 1
            turn_label = f"agent-response-{agent_count}"

        turns.append(
            {
                "turn_index": index + 1,
                "speaker": speaker,
                "turn_label": turn_label,
                "source_speaker_label": source_label,
                "text": clean_text,
            }
        )

    return turns


def render_turns(turns: list[dict[str, Any]]) -> str:
    return "\n".join(f"{turn['turn_label']}: {turn['text']}" for turn in turns)


def first_customer_turn(turns: list[dict[str, Any]]) -> dict[str, Any] | None:
    for turn in turns:
        if turn["speaker"] == "customer":
            return turn
    return None
