"""Shared helpers for the moRAG prototype scripts."""

from __future__ import annotations

import json
import importlib.util
import os
import pathlib
import re
import tomllib
from typing import Any

from openai import BadRequestError, OpenAI


def load_toml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_json(path: pathlib.Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_prompt(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def render_template(template: str, replacements: dict[str, Any]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    unresolved = sorted(set(re.findall(r"\{\{\s*([^{}]+?)\s*\}\}", rendered)))
    if unresolved:
        raise ValueError(f"Unresolved prompt template placeholders: {', '.join(unresolved)}")
    return rendered


def openai_client_from_config(model_config: dict[str, Any]) -> OpenAI:
    env_name = model_config["secrets"]["openai_api_key_env"]
    api_key = os.environ.get(env_name)
    if not api_key:
        raise RuntimeError(f"Missing OpenAI API key environment variable: {env_name}")
    return OpenAI(api_key=api_key)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a model response that should contain a single JSON object."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def response_text(response: Any) -> str:
    """Return text from Responses API objects across SDK versions."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)


def usage_dict(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in dir(usage)
        if not key.startswith("_") and isinstance(getattr(usage, key), (int, float, str))
    }


def call_json_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
        "text": {"format": {"type": "json_object"}},
    }
    try:
        response = client.responses.create(**request)
    except BadRequestError as error:
        if "temperature" not in str(error):
            raise
        request.pop("temperature", None)
        response = client.responses.create(**request)
    return extract_json_object(response_text(response)), usage_dict(response)


def call_text_model(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
) -> tuple[str, dict[str, Any]]:
    request = {
        "model": model,
        "input": prompt,
        "temperature": temperature,
    }
    try:
        response = client.responses.create(**request)
    except BadRequestError as error:
        if "temperature" not in str(error):
            raise
        request.pop("temperature", None)
        response = client.responses.create(**request)
    return response_text(response), usage_dict(response)
