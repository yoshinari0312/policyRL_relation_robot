#!/usr/bin/env python3
"""Interactive CLI to inspect cosine similarity between human and robot utterances.

This utility reuses the Azure OpenAI embedding configuration defined in
`config.local.yaml`. It will prompt for the two utterances (unless supplied via
command-line arguments), fetch embeddings from Azure, and report cosine
similarity plus raw vector norms for additional debugging.
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional

from config import get_config

try:
    from openai import AzureOpenAI  # type: ignore
except ImportError as exc:  # pragma: no cover - optional dependency
    print(
        "[error] Missing Azure OpenAI dependencies. Install them with "
        "`pip install openai`.\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    if len(vec1) != len(vec2):
        raise ValueError("Embedding vectors must have the same length")

    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def get_azure_embedding_client():
    cfg = get_config()
    openai_cfg = cfg.openai

    required_fields = {
        "azure_endpoint": openai_cfg.azure_endpoint,
        "azure_api_key": openai_cfg.azure_api_key,
        "azure_api_version": openai_cfg.azure_api_version,
        "embedding_deployment": openai_cfg.embedding_deployment,
    }
    missing = [name for name, value in required_fields.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing Azure embedding settings in config.local.yaml: " + joined
        )

    client = AzureOpenAI(
        api_version=openai_cfg.azure_api_version,
        azure_endpoint=openai_cfg.azure_endpoint,
        api_key=openai_cfg.azure_api_key,
    )
    return client, openai_cfg.embedding_deployment


def fetch_embeddings(client: AzureOpenAI, deployment: str, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    response = client.embeddings.create(input=texts, model=deployment)
    # Ensure the order matches the input order (API contract)
    vectors = [item.embedding for item in response.data]
    return vectors


def prompt_if_missing(prompt: str, value: Optional[str]) -> str:
    if value:
        return value
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def main():
    parser = argparse.ArgumentParser(description="Evaluate cosine similarity using Azure embeddings")
    parser.add_argument("--human", help="Human utterance to compare")
    parser.add_argument("--robot", help="Robot utterance to compare")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single comparison and exit (default is interactive loop)",
    )
    args = parser.parse_args()

    try:
        client, deployment = get_azure_embedding_client()
    except Exception as exc:
        print(f"[error] Failed to initialize Azure embedding client: {exc}", file=sys.stderr)
        sys.exit(1)

    def run_single(human_text: str, robot_text: str) -> None:
        if not human_text or not robot_text:
            print("[warn] Both human and robot utterances must be non-empty.")
            return
        try:
            human_vec, robot_vec = fetch_embeddings(client, deployment, [human_text, robot_text])
        except Exception as exc:
            print(f"[error] Failed to fetch embeddings: {exc}")
            return
        similarity = cosine_similarity(human_vec, robot_vec)
        human_norm = math.sqrt(sum(x * x for x in human_vec))
        robot_norm = math.sqrt(sum(x * x for x in robot_vec))
        print("--- Similarity Report ---")
        print(f"Human : {human_text}")
        print(f"Robot : {robot_text}")
        print(f"Cosine similarity : {similarity:.6f}")
        print(f"Human vector norm : {human_norm:.6f}")
        print(f"Robot vector norm : {robot_norm:.6f}")
        print(f"Embedding length   : {len(human_vec)}")
        print("-----------------------")

    if args.once:
        human_text = prompt_if_missing("[Input] Human utterance: ", args.human)
        robot_text = prompt_if_missing("[Input] Robot utterance: ", args.robot)
        run_single(human_text, robot_text)
        return

    print("Enter human/robot utterances to measure similarity. Press Ctrl-D or submit empty input to exit.")
    while True:
        human_text = prompt_if_missing("[Input] Human utterance: ", args.human)
        if not human_text:
            break
        robot_text = prompt_if_missing("[Input] Robot utterance : ", args.robot)
        if not robot_text:
            break
        run_single(human_text, robot_text)
        # Reset CLI arguments after the first iteration (if provided)
        args.human = None
        args.robot = None
        print()


if __name__ == "__main__":
    main()
