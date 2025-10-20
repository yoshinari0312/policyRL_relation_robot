"""Utility to score robot utterances against conversation context using Ollama LLM."""
from __future__ import annotations

import json
import os
import re
import textwrap
from typing import Any, Dict, List, Tuple

import requests

from config import get_config

_CFG = get_config()
_OLLAMA_MODEL = getattr(_CFG.ollama, "model_rl", None) or _CFG.ollama.model or "gpt-oss:20b"
_GEN_OPTS = getattr(_CFG.ollama, "gen_options", None)
_NUM_CTX = getattr(_GEN_OPTS, "num_ctx", None) or 2048

_SCORE_PATTERN = re.compile(r"score" r"\s*" r"[:=]" r"\s*(?P<val>10|[1-9])")
_INT_PATTERN = re.compile(r"\b(10|[1-9])\b")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _select_reward_base() -> str:
    candidates = [
        os.getenv("OLLAMA_REWARD_BASE"),
        os.getenv("OLLAMA_SCORER_BASE"),
    ]
    bases_env = os.getenv("OLLAMA_BASES")
    if bases_env:
        parts = [p.strip() for p in bases_env.split(",") if p.strip()]
        if parts:
            candidates.append(parts[0])
    candidates.append(os.getenv("OLLAMA_BASE"))

    cfg_candidates = [
        getattr(_CFG.ollama, "reward_base", None),
        getattr(_CFG.ollama, "scorer_base", None),
    ]
    bases_cfg = getattr(_CFG.ollama, "bases", None)
    if isinstance(bases_cfg, (list, tuple)) and bases_cfg:
        cfg_candidates.append(bases_cfg[0])
    cfg_candidates.append(getattr(_CFG.ollama, "base_url", None))

    for cand in candidates + cfg_candidates:
        if cand:
            return str(cand).rstrip("/")
    return "http://127.0.0.1:11434"


def _build_prompt(logs: List[Dict[str, Any]], personas: List[str], robot_idx: int) -> str:
    persona_lines = "\n".join(personas) if personas else "(指定なし)"
    history_lines = []
    for idx, entry in enumerate(logs, start=1):
        speaker = entry.get("speaker", "?")
        utterance = entry.get("utterance", "").strip()
        history_lines.append(f"{idx}. [{speaker}] {utterance}")
    history = "\n".join(history_lines)
    robot_utt = logs[robot_idx].get("utterance", "").strip()

    prompt = textwrap.dedent(
        f"""
        あなたは会話の評価専門家です。以下の人物設定と会話履歴を踏まえ、最新のロボット発話が文脈に沿っているか評価してください。

        会話履歴（時系列順）:
        <<<
        {history}
        <<<

        評価対象のロボット発話:
        <<<
        {robot_utt}
        <<<

        評価基準:
        - スコアは 1 から 10 の整数
        - 10: 文脈に完全に一致している
        - 6-9: 概ね文脈に沿っている
        - 2-5: 一部ちぐはぐだが致命的ではない
        - 1: 文脈から逸脱し、意味が通らない

        出力形式（JSON のみ。余計な文章は禁止）:
        {{"score": <1から10の整数>"}}

        上記 JSON だけを出力してください。
        """
    ).strip()
    return prompt


def _parse_response(text: str) -> Tuple[float, str]:
    raw = text.strip()
    if not raw:
        return 0.0, "empty"

    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            score_val = payload.get("score")
            if isinstance(score_val, str):
                score_val = score_val.strip()
                if score_val.isdigit():
                    score_val = int(score_val)
            if isinstance(score_val, (int, float)):
                return float(score_val), "json"
        except json.JSONDecodeError:
            pass

    score_match = _SCORE_PATTERN.search(raw)
    if score_match:
        val = float(score_match.group("val"))
        return val, "regex_label"

    int_match = _INT_PATTERN.search(raw)
    if int_match:
        val = float(int_match.group(0))
        return val, "regex_fallback"

    return 0.0, "unparsed"


def score_context_alignment(
    logs: List[Dict[str, Any]],
    personas: List[str],
    debug: bool = False,
) -> Tuple[float, Dict[str, Any]]:
    """Return (score, details) where score is a float within 0-10."""
    details: Dict[str, Any] = {
        "model": _OLLAMA_MODEL,
    }
    if not logs:
        details["error"] = "empty_logs"
        return 0.0, details

    robot_indices = [idx for idx in range(len(logs) - 1, -1, -1) if logs[idx].get("speaker") == "ロボット"]
    if not robot_indices:
        details["error"] = "no_robot_utterance"
        return 0.0, details
    robot_idx = robot_indices[0]

    prompt = _build_prompt(logs, personas, robot_idx)
    base = _select_reward_base()
    url = f"{base}/api/generate"
    payload: Dict[str, Any] = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.0,
            "num_ctx": _NUM_CTX,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=180)
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            data = {"response": response.text}
    except Exception as exc:  # pragma: no cover - network failure
        details.update({
            "error": str(exc),
            "base": base,
        })
        return 0.0, details

    raw_text = str(data.get("response", "")).strip()
    score_val, source = _parse_response(raw_text)
    score_val = _clamp(score_val, 0.0, 10.0)

    details.update({
        "base": base,
        "raw_response": raw_text,
        "parse_source": source,
    })
    if debug:
        details["prompt_tail"] = prompt[-1200:]
    return score_val, details