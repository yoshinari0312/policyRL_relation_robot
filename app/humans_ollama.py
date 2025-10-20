import os
import re
import time
import requests
from typing import List, Dict
from config import get_config

# モデルやベースURLは環境変数で切り替え可能
_CFG = get_config()

cfg_bases = list(getattr(_CFG.ollama, "bases", []) or [])
cfg_default_base = getattr(_CFG.ollama, "base_url", "http://ollama_a:11434")

# 1つだけ指定の場合（従来互換）
OLLAMA_BASE = os.getenv("OLLAMA_BASE") or (cfg_bases[0] if cfg_bases else cfg_default_base)

# 人間を複数GPUに振り分けるため、カンマ区切りの複数エンドポイントも受け付け
env_bases = os.getenv("OLLAMA_BASES")
if env_bases:
    base_candidates = env_bases.split(",")
elif cfg_bases:
    base_candidates = cfg_bases
else:
    base_candidates = [OLLAMA_BASE]
_BASES = [b.strip() for b in base_candidates if b and b.strip()]

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or getattr(_CFG.ollama, "model", "gpt-oss:20b")
GEN_OPTS     = _CFG.ollama.gen_options

_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")


def _contains_hiragana(text: str) -> bool:
    return bool(text) and bool(_HIRAGANA_RE.search(text))

def _post_ollama(prompt: str, model: str = OLLAMA_MODEL, base: str = None, timeout: int = 60) -> str:
    """Ollama /api/generate を叩いて一括応答を返す（ストリームなし）。簡易リトライ付き。"""
    base = base or OLLAMA_BASE
    url = f"{base}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        # 出力を短く・丁寧に
        "options": {
            "temperature": GEN_OPTS.temperature,
            "top_p": GEN_OPTS.top_p,
            "num_ctx": GEN_OPTS.num_ctx
        }
    }
    for attempt in range(1,5):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except Exception as e:
            status = getattr(e.response, "status_code", None)
            text   = getattr(e.response, "text", "")
            # モデル未導入の典型パターン: 404 + "model not found" 等
            if status == 404 and any(s in text.lower() for s in ["model", "not", "found", "missing"]):
                # 1回だけ pull を試みる（失敗しても次のリトライへ）
                try:
                    pull = requests.post(
                        f"{base}/api/pull", json={"name": model, "stream": False}, timeout=600
                    )
                    # 200/201 以外でも次のリトライに委ねる
                except Exception:
                    pass
            if attempt == 4:
                raise
            backoff = min(20.0, 2.0 ** max(0, attempt - 1))
            time.sleep(backoff)
        except requests.RequestException:
            if attempt == 4:
                raise
            backoff = min(20.0, 2.0 ** max(0, attempt - 1))
            time.sleep(backoff)
    return ""

def build_persona_prompt(persona: str, logs: List[Dict]) -> str:
    """
    persona 例:
      "A:穏健で協調的" / "B:反論しがちで論理的" / "C:静かだが核心を突く"
    logs: [{"speaker": "...", "utterance": "..."}]
    """
    history = "\n".join([f"[{L['speaker']}] {L['utterance']}" for L in logs[-10:]])
    name, trait = persona.split(":", 1) if ":" in persona else (persona, "")
    sys = (
        f"あなたは会話参加者の一人「{name}」です。性格・口調の特徴は「{trait}」です。\n"
        "以下の会話履歴に続けて、あなたの発言を一文で返してください。\n"
        "会話の流れに沿い、他の参加者の発言に配慮しつつ、あなたらしい口調で答えてください。\n"
        "会話の開始時点では、他の参加者がまだ発言していない場合もあります。その場合は何かしら話題を提供してください。\n"
        "・自己紹介や役割説明は不要です。\n"
        "・発言は一文で簡潔にしてください。\n"
    )
    prompt = sys + "\n--- 会話履歴 ---\n" + (history or "(まだ発言なし)") + "\n\n[あなたの返答]"
    return prompt

def human_reply(logs: List[Dict], personas: List[str]) -> List[Dict]:
    """複数ペルソナの“人間側”発話を返す。

    同一ターン内で後続の人が直前の発言を参照できるように、返信を逐次蓄積する。
    """
    replies = []
    working_logs: List[Dict] = [dict(item) for item in logs]
    max_attempts = 3
    fallback_text = "すみません、落ち着いて話を続けましょう。"

    for idx, persona in enumerate(personas):
        base = _BASES[idx % len(_BASES)]
        prompt_logs = working_logs
        speaker = persona.split(":", 1)[0]

        text = ""
        for attempt in range(1, max_attempts + 1):
            candidate = _post_ollama(build_persona_prompt(persona, prompt_logs), base=base)
            if _contains_hiragana(candidate):
                text = candidate
                break
            if attempt < max_attempts and _CFG.env.debug:
                print(f"[human_reply] retrying for {speaker}: missing hiragana (attempt {attempt})")
            text = candidate

        if not _contains_hiragana(text):
            text = fallback_text

        if not text:
            text = fallback_text

        reply = {"speaker": speaker, "utterance": text}
        replies.append(reply)
        working_logs.append(reply)
    return replies
