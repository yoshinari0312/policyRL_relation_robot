"""Generate sample dialogues using a trained PPO policy model."""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, Optional

FastLanguageModel = None  # type: ignore
_UNSLOTH_AVAILABLE = False

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

from config import get_config
from env.convo_env import ConversationEnv
from ppo_train import build_robot_messages
from network_metrics import BALANCED, UNBALANCED

_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_THINK_CLOSE_TAG = "</think>"
def _strip_think_tags(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    end_idx = lowered.rfind(_THINK_CLOSE_TAG)
    if end_idx != -1:
        return text[end_idx + len(_THINK_CLOSE_TAG):].lstrip()
    return re.sub(r"<\s*/?\s*think\s*>", "", text, flags=re.IGNORECASE).strip()


def _ensure_unsloth_loaded() -> bool:
    global FastLanguageModel, _UNSLOTH_AVAILABLE
    if _UNSLOTH_AVAILABLE:
        return True
    try:
        import unsloth  # type: ignore  # noqa: F401
        from unsloth import FastLanguageModel as _FastLanguageModel  # type: ignore
    except ImportError:
        return False
    FastLanguageModel = _FastLanguageModel  # type: ignore
    _UNSLOTH_AVAILABLE = True
    return True


def _text_similarity(lhs: str, rhs: str) -> float:
    if not lhs or not rhs:
        return 0.0
    return SequenceMatcher(None, lhs, rhs).ratio()


def _has_hiragana(text: str) -> bool:
    return bool(_HIRAGANA_RE.search(text))


def _extract_primary_japanese_sentence(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

    # locate the span that actually contains Japanese characters
    first = _JAPANESE_CHAR_RE.search(normalized)
    if not first:
        return ""

    last = first
    for match in _JAPANESE_CHAR_RE.finditer(normalized):
        last = match

    start = first.start()
    end = last.end()

    if start > 0 and normalized[start - 1].isascii() and normalized[start - 1].isalnum():
        i = start
        while i > 0 and normalized[i - 1].isascii() and normalized[i - 1].isalnum():
            i -= 1
        prefix = normalized[i:start]
        if len(prefix) == 1 and prefix.isalpha() and prefix.isupper():
            start = i

    # include trailing punctuation immediately following the final japanese character
    while end < len(normalized) and normalized[end] in "。．！？!?……〜ー】〕））」』'\"”] )":
        end += 1

    candidate = normalized[start:end].strip()
    candidate = re.sub(r"^\s*\[([^\]]+)\]\s*", r"\1", candidate)
    candidate = re.sub(r"^\s*([A-Za-zＡ-Ｚぁ-んァ-ン一-龥]+)\s*[:：]\s*", "", candidate)
    return candidate.strip(' "“”『』「」[]()')


_SENTENCE_ENDINGS = ["。", "！", "？", "．", "!", "?"]


def _enforce_single_sentence(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    cleaned = re.sub(r"^\s*\[([^\]]+)\]\s*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[\-‐‑‒–—―ー〜~]*\s*([A-Za-zＡ-Ｚ])\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*([A-Za-zＡ-Ｚぁ-んァ-ン一-龥]+)\s*[:：]\s*", "", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return cleaned
    if not cleaned.endswith(tuple(_SENTENCE_ENDINGS)):
        cleaned = cleaned.rstrip(" 　。、！？」") + "。"
    return cleaned


def _map_device(map_arg: str | None):
    if map_arg is None:
        return "auto"
    if map_arg.lower() in {"cpu", "cuda"}:
        return map_arg.lower()
    try:
        # allow "0" "0,1" etc
        indices = [int(x.strip()) for x in map_arg.split(",") if x.strip()]
        return {"": indices[0]} if len(indices) == 1 else {str(i): i for i in indices}
    except Exception:
        raise ValueError(f"Unrecognised device map argument: {map_arg}")


def _triangle_status(struct: str | None) -> str:
    if struct in BALANCED:
        return "balanced"
    if struct in UNBALANCED:
        return "unbalanced"
    return "unknown"


def _format_step(step_data: Dict[str, Any]) -> str:
    lines = [
        f"Step {step_data['step']} | reward={step_data['reward']:+.3f} | done={step_data['done']}"
    ]
    for reply in step_data["humans"]:
        lines.append(f"  {reply['speaker']}: {reply['utterance']}")
    lines.append(f"  Robot : {step_data['robot']}")
    if step_data.get("reward_breakdown"):
        rb = step_data["reward_breakdown"]
        lines.append(
            "  Reward -> R_tri={:+.3f}, R_iso={:+.3f}, R_all={:+.3f}".format(
                rb.get("R_tri", 0.0), rb.get("R_iso", 0.0), rb.get("R_all", 0.0)
            )
        )
        if rb.get("human_iso_count"):
            lines.append(f"    isolated humans: {rb['human_iso_count']}")
    lines.append("  Triads:")
    for tri in step_data["triangles"]:
        nodes = ",".join(tri.get("nodes", []))
        lines.append(f"    [{nodes}] -> {tri['struct']} ({tri['status']})")
    return "\n".join(lines)


def _json_safe(obj):
    if isinstance(obj, dict):
        safe = {}
        for key, value in obj.items():
            if isinstance(key, (int, float, bool)) or key is None:
                safe_key = key
            elif isinstance(key, str):
                safe_key = key
            elif isinstance(key, (tuple, list, set)):
                safe_key = ",".join(str(x) for x in key)
            else:
                safe_key = str(key)
            safe[safe_key] = _json_safe(value)
        return safe
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(item) for item in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def run_episode(
    model,
    tokenizer,
    env: ConversationEnv,
    generation_kwargs: Dict[str, Any],
    similarity_threshold: float = 0.9,
    similarity_max_attempts: int = 3,
    base_model_id: Optional[str] = None,
):
    env.reset()
    active_personas = list(env.personas)
    if active_personas:
        print("Active personas:")
        for persona in active_personas:
            print(f"  {persona}")
        print()
    device = next(model.parameters()).device
    logs = []
    similarity_threshold = max(0.0, min(0.999, similarity_threshold))
    similarity_max_attempts = max(1, int(similarity_max_attempts))
    previous_robot_text = ""
    for step in range(1, env.max_steps + 1):
        env.ensure_human_turn()
        messages = build_robot_messages(env.logs, env.personas, env=env, model_name=base_model_id)
        if getattr(env, "debug", False):
            print("\n[run_robot_inference] === Prompt feed ===")
            print(f"episode={env.episode}, step={step}, total_steps={env.total_steps}")
            print("[run_robot_inference] conversation logs (env.logs):")
            for entry in env.logs:
                speaker = entry.get("speaker", "?")
                utt = entry.get("utterance", "")
                print(f"  [{speaker}] {utt}")
            print("[run_robot_inference] chat_template user payload:")
            try:
                user_payload = next((m for m in messages if m.get("role") == "user"), None)
                if user_payload is not None:
                    print(user_payload.get("content", ""))
                else:
                    print(messages)
            except Exception:
                print(messages)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        robot_text = ""
        language_retries = 0
        similarity_retries = 0
        max_attempts = max(3, similarity_max_attempts)
        last_candidate = ""
        last_candidate_raw = ""
        base_temperature = generation_kwargs.get("temperature")
        base_top_p = generation_kwargs.get("top_p")
        temp_override = base_temperature
        top_p_override = base_top_p
        attempt_raws: list[str] = []
        for attempt in range(1, max_attempts + 1):
            attempt_kwargs = dict(generation_kwargs)
            if temp_override is not None:
                attempt_kwargs["temperature"] = temp_override
            if top_p_override is not None:
                attempt_kwargs["top_p"] = top_p_override
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    **attempt_kwargs,
                )
            gen_ids = generated[0, encoded["input_ids"].shape[-1]:]
            raw_candidate = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            attempt_raws.append(raw_candidate)
            visible_candidate = _strip_think_tags(raw_candidate)
            cleaned_candidate = _extract_primary_japanese_sentence(visible_candidate)
            last_candidate_raw = raw_candidate
            last_candidate = visible_candidate
            if cleaned_candidate:
                candidate_use = cleaned_candidate
            else:
                candidate_use = visible_candidate
            if not _has_hiragana(candidate_use):
                language_retries += 1
                if attempt < max_attempts:
                    print(f"[warn] Non-Japanese output detected (attempt {attempt}). Retrying...")
                    if temp_override is not None:
                        temp_override = min(2.0, temp_override * 1.05)
                    elif base_temperature is not None:
                        temp_override = min(2.0, base_temperature * 1.05)
                    if top_p_override is not None:
                        top_p_override = min(0.98, top_p_override + 0.02)
                    elif base_top_p is not None:
                        top_p_override = min(0.98, base_top_p + 0.02)
                    continue
                else:
                    print("[warn] Non-Japanese output persisted; using fallback candidate.")
            candidate_sentence = _enforce_single_sentence(candidate_use)
            similarity_score = 0.0
            if previous_robot_text:
                similarity_score = _text_similarity(candidate_sentence, previous_robot_text)
            if previous_robot_text and similarity_score >= similarity_threshold and attempt < max_attempts:
                similarity_retries += 1
                if base_temperature is not None:
                    temp_override = min(2.0, base_temperature * (1.05 ** similarity_retries))
                if base_top_p is not None:
                    top_p_override = min(0.98, base_top_p + 0.02 * similarity_retries)
                print(
                    f"[retry] High similarity detected ({similarity_score:.3f} >= {similarity_threshold:.3f}); resampling (attempt {attempt}/{max_attempts})."
                )
                continue
            if previous_robot_text and similarity_score >= similarity_threshold:
                print(
                    f"[warn] High similarity ({similarity_score:.3f}) remains after maximum retries; accepting candidate."
                )
            robot_text = candidate_sentence
            break
        if not robot_text:
            fallback_source = _extract_primary_japanese_sentence(last_candidate)
            if not fallback_source:
                fallback_source = last_candidate
            fallback = _enforce_single_sentence(fallback_source)
            if not _has_hiragana(fallback):
                fallback = "お待たせいたしました、状況を丁寧に整えますね。"
            robot_text = fallback
        state, reward, done, info = env.step(robot_text)
        previous_robot_text = robot_text
        triangles_raw = info.get("rel", {}).get("triangles", [])
        triangles = [
            {
                "nodes": tri.get("nodes", []),
                "struct": tri.get("struct"),
                "status": _triangle_status(tri.get("struct")),
            }
            for tri in triangles_raw
        ]
        logs.append(
            {
                "step": step,
                "reward": float(reward),
                "robot": robot_text,
                "robot_raw": last_candidate_raw,
                "humans": info.get("replies", []),
                "triangles": triangles,
                "env_metrics": info.get("rel", {}),
                "state_after": state,
                "done": bool(done),
                "reward_breakdown": info.get("reward_breakdown", {}),
                "personas": info.get("personas", list(env.personas)),
                "language_retry_count": language_retries,
                "similarity_retry_count": similarity_retries,
                "raw_attempts": list(attempt_raws),
            }
        )
        print(_format_step(logs[-1]))
        print()
        if done:
            break
    return {
        "personas": active_personas,
        "steps": logs,
    }


def _has_lora_adapter(model_dir: Optional[Path]) -> bool:
    if model_dir is None:
        return False
    candidates = [
        model_dir / "adapter_model.safetensors",
        model_dir / "adapter_model.bin",
        model_dir / "pytorch_model.bin",
    ]
    return any(path.exists() for path in candidates)


def load_policy_model(
    model_dir: Optional[Path],
    base_model_id: str,
    device_map,
    use_quant: bool,
    load_adapters: bool = True,
):
    tokenizer_override = None
    model_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    base_lower = base_model_id.lower()

    if "gpt-oss" in base_lower:
        if not _ensure_unsloth_loaded():
            raise ImportError(
                "FastLanguageModel (unsloth) is required to load gpt-oss models. Install via `pip install unsloth`"
            )

        print(f"Loading base policy model via Unsloth FastLanguageModel: {base_model_id}")
        model, tokenizer_override = FastLanguageModel.from_pretrained(
            model_name=base_model_id,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        if use_quant:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_cfg
        else:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["torch_dtype"] = dtype

        print(f"Loading base policy model: {base_model_id}")
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)
        except torch.cuda.OutOfMemoryError as exc:
            diag = []
            if torch.cuda.is_available():
                try:
                    for idx in range(torch.cuda.device_count()):
                        free, total = torch.cuda.mem_get_info(idx)
                        used = total - free
                        diag.append(
                            f"GPU{idx}: used={used / (1024 ** 3):.1f} GiB / total={total / (1024 ** 3):.1f} GiB (free={free / (1024 ** 3):.1f} GiB)"
                        )
                except Exception:
                    pass
            if diag:
                print("[error] CUDA OOM while loading base model. Current usage:")
                for line in diag:
                    print(f"  {line}")
            print("[hint] Free up GPU memory or run with --device-map cpu / switch to a smaller base model.")
            raise

    if load_adapters and _has_lora_adapter(model_dir):
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(model_dir))
            print(f"Loaded LoRA adapter weights from {model_dir}.")
        except Exception as exc:
            print(f"[warn] Failed to load LoRA adapters from {model_dir}: {exc}")
    else:
        if load_adapters and model_dir is not None and (model_dir / "model.safetensors").exists():
            print(
                f"[warn] Found model.safetensors in {model_dir} but no LoRA adapters; this file typically only contains the PPO value head and is ignored for inference."
            )
        elif load_adapters and model_dir is not None:
            print(f"[warn] No adapter weights found in {model_dir}; using base model only.")

    model.eval()
    return model, tokenizer_override


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained PPO robot model")
    parser.add_argument("--model-dir", default=None, help="Path to the saved PPO policy directory")
    parser.add_argument("--base-model-id", default=None, help="Hugging Face model ID to use as the base model")
    parser.add_argument("--base-only", action="store_true", help="Ignore local PPO adapters and generate from the base model")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to simulate")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override max_new_tokens; defaults to a generous value derived from config")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Override top_p")
    parser.add_argument("--device-map", default=None, help="Device map hint (auto/cpu/cuda or indices)")
    parser.add_argument("--save-json", default=None, help="Optional path to store generated dialogue as JSON")
    args = parser.parse_args()

    cfg = get_config()
    ppo_cfg = cfg.ppo

    base_only = bool(args.base_only)

    default_model_dir = None
    try:
        output_dir_cfg = getattr(getattr(cfg, "ppo", object()), "output_dir", None)
        if output_dir_cfg:
            default_model_dir = Path(output_dir_cfg) / "final"
    except Exception:
        default_model_dir = None

    def _candidate_paths(path_like: Path) -> list[Path]:
        candidates = [path_like]
        if not path_like.is_absolute():
            script_dir = Path(__file__).resolve().parent
            workspace_root = script_dir.parent
            candidates.append((script_dir / path_like).resolve())
            candidates.append((workspace_root / path_like).resolve())
        return candidates

    model_dir_input = Path(args.model_dir) if args.model_dir else default_model_dir
    model_dir_candidates: list[Path] = []
    if model_dir_input is not None:
        model_dir_candidates.extend(_candidate_paths(model_dir_input))

    model_dir = None
    if not base_only:
        for cand in model_dir_candidates:
            if cand is not None and cand.exists():
                model_dir = cand
                break
        if model_dir is None:
            tried = [str(c.resolve()) for c in model_dir_candidates]
            raise FileNotFoundError(
                "Model directory not found. Tried: " + (", ".join(tried) if tried else "<none>")
            )

    base_model_id = args.base_model_id or getattr(ppo_cfg, "model_name_or_path", str(model_dir) if model_dir else None)
    if base_model_id is None:
        raise ValueError("Base model ID could not be determined. Set 'ppo.model_name_or_path' in config or pass --base-model-id.")

    device_map = _map_device(args.device_map)
    use_quant = bool(device_map != "cpu" and torch.cuda.is_available())

    model, tokenizer_override = load_policy_model(
        model_dir if not base_only else None,
        base_model_id,
        device_map,
        use_quant,
        load_adapters=not base_only,
    )

    if tokenizer_override is not None:
        tokenizer = tokenizer_override
    else:
        tokenizer_source = base_model_id if (base_only or model_dir is None) else str(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    gen_cfg = GenerationConfig.from_model_config(model.config)
    gen_cfg.pad_token_id = tokenizer.pad_token_id
    gen_cfg.eos_token_id = tokenizer.eos_token_id
    model.generation_config = gen_cfg

    cfg_max_new_tokens = getattr(ppo_cfg, "max_new_tokens", None)
    try:
        cfg_max_new_tokens = int(cfg_max_new_tokens) if cfg_max_new_tokens is not None else None
    except (TypeError, ValueError):
        cfg_max_new_tokens = None
    default_max_new_tokens = 80 if cfg_max_new_tokens is None else max(cfg_max_new_tokens, 80)
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else default_max_new_tokens

    generation_kwargs = dict(
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature if args.temperature is not None else getattr(ppo_cfg, "temperature", 0.7),
        top_p=args.top_p if args.top_p is not None else getattr(ppo_cfg, "top_p", 0.9),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    rep_penalty_cfg = getattr(ppo_cfg, "decode_repetition_penalty", None)
    try:
        repetition_penalty = float(rep_penalty_cfg) if rep_penalty_cfg is not None else 1.18
    except (TypeError, ValueError):
        repetition_penalty = 1.18
    if repetition_penalty and repetition_penalty > 1.0:
        generation_kwargs["repetition_penalty"] = repetition_penalty

    no_repeat_cfg = getattr(ppo_cfg, "decode_no_repeat_ngram_size", None)
    try:
        no_repeat_ngram_size = int(no_repeat_cfg) if no_repeat_cfg is not None else 6
    except (TypeError, ValueError):
        no_repeat_ngram_size = 6
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        generation_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    typical_p_cfg = getattr(ppo_cfg, "decode_typical_p", None)
    try:
        typical_p_val = float(typical_p_cfg) if typical_p_cfg is not None else 0.95
    except (TypeError, ValueError):
        typical_p_val = 0.95
    if 0.0 < typical_p_val < 1.0:
        generation_kwargs["typical_p"] = typical_p_val

    min_p_cfg = getattr(ppo_cfg, "decode_min_p", None)
    try:
        min_p_val = float(min_p_cfg) if min_p_cfg is not None else None
    except (TypeError, ValueError):
        min_p_val = None
    if min_p_val is not None and 0.0 < min_p_val < 1.0:
        generation_kwargs["min_p"] = min_p_val

    env = ConversationEnv(
        max_steps=cfg.env.max_steps,
        personas=cfg.env.personas,
        include_robot=cfg.env.include_robot,
        ema_alpha=cfg.scorer.ema_alpha,
        decay_factor=cfg.scorer.decay_factor,
        backend=cfg.scorer.backend,
        debug=True,
        max_history=cfg.env.max_history,
    )

    all_logs = []
    similarity_threshold = getattr(ppo_cfg, "similarity_retry_threshold", 0.9) or 0.9
    try:
        similarity_threshold = float(similarity_threshold)
    except (TypeError, ValueError):
        similarity_threshold = 0.9
    similarity_max_attempts = getattr(ppo_cfg, "similarity_retry_max_attempts", 3) or 3
    try:
        similarity_max_attempts = int(similarity_max_attempts)
    except (TypeError, ValueError):
        similarity_max_attempts = 3
    for ep in range(1, args.episodes + 1):
        print(f"=== Episode {ep} ===")
        episode_data = run_episode(
            model,
            tokenizer,
            env,
            generation_kwargs,
            similarity_threshold=similarity_threshold,
            similarity_max_attempts=similarity_max_attempts,
            base_model_id=base_model_id,
        )
        all_logs.append({"episode": ep, **episode_data})

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(all_logs), f, ensure_ascii=False, indent=2)
        print(f"Saved dialogue logs to {output_path}")


if __name__ == "__main__":
    main()
