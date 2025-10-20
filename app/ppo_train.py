# 強化学習(PPO)で「ロボットの発言」方策を更新する最小スケルトン
# - TRL PPO + LoRA (PEFT)
# - 既存 env.ConversationEnv を 1 ステップ環境として用い、
#   報酬 = 不安定三角形の減少量 (Before-After) をそのまま使用
# - 収集: 1 ステップ × batch_size サンプルで1回の PPO 更新#
# 依存:
#   pip install "trl>=0.23.0" transformers accelerate peft bitsandbytes datasets
#
# 使い方:
#   python -m ppo_train --dry-run  # まずはCUDAやモデルのロード確認
#   python -m ppo_train            # 学習実行
#
from __future__ import annotations
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")  # 断片化対策
import dataclasses
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from functools import lru_cache
import types
import json
import itertools
from difflib import SequenceMatcher
import atexit
import re
from pathlib import Path
from datetime import datetime

_UNSLOTH_AVAILABLE = False
import math

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import (
    AutoTokenizer,
    TrainerCallback,
)
from accelerate import init_empty_weights, infer_auto_device_map
from accelerate.utils import get_balanced_memory
from peft import LoraConfig
from trl import PPOConfig, PPOTrainer
from trl.models import AutoModelForCausalLMWithValueHead
from transformers import BitsAndBytesConfig, AutoConfig, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader as TorchDataLoader, IterableDataset as TorchIterableDataset

try:
    from openai import AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AzureOpenAI = None

try:
    from datasets import IterableDataset as HFIterableDataset
except ImportError:  # pragma: no cover - optional dependency
    HFIterableDataset = None

from config import get_config
from env.convo_env import ConversationEnv
from network_metrics import BALANCED, UNBALANCED


def _patch_gpt_oss_moe_layers(model_name: str | None = None):
    if not model_name or "gpt-oss" not in model_name.lower():
        return

    """Ensure GPT-OSS MoE experts expose per-expert Linear modules.

    Some quantized checkpoints (e.g. Unsloth 4bit variants) export weights
    as ``experts.down_projs.{i}.weight`` whereas the upstream
    implementation in ``transformers`` still packs all experts into single
    tensors.  We replace the experts implementation with a semantically
    equivalent, but module-list based version so that weight loading works
    out-of-the-box.
    """

    try:
        from transformers.models.gpt_oss import modeling_gpt_oss as gpt_oss_mod
    except Exception:
        return

    if getattr(gpt_oss_mod, "_ppo_patch_moe_applied", False):
        return

    class PatchedGptOssExperts(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.intermediate_size = config.intermediate_size
            self.num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 1))
            self.hidden_size = config.hidden_size
            self.expert_dim = self.intermediate_size
            self.alpha = 1.702
            self.limit = 7.0

            def _make_gate():
                layer = torch.nn.Linear(self.hidden_size, 2 * self.expert_dim, bias=True)
                return layer

            def _make_down():
                layer = torch.nn.Linear(self.expert_dim, self.hidden_size, bias=True)
                return layer

            self.gate_up_projs = torch.nn.ModuleList(_make_gate() for _ in range(self.num_experts))
            self.down_projs = torch.nn.ModuleList(_make_down() for _ in range(self.num_experts))

        def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
            original_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, self.hidden_size)

            if routing_weights is None:
                routing_weights = hidden_states.new_ones(hidden_states.shape[0], self.num_experts)
                routing_weights /= float(self.num_experts)
            else:
                routing_weights = routing_weights.to(hidden_states.dtype)

            outputs = hidden_states.new_zeros(hidden_states.shape[0], self.hidden_size)

            for expert_idx, (gate_up_layer, down_layer) in enumerate(zip(self.gate_up_projs, self.down_projs)):
                weights = routing_weights[:, expert_idx]
                if torch.count_nonzero(weights).item() == 0:
                    continue

                gate_up = gate_up_layer(hidden_states)
                gate, up = gate_up.chunk(2, dim=-1)
                gate = torch.clamp(gate, max=self.limit)
                up = torch.clamp(up, min=-self.limit, max=self.limit)
                glu = gate * torch.sigmoid(gate * self.alpha)
                expert_output = down_layer((up + 1.0) * glu)
                outputs += expert_output * weights.unsqueeze(-1)

            return outputs.view(*original_shape[:-1], self.hidden_size)

    PatchedGptOssExperts.__name__ = "GptOssExperts"
    PatchedGptOssExperts.__qualname__ = "GptOssExperts"

    class PatchedGptOssTopKRouter(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.top_k = config.num_experts_per_tok
            self.num_experts = config.num_local_experts
            self.hidden_dim = config.hidden_size
            self.linear = torch.nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        def forward(self, hidden_states):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = self.linear(hidden_states)
            router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
            router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
            router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
            return router_scores, router_indices

    gpt_oss_mod.GptOssExperts = PatchedGptOssExperts
    gpt_oss_mod.GptOssTopKRouter = PatchedGptOssTopKRouter
    gpt_oss_mod._ppo_patch_moe_applied = True


def _patch_quantizer_module_resolution(model_name: str | None = None):
    if not model_name or "gpt-oss" not in model_name.lower():
        return

    """Work around mismatched module names in some quantized Expert blocks.

    Certain GPT-OSS checkpoints expose parameters under names like
    ``down_projs`` while the actual PyTorch module only registers
    ``down_proj``. Newer ``transformers`` releases then fail when
    ``BitsAndBytes`` attempts to resolve the module path. We patch the
    resolver to retry with singular names when needed.
    """

    try:
        from transformers.quantizers import quantizers_utils as _quant_utils  # type: ignore
    except Exception:
        return

    if getattr(_quant_utils, "_gpt_oss_patch_applied", False):
        return

    replacements = {
        "down_projs": "down_proj",
        "up_projs": "up_proj",
        "gate_projs": "gate_proj",
    }

    def _resolved_get_module(module, tensor_name: str):
        if "." not in tensor_name:
            return module, tensor_name

        module_name, inner_tensor = tensor_name.rsplit(".", 1)
        try:
            submodule = module.get_submodule(module_name)
        except AttributeError as exc:
            for plural, singular in replacements.items():
                if plural in module_name:
                    alt_name = module_name.replace(plural, singular)
                    try:
                        submodule = module.get_submodule(alt_name)
                        break
                    except AttributeError:
                        continue
            else:
                print(f"[quantizer patch] failed to resolve '{module_name}' for tensor '{tensor_name}'")
                raise exc
        return submodule, inner_tensor

    _quant_utils.get_module_from_name = _resolved_get_module
    try:
        from transformers.quantizers import quantizer_bnb_4bit as _quant_bnb  # type: ignore
        _quant_bnb.get_module_from_name = _resolved_get_module
    except Exception:
        pass
    _quant_utils._gpt_oss_patch_applied = True


def _patch_trl_selective_log_softmax():
    try:
        from trl.trainer import utils as _trl_utils
        from trl.trainer import ppo_trainer as _trl_ppo_trainer
    except Exception:
        return

    if getattr(_trl_utils, "_ppo_train_device_patch", False):
        return

    _orig_selective = _trl_utils.selective_log_softmax

    def _device_safe_selective_log_softmax(logits, actions):
        if logits is None or actions is None:
            return _orig_selective(logits, actions)

        target_device = getattr(logits, "device", None)
        target_dtype = torch.long

        orig_actions_device = None
        if torch.is_tensor(actions):
            orig_actions_device = actions.device

        if target_device is not None:
            if not torch.is_tensor(actions):
                actions = torch.as_tensor(actions, device=target_device, dtype=target_dtype)
            else:
                need_move = actions.device != target_device
                need_cast = actions.dtype not in (torch.long, torch.int64)
                if need_move or need_cast:
                    actions = actions.to(device=target_device, dtype=target_dtype)
        else:
            if torch.is_tensor(actions) and actions.dtype not in (torch.long, torch.int64):
                actions = actions.to(dtype=target_dtype)

        output = _orig_selective(logits, actions)

        if orig_actions_device is not None and output.device != orig_actions_device:
            output = output.to(orig_actions_device)

        return output

    _trl_utils.selective_log_softmax = _device_safe_selective_log_softmax
    try:
        setattr(_trl_ppo_trainer, "selective_log_softmax", _device_safe_selective_log_softmax)
    except Exception:
        pass
    _trl_utils._ppo_train_device_patch = True


def _ensure_unsloth_loaded() -> bool:
    global _UNSLOTH_AVAILABLE
    if _UNSLOTH_AVAILABLE:
        return True
    try:
        import unsloth  # type: ignore  # noqa: F401
        from unsloth import FastLanguageModel  # type: ignore  # noqa: F401
    except ImportError:
        return False
    _UNSLOTH_AVAILABLE = True
    return True
import torch.nn as nn
from trl.trainer import ppo_trainer as trl_ppo_trainer
from trl.trainer import utils as trl_utils
from trl.trainer.ppo_trainer import first_true_indices, PolicyAndValueWrapper

# ===================== ユーティリティ =====================
DEVICE_ID = 0  # 使う GPU の論理番号（compose等で見せている 1GPU 内の 0）

_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")
_JAPANESE_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")


def _contains_hiragana(text: str) -> bool:
    return bool(_HIRAGANA_RE.search(text))


def _text_similarity(lhs: str, rhs: str) -> float:
    if not lhs or not rhs:
        return 0.0
    return SequenceMatcher(None, lhs, rhs).ratio()


def _extract_primary_japanese_sentence(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\n", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return ""

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

    while end < len(normalized) and normalized[end] in "。．！？!?……〜ー】〕））」』'\"”] )":
        end += 1

    candidate = normalized[start:end].strip()
    candidate = re.sub(r"^\s*\[([^\]]+)\]\s*", r"\1", candidate)
    candidate = re.sub(r"^\s*([A-Za-zＡ-Ｚぁ-んァ-ン一-龥]+)\s*[:：]\s*", "", candidate)
    return candidate.strip(' "“”『』「」[]()')

_AZURE_EMBED_CLIENTS: Dict[tuple[str, str, str], AzureOpenAI] = {}
_AZURE_EMBED_FAILURES: set[tuple[str, str, str]] = set()


def _extract_reward_components(breakdown: Dict[str, Any] | None) -> Dict[str, float]:
    """抽出対象の報酬要素だけを選別し、合計しやすい形で返す。"""

    def _as_float(val: Any) -> float | None:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return None

    components: Dict[str, float] = {}
    if not breakdown:
        return components

    base_relation = _as_float(breakdown.get("base_reward_relation"))
    if base_relation is not None:
        components["relation"] = base_relation
    else:
        relation_parts: list[float] = []
        for key in ("R_tri", "R_iso", "R_all"):
            val = _as_float(breakdown.get(key))
            if val is not None:
                relation_parts.append(val)
        if relation_parts:
            components["relation"] = sum(relation_parts)

    context_reward = _as_float(breakdown.get("context_alignment_reward"))
    if context_reward is not None:
        components["context_alignment"] = context_reward
    else:
        context_norm = _as_float(breakdown.get("context_alignment_normalized"))
        if context_norm is not None:
            components["context_alignment"] = context_norm

    fallback_penalty = _as_float(breakdown.get("fallback_penalty"))
    if fallback_penalty is not None and fallback_penalty != 0.0:
        components["fallback_penalty"] = fallback_penalty

    language_penalty = _as_float(breakdown.get("language_penalty"))
    if language_penalty is not None and language_penalty != 0.0:
        components["language_penalty"] = language_penalty

    for key in ("topic_similarity_bonus", "topic_similarity_penalty", "brevity_penalty"):
        val = _as_float(breakdown.get(key))
        if val is not None and val != 0.0:
            components[key] = val

    return components


def _format_triangle_summary(rel: Dict[str, Any] | None) -> str:
    """`analyze_relations_from_scores` の結果から三角形状態や孤立ノードを短い文字列に整形する。"""

    if not isinstance(rel, dict):
        return ""

    triangles = rel.get("triangles")
    if not triangles:
        return ""

    if isinstance(triangles, dict):
        items = []
        for nodes, status in triangles.items():
            node_list = list(nodes) if isinstance(nodes, (list, tuple)) else [nodes]
            items.append({"nodes": node_list, "status": status})
    else:
        items = list(triangles)

    entries: list[str] = []

    def _display_node(name: Any) -> str:
        label = str(name)
        return "ロボ" if label == "ロボット" else label

    for tri in items:
        nodes = tri.get("nodes") if isinstance(tri, dict) else None
        if not nodes or not isinstance(nodes, (list, tuple)):
            continue
        display_nodes = [_display_node(node) for node in nodes]
        display_nodes = sorted(display_nodes, key=lambda x: (x == "ロボ", x))

        struct = ""
        if isinstance(tri, dict):
            raw_struct = tri.get("struct") or tri.get("pattern") or tri.get("structure")
            if isinstance(raw_struct, (list, tuple)):
                struct = "".join(str(part) for part in raw_struct)
            elif raw_struct is not None:
                struct = str(raw_struct)
        struct = struct.strip()

        status = None
        if isinstance(tri, dict):
            status = tri.get("status") or tri.get("state") or tri.get("label")

        if not status and struct:
            if struct in BALANCED:
                status = "S"
            elif struct in UNBALANCED:
                status = "U"

        if not status:
            status = "?"

        struct_display = struct if struct else "?"
        entries.append(f"{''.join(display_nodes)}:{struct_display}({status})")

    entries.sort()

    isolated_nodes = rel.get("isolated_humans") or rel.get("isolated_nodes") or []
    if isinstance(isolated_nodes, (set, tuple)):
        isolated_nodes = list(isolated_nodes)
    if isinstance(isolated_nodes, list):
        isolated_display = sorted({_display_node(node) for node in isolated_nodes})
        if isolated_display:
            entries.append(f"孤立:{','.join(isolated_display)}")

    return ", ".join(entries)


def _get_azure_embedding_helper(
    endpoint: str,
    deployment: str,
    api_version: str,
    api_key: str,
) -> Dict[str, Any]:
    if AzureOpenAI is None:
        raise ImportError("Azure OpenAI client libraries are not available")
    if not endpoint or not deployment:
        raise ValueError("Azure OpenAI endpoint and deployment must be configured")
    if not api_key:
        raise ValueError("Azure OpenAI API key is required for embeddings")

    key = (endpoint, deployment, api_version)
    client = _AZURE_EMBED_CLIENTS.get(key)
    if client is None:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        _AZURE_EMBED_CLIENTS[key] = client

    return {
        "client": client,
        "deployment": deployment,
        "endpoint": endpoint,
        "api_version": api_version,
    }


def _encode_embeddings_azure(texts: List[str], helper: Dict[str, Any]) -> torch.Tensor:
    if not texts:
        return torch.empty(0, 0)
    client: AzureOpenAI = helper["client"]
    deployment = helper["deployment"]
    response = client.embeddings.create(input=texts, model=deployment)
    vectors = []
    for item in response.data:
        vec = torch.tensor(item.embedding, dtype=torch.float32)
        vectors.append(vec)
    if not vectors:
        return torch.empty(0, 0)
    emb = torch.stack(vectors)
    emb = F.normalize(emb, p=2, dim=1)
    return emb


_SENTENCE_ENDINGS = ["。", "！", "？", "．", "!", "?"]


def _enforce_single_sentence(text: str) -> str:
    if not text:
        return text
    cleaned = text.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # 推論スクリプトと同じ正規化ルールに揃える
    cleaned = re.sub(r"^\s*\[([^\]]+)\]\s*", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[\-‐‑‒–—―ー〜~]*\s*([A-Za-zＡ-Ｚ])\s*[:：]\s*", "", cleaned)
    cleaned = re.sub(r"^\s*([A-Za-zＡ-Ｚぁ-んァ-ン一-龥]+)\s*[:：]\s*", "", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return cleaned

    if not cleaned.endswith(tuple(_SENTENCE_ENDINGS)):
        cleaned = cleaned.rstrip(" 　。、！？」") + "。"
    return cleaned


_FALLBACK_SENTENCE = "お待たせいたしました、状況を丁寧に整えますね。"
_THINK_CLOSE_TAG = "</think>"


def _strip_think_tags(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    end_idx = lowered.rfind(_THINK_CLOSE_TAG)
    if end_idx != -1:
        return text[end_idx + len(_THINK_CLOSE_TAG):].lstrip()
    return re.sub(r"<\s*/?\s*think\s*>", "", text, flags=re.IGNORECASE).strip()


def _prepare_robot_response(raw_text: str) -> str:
    """推論スクリプトと同じ後処理を適用して、環境に送る文字列を整える。"""
    visible = _strip_think_tags(raw_text or "")
    primary = _extract_primary_japanese_sentence(visible)
    candidate = _enforce_single_sentence(primary or "")
    if candidate and _contains_hiragana(candidate):
        return candidate

    fallback_source = primary or visible
    fallback = _enforce_single_sentence(fallback_source) if fallback_source else ""
    if fallback and _contains_hiragana(fallback):
        return fallback

    return _FALLBACK_SENTENCE

def set_seed(seed: int = 42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _unwrap_module(m: torch.nn.Module):
    """Accelerate / FSDP / DDP / TRL のラッパをできるだけ剥がしてベースモデルを返す。"""
    cur = m
    visited = set()
    while True:
        obj_id = id(cur)
        if obj_id in visited:
            break
        visited.add(obj_id)

        if hasattr(cur, "module"):
            cur = cur.module
            continue
        if isinstance(cur, PolicyAndValueWrapper) and hasattr(cur, "policy"):
            cur = cur.policy
            continue
        if hasattr(cur, "pretrained_model"):
            cur = cur.pretrained_model
            continue
        break
    return cur

def _force_model_output_tuple_safe(m: torch.nn.Module):
    """
    ベースモデルの forward を直接パッチして:
      - return_dict=True を強制
      - tuple が返ったら out[0] を logits とみなして ModelOutput に包む
    """
    base = _unwrap_module(m)
    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
    except Exception:
        CausalLMOutputWithPast = None

    orig_forward = base.forward

    def _wrapped_forward(*args, **kwargs):
        kwargs.setdefault("return_dict", True)
        out = orig_forward(*args, **kwargs)
        if isinstance(out, tuple):
            # 慣例的に out[0] が logits
            if CausalLMOutputWithPast is not None:
                return CausalLMOutputWithPast(logits=out[0])
            return type("Obj", (), {"logits": out[0]})
        return out

    base.forward = _wrapped_forward


class TrainingRunLogger:
    """学習中の会話ログとメトリクスを JSONL で蓄積するユーティリティ。"""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._convo_path = self.run_dir / "conversation_summary.jsonl"
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._events_path = self.run_dir / "run_events.jsonl"
        self._llm_raw_path = self.run_dir / "llm_details.jsonl"
        self._convo_fp = open(self._convo_path, "a", encoding="utf-8")
        self._metrics_fp = open(self._metrics_path, "a", encoding="utf-8")
        self._events_fp = open(self._events_path, "a", encoding="utf-8")
        self._llm_raw_fp = open(self._llm_raw_path, "a", encoding="utf-8")
        atexit.register(self.close)

    def _round_float(self, value: float) -> float:
        if math.isnan(value) or math.isinf(value):
            return value
        magnitude = abs(value)
        if magnitude == 0:
            return 0.0
        if magnitude >= 1000:
            digits = 0
        elif magnitude >= 100:
            digits = 1
        elif magnitude >= 10:
            digits = 2
        elif magnitude >= 1:
            digits = 3
        else:
            digits = 4
        rounded = round(value, digits)
        if rounded == -0.0:
            rounded = 0.0
        return rounded

    def _sanitize(self, obj):
        if isinstance(obj, dict):
            safe = {}
            for k, v in obj.items():
                if isinstance(k, (str, int, float, bool)) or k is None:
                    safe_key = k
                elif isinstance(k, (tuple, list, set)):
                    safe_key = ",".join(str(item) for item in k)
                else:
                    safe_key = str(k)
                safe[safe_key] = self._sanitize(v)
            return safe
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize(v) for v in obj]
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return self._sanitize(obj.item())
            except Exception:
                pass
        if isinstance(obj, float):
            return self._round_float(obj)
        if isinstance(obj, (str, int, bool)) or obj is None:
            return obj
        return str(obj)

    def _write_jsonl(self, fp, payload: dict):
        record = {
            **payload,
            "timestamp": payload.get("timestamp") or datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        fp.write(json.dumps(self._sanitize(record), ensure_ascii=False) + "\n")
        fp.flush()

    def log_interaction(self, **payload):
        self._write_jsonl(self._convo_fp, payload)

    def log_llm_response(self, **payload):
        record = {
            "interaction_id": payload.get("interaction_id"),
            "episode": payload.get("episode"),
            "turn": payload.get("turn"),
            "draft": payload.get("draft_response"),
            "sanitized": payload.get("sanitized_response"),
            "fallback_used": payload.get("fallback_used", False),
        }
        if payload.get("state_before") is not None:
            record["state_before"] = payload["state_before"]
        if payload.get("history_before"):
            record["history_before"] = payload["history_before"]
        if payload.get("personas"):
            record["personas"] = payload["personas"]
        other_keys = {k: v for k, v in payload.items() if k not in {
            "interaction_id",
            "episode",
            "turn",
            "draft_response",
            "sanitized_response",
            "fallback_used",
            "state_before",
            "history_before",
            "personas",
        }}
        if other_keys:
            record["details"] = other_keys
        self._write_jsonl(self._llm_raw_fp, record)

    def log_metrics(self, split: str, metrics: dict, step: int | None = None, epoch: float | None = None):
        clean = self._sanitize(metrics or {})
        payload = {"split": split, "metrics": clean}
        if step is not None:
            payload["step"] = step
        if epoch is not None:
            payload["epoch"] = epoch
        self._write_jsonl(self._metrics_fp, payload)

    def log_event(self, **payload):
        event = payload.pop("event", "info")
        record: Dict[str, Any] = {"event": event}
        for key in ("run_id", "episode", "step", "total_steps"):
            if key in payload:
                record[key] = payload[key]

        if event == "run_start" and "state" in payload:
            record["state"] = payload["state"]

        if event == "episode_start":
            record["state"] = payload.get("state")
            if payload.get("personas"):
                record["personas"] = payload["personas"]
        elif event == "episode_end":
            record["reward"] = {
                "total": payload.get("reward"),
                "components": _extract_reward_components(payload.get("reward_breakdown")),
            }
            if payload.get("env_metrics"):
                record["env_summary"] = {
                    "triangles_balanced": payload["env_metrics"].get("balanced_triads"),
                    "triangles_unbalanced": payload["env_metrics"].get("unstable_triads"),
                }

        other_keys = {k: v for k, v in payload.items() if k not in {"state", "personas", "reward", "reward_breakdown", "env_metrics"}}
        if other_keys:
            record["details"] = other_keys

        self._write_jsonl(self._events_fp, record)

    def write_json(self, name: str, data: dict):
        target = self.run_dir / name
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._sanitize(data), f, ensure_ascii=False, indent=2)

    def close(self):
        for fp in (
            getattr(self, attr, None)
            for attr in ("_convo_fp", "_metrics_fp", "_events_fp", "_llm_raw_fp")
        ):
            if fp and not fp.closed:
                fp.flush()
                fp.close()

@lru_cache(maxsize=1)
def _default_model_name() -> str | None:
    try:
        cfg = get_config()
        return getattr(getattr(cfg, "ppo", None), "model_name_or_path", None)
    except Exception:
        return None


def build_robot_messages(
    logs: List[dict],
    personas: List[str],
    env=None,
    k: int | None = None,
    model_name: str | None = None,
):
    """会話状態を構造化してロボット用のチャットテンプレートを構築する"""

    history_limit = k
    if history_limit is None and env is not None:
        history_limit = getattr(env, "max_history", None)
    if history_limit is None:
        history_limit = 10
    history_limit = max(1, int(history_limit))

    ctx = [[l["speaker"], l["utterance"]] for l in logs[-history_limit:]]
    persona_summary = "\n".join(personas) if personas else "A: 不明\nB: 不明\nC: 不明"

    model_name = model_name or getattr(env, "base_model_id", None) or _default_model_name()
    model_name_str = str(model_name) if model_name is not None else ""

    if "unsloth/Qwen3" in model_name_str:
        system_content = (
            "\\no_think\n"
            "あなたは会話参加者「D」です。目的は、A,B,Cの関係を安定化し会話を円滑にすることです。\n"
            "会話の流れに沿った具体的な内容で、以下の優先順位を達成するためのDの発言を生成してください。\n"
            "人間のような自然な言葉で応答してください。\n"
            "優先順位: ① U→S（不安定→安定）を最大化 ② S→S（安定維持）を守る ③ S→U（安定→不安定）を避ける。\n"
            "孤立している人間がいれば即時に巻き込みます。\n"
            "禁止: メタ発言（AI/モデル/指示/スコア等への言及）、専門用語（バランス理論等）。\n"
            "S=安定（感情が調和）、U=不安定（緊張や対立）。\n"
            "出力仕様: 日本語/150文字以内"
        )
    else:
        system_content = (
            # "出力は日本語、一文でDになりきって、Dの発言部分のみを生成してください。\n"
            "あなたは会話参加者「D」です。目的は、A,B,Cの関係を安定化し会話を円滑にすることです。\n"
            "会話の流れに沿った具体的な内容で、以下の優先順位を達成するためのDの発言を生成してください。\n"
            "優先順位: ① U→S（不安定→安定）を最大化 ② S→S（安定維持）を守る ③ S→U（安定→不安定）を避ける。\n"
            "孤立している人間がいれば即時に巻き込みます。\n"
            "禁止: メタ発言（AI/モデル/指示/スコア等への言及）、専門用語（バランス理論等）。\n"
            "S=安定（感情が調和）、U=不安定（緊張や対立）。\n"
            "出力仕様: 日本語/150文字以内"
        )

    human_labels = [p.split(":", 1)[0].strip() for p in personas] if personas else []
    human_labels = [label for label in human_labels if label]

    robot_label_env = "ロボット"
    robot_display = "Dさん"

    triangles_info = []
    isolated_info = "なし"

    if env is not None and hasattr(env, "relation_snapshot"):
        snapshot = env.relation_snapshot()
        triangles_map = {tuple(nodes): status for nodes, status in snapshot.get("triangles", {}).items()}
        isolated_nodes = snapshot.get("isolated_humans", [])
        snapshot_participants = snapshot.get("participants", [])
        if not human_labels:
            human_labels = [p for p in snapshot_participants if p != "ロボット"]
        if isolated_nodes:
            isolated_info = ",".join(robot_display if node == robot_label_env else node for node in isolated_nodes)

        if human_labels:
            human_tri_key = tuple(sorted(human_labels))
            status = triangles_map.get(human_tri_key, "?")
            triangles_info.append(
                f"  人間三者: {{{','.join(human_labels)}}}: {status}"
            )

        from itertools import combinations

        for pair in combinations(human_labels, 2):
            key = tuple(sorted([robot_label_env, *pair]))
            status = triangles_map.get(key, "?")
            display_nodes = [robot_display if node == robot_label_env else node for node in [robot_label_env, *pair]]
            triangles_info.append(
                f"  {robot_display}入り: {{{','.join(display_nodes)}}}: {status}"
            )

    if not triangles_info:
        triangles_info.append("  (情報なし)")

    history_lines = [f"ー {speaker}: {utterance}" for speaker, utterance in ctx]
    if not history_lines:
        history_lines.append("ー (まだ発言なし)")

    user_lines = [
        f"参加者: 人間=[{','.join(human_labels) if human_labels else 'A,B,C'}], D=[{robot_display}]",
        "三角形（S=安定/U=不安定）:",
        *triangles_info,
        f"孤立ノード（人間のみ）: {isolated_info}",
        "ログ:",
        *history_lines,
    ]

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "\n".join(user_lines)},
    ]
    return messages

# ===================== 有限ストリーム・データセット =====================
class FiniteOnlineDataset(TorchIterableDataset):
    """
    合計 total_samples 件だけプロンプトを供給する IterableDataset。
    TRL 0.23 の `trainer.train()` は、データローダが尽きるまでを1エポックとして学習します。
    `num_ppo_epochs=1` にしておけば、合計サンプル数＝更新回数×バッチサイズで
    ちょうど意図した回数だけ PPO 更新が走ります。
    """
    def __init__(self, tokenizer, env, build_messages_fn, total_samples: int):
        self.tokenizer = tokenizer
        self.env = env
        self.build_messages_fn = build_messages_fn
        self.total_samples = int(total_samples)
        self._epoch = 0
    # accelerate.DataLoader から呼ばれる
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
    def __iter__(self):
        for _ in range(self.total_samples):
            self.env.ensure_human_turn()
            messages = self.build_messages_fn(self.env.logs, self.env.personas, env=self.env)
            if getattr(self.env, "debug_prompts", getattr(self.env, "debug", False)):
                print("\n[ppo_train] === Prompt feed ===")
                print(f"episode={self.env.episode}, turn={self.env.t}, total_steps={self.env.total_steps}")
                print("[ppo_train] conversation logs (env.logs):")
                for entry in self.env.logs:
                    speaker = entry.get("speaker", "?")
                    utt = entry.get("utterance", "")
                    print(f"  [{speaker}] {utt}")
                print("[ppo_train] chat_template user payload:")
                try:
                    user_payload = next((m for m in messages if m.get("role") == "user"), None)
                    if user_payload is not None:
                        print(user_payload.get("content", ""))
                    else:
                        print(messages)
                except Exception:
                    print(messages)
            query = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            yield {"query": query}
    def __len__(self):
        return self.total_samples


class DummyRewardModel(nn.Module):
    """環境ベース報酬を使うためのダミー報酬モデル。

    TRL 0.23 の PPOTrainer は torch.nn.Module を期待し `.to()` を呼ぶため、
    forward が呼ばれない最小実装を用意する。
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):  # pragma: no cover - 呼ばれない想定
        raise RuntimeError("DummyRewardModel.forward should not be called")

def main():
    cfg = get_config()
    ppo_cfg = cfg.ppo

    set_seed(42)

    model_id = getattr(ppo_cfg, "model_name_or_path", None)

    _patch_gpt_oss_moe_layers(model_id)
    _patch_quantizer_module_resolution(model_id)
    _patch_trl_selective_log_softmax()

    cuda_visible = getattr(ppo_cfg, "cuda_visible_devices", None)
    if cuda_visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)

    device = torch.device("cuda")
    visible_raw = os.getenv("CUDA_VISIBLE_DEVICES", "(all)")
    print(f"[CUDA] CUDA_VISIBLE_DEVICES={visible_raw}")
    logical_count = torch.cuda.device_count()
    print(f"[CUDA] torch.cuda.device_count={logical_count}")
    if visible_raw not in (None, "", "(all)"):
        logical_to_physical = {logical: gpu for logical, gpu in enumerate([v.strip() for v in visible_raw.split(",") if v.strip()])}
        print(f"[CUDA] logical→physical map: {logical_to_physical}")

    # --- トークナイザ & モデル（ValueHead付き） ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=getattr(ppo_cfg, "revision", None),
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # 生成時のバッチ安定のため、左パディングに変更

    # LoRA 設定
    lora = LoraConfig(
        r=ppo_cfg.lora_rank,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen/Llama系と同様の定番。MoEでもまずは注意機構を押さえる
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )

    quant = BitsAndBytesConfig(
        load_in_4bit=True,              # QLoRA 前提
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # --- (A) 層クラス名の抽出（空モデル; VRAM消費なし） ---
    cfg_hf = AutoConfig.from_pretrained(model_id, revision=getattr(ppo_cfg, "revision", None))
    with init_empty_weights():
        base_empty = AutoModelForCausalLM.from_config(cfg_hf)
    base_for_map = base_empty
    # 候補: “Layer/Block/Decoder/Encoder” を含むクラス名を抽出
    layer_classes = sorted(
        {m.__class__.__name__ for m in base_empty.modules()
         if any(k in m.__class__.__name__ for k in ["Layer","Block","Decoder","Encoder"])}
    )
    # まずは DecoderLayer/Block を優先。なければ全候補を使う
    preferred = [n for n in layer_classes if ("DecoderLayer" in n) or n.endswith("Block")]
    no_split = preferred or layer_classes
    print("[no_split_module_classes]", no_split)

    # --- (B) device_map を先に作る ---
    # 使いたい GPU を列挙（例: 0,1,2,6,7,8）
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    visible_list = [v.strip() for v in visible.split(",") if v.strip()]
    if visible_list:
        gpu_indices = list(range(len(visible_list)))
    else:
        gpu_indices = list(range(torch.cuda.device_count()))

    def _resolve_device_id(value):
        if value is None:
            return None
        if isinstance(value, str):
            s = value.strip()
            if s in visible_list:
                return visible_list.index(s)
            if s.isdigit():
                num = int(s)
            else:
                raise ValueError(f"ref_device 指定 '{value}' が解釈できません")
        else:
            num = int(value)
        if 0 <= num < len(gpu_indices):
            return num
        if visible_list and str(num) in visible_list:
            return visible_list.index(str(num))
        raise ValueError(f"GPU index {value} が利用可能なデバイスに存在しません")

    def _ensure_float(value) -> float:
        if isinstance(value, str):
            v = value.strip().lower()
            for suffix in ("gib", "gb"):
                if v.endswith(suffix):
                    v = v[: -len(suffix)]
                    break
            try:
                return float(v)
            except ValueError as exc:
                raise ValueError(f"max_memory の値 '{value}' をfloatに変換できません") from exc
        return float(value)

    def _format_gib(value: float) -> str:
        return f"{float(value):.2f}GiB"

    def _default_limit_from_total(total_gb: float, *, min_limit: float = 6.0) -> float:
        """Derive a safe per-GPU memory limit with headroom for activations/overheads."""
        safety_margin = max(4.0, min(8.0, total_gb * 0.1))
        return max(min_limit, total_gb - safety_margin)

    device_map_strategy = getattr(ppo_cfg, "device_map_strategy", None)
    if isinstance(device_map_strategy, str):
        device_map_strategy = device_map_strategy.strip() or None

    max_memory = None
    device_map = None

    if device_map_strategy is not None:
        strategy_name = device_map_strategy.lower()
        print(f"[device_map] using strategy='{device_map_strategy}'")
        if strategy_name in {"balanced", "balanced_low_0"}:
            low_zero = strategy_name == "balanced_low_0"
            balanced_memory = get_balanced_memory(
                base_for_map,
                dtype=torch.float16,
                no_split_module_classes=no_split,
                low_zero=low_zero,
            )
            filtered_memory = {}
            for logical_idx in gpu_indices:
                val = balanced_memory.get(logical_idx)
                if val is None:
                    val = balanced_memory.get(str(logical_idx))
                if isinstance(val, (int, float)) and val > 0:
                    filtered_memory[logical_idx] = int(val)
            if not filtered_memory:
                for key, val in (balanced_memory or {}).items():
                    try:
                        idx = int(key)
                    except Exception:
                        continue
                    if val and idx in gpu_indices:
                        filtered_memory[idx] = int(val)
            if filtered_memory:
                max_memory = {
                    idx: _format_gib(val / (1024 ** 3))
                    for idx, val in filtered_memory.items()
                }
                device_map = infer_auto_device_map(
                    base_for_map,
                    no_split_module_classes=no_split,
                    max_memory=filtered_memory,
                    dtype=torch.float16,
                )
                print("[device_map inferred]", device_map)
                print("[max_memory constraints]", max_memory)
                print("[device_map ready] (strategy)")
            else:
                print("[device_map] balanced strategy returned no GPU hints; falling back to auto inference")
                device_map = infer_auto_device_map(
                    base_for_map,
                    no_split_module_classes=no_split,
                    dtype=torch.float16,
                )
                max_memory = None
        else:
            device_map = device_map_strategy
            print("[device_map ready] (raw strategy hint)")
    else:
        max_memory_cfg = getattr(ppo_cfg, "max_memory_map", None)
        per_device_limit_cfg = getattr(ppo_cfg, "max_memory_per_device_gib", None)

        if isinstance(max_memory_cfg, dict):
            max_memory = {}
            for logical_idx in gpu_indices:
                val = (max_memory_cfg.get(logical_idx)
                       or max_memory_cfg.get(str(logical_idx)))
                if val is None:
                    raise ValueError(f"max_memory_map に論理GPU {logical_idx} の設定が必要です")
                max_memory[logical_idx] = _format_gib(_ensure_float(val))
        elif isinstance(per_device_limit_cfg, (list, tuple)):
            max_memory = {
                idx: _format_gib(_ensure_float(per_device_limit_cfg[min(pos, len(per_device_limit_cfg) - 1)]))
                for pos, idx in enumerate(gpu_indices)
            }
        elif isinstance(per_device_limit_cfg, (int, float)) and per_device_limit_cfg:
            max_memory = {idx: _format_gib(_ensure_float(per_device_limit_cfg)) for idx in gpu_indices}

        auto_adjust = False
        if max_memory is None and len(gpu_indices) > 1:
            auto_adjust = True
            try:
                total_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            except Exception:
                total_gb = 49.0
            limit = _default_limit_from_total(total_gb)
            for _ in range(12):
                candidate_memory = {idx: _format_gib(limit) for idx in gpu_indices}
                try:
                    candidate_map = infer_auto_device_map(
                        base_for_map,
                        no_split_module_classes=no_split,
                        max_memory=candidate_memory,
                        dtype=torch.float16,
                    )
                except Exception:
                    if limit <= 3.0:
                        break
                    limit = max(3.0, limit - 1.0)
                    continue
                unique_devices = set(candidate_map.values())
                if len(unique_devices) == 1 and limit > 3.0:
                    limit = max(3.0, limit - 1.0)
                    continue
                max_memory = candidate_memory
                device_map = candidate_map
                break

        if device_map is None:
            device_map = infer_auto_device_map(
                base_for_map,
                no_split_module_classes=no_split,
                max_memory=max_memory,
                dtype=torch.float16,  # 4bit計算 dtype のヒント
            )

        def _ensure_gpu_only_map(existing_map, current_max):
            if not isinstance(existing_map, dict):
                return existing_map, current_max
            invalid = {name: target for name, target in existing_map.items() if not isinstance(target, int)}
            if not invalid:
                return existing_map, current_max
            if not gpu_indices:
                raise RuntimeError("GPU indices are empty; cannot sanitize device_map")
            cycle_targets = itertools.cycle(gpu_indices)
            sanitized = {}
            for name, target in existing_map.items():
                if name in invalid:
                    assigned = next(cycle_targets)
                    sanitized[name] = assigned
                else:
                    sanitized[name] = target
            print(f"[device_map] removed non-GPU targets {invalid}; reassigned to GPUs {gpu_indices}")
            return sanitized, None  # max_memory no longer valid

        def _force_multi_device_map(existing_map, current_max):
            if not isinstance(existing_map, dict):
                return existing_map, current_max
            gpu_devices = [dev for dev in existing_map.values() if isinstance(dev, int)]
            unique_gpu_devices = sorted(set(gpu_devices))
            if len(unique_gpu_devices) <= 1 and len(gpu_indices) > 1:
                primary = unique_gpu_devices[0] if unique_gpu_devices else gpu_indices[0]
                limit_pool = {}
                if isinstance(current_max, dict) and current_max:
                    for idx in gpu_indices:
                        raw = current_max.get(idx, current_max.get(str(idx)))
                        if raw is not None:
                            try:
                                limit_pool[idx] = _ensure_float(raw)
                            except Exception:
                                pass
                if not limit_pool:
                    try:
                        total_gb0 = torch.cuda.get_device_properties(gpu_indices[0]).total_memory / 1024 ** 3
                    except Exception:
                        total_gb0 = 49.0
                    fallback_limit = _default_limit_from_total(total_gb0, min_limit=12.0)
                    limit_pool = {idx: fallback_limit for idx in gpu_indices}
                else:
                    fallback_limit = limit_pool.get(primary)
                    if fallback_limit is None:
                        try:
                            total_gb0 = torch.cuda.get_device_properties(gpu_indices[0]).total_memory / 1024 ** 3
                        except Exception:
                            total_gb0 = 49.0
                        fallback_limit = _default_limit_from_total(total_gb0, min_limit=12.0)
                for idx in gpu_indices:
                    limit_pool.setdefault(idx, fallback_limit)
                print("[device_map] attempting redistribution across GPUs", gpu_indices)
                for attempt in range(1, 6):
                    limit_pool[primary] = max(6.0, limit_pool[primary] - 2.0)
                    if start > 0 and normalized[start - 1].isascii() and normalized[start - 1].isalnum():
                        i = start
                        while i > 0 and normalized[i - 1].isascii() and normalized[i - 1].isalnum():
                            i -= 1
                        prefix = normalized[i:start]
                        if len(prefix) == 1 and prefix.isalpha() and prefix.isupper():
                            start = i
                    try:
                        candidate = infer_auto_device_map(
                            base_for_map,
                            no_split_module_classes=no_split,
                            max_memory=limit_payload,
                            dtype=torch.float16,
                        )
                    except Exception as exc:
                        print(f"[device_map] redistribution attempt {attempt} failed: {exc}")
                        continue
                    used = {dev for dev in candidate.values() if isinstance(dev, int)}
                    if len(used) > 1:
                        print(f"[device_map] succeeded after attempt {attempt}; GPU{primary} limit => {limit_payload[primary]}")
                        return candidate, limit_payload
                print("[device_map] redistribution attempts exhausted; keeping single-GPU map.")
            return existing_map, current_max

        device_map, max_memory = _ensure_gpu_only_map(device_map, max_memory)
        device_map, max_memory = _force_multi_device_map(device_map, max_memory)

        if max_memory is not None:
            print("[max_memory constraints]", max_memory)
        print("[device_map ready]")

    # --- (C) ここで from_pretrained。no_split は渡さず、作成済み device_map を渡す ---
    if "gpt-oss" in str(model_id).lower():
        if _ensure_unsloth_loaded():
            print("[unsloth] FastLanguageModel patches active for GPT-OSS base")
        else:
            print(
                "[warn] GPT-OSS model requested but unsloth is unavailable. Install unsloth for optimal performance."
            )

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_id,
        revision=getattr(ppo_cfg, "revision", None),
        device_map=device_map,
        quantization_config=quant,
        peft_config=lora,
        low_cpu_mem_usage=True,
    )
    # ValueHeadラッパに base モデルの属性を生やす（TRL 0.23 初期化対策）----
    from transformers import GenerationConfig
    base = getattr(policy, "pretrained_model", policy)
    policy.generation_config = GenerationConfig.from_model_config(base.config)
    policy.generation_config.pad_token_id = policy.generation_config.pad_token_id or tokenizer.eos_token_id
    policy.generation_config.eos_token_id = tokenizer.eos_token_id

    def _attach_score_method(target: torch.nn.Module, head_owner: torch.nn.Module):
        """Ensure the given module exposes a `.score` method backed by the value head."""
        if target is None or hasattr(target, "score"):
            return

        def _score(self, hidden_states: torch.Tensor):
            head = getattr(self, "v_head", None)
            if head is None and head_owner is not None:
                head = getattr(head_owner, "v_head", None)
            if head is None:
                raise AttributeError("No value head available to compute scores.")
            try:
                head_weight = next(head.parameters())
                head_dev = head_weight.device
                head_dtype = head_weight.dtype
            except StopIteration:
                head_dev = getattr(hidden_states, "device", None)
                head_dtype = getattr(hidden_states, "dtype", torch.float32)

            hs_dev = hidden_states.device
            hs_dtype = hidden_states.dtype
            if (head_dev != hs_dev) or (head_dtype != hs_dtype):
                head.to(device=hs_dev, dtype=hs_dtype)

            value = head(hidden_states)
            if value.ndim == 3 and value.size(-1) == 1:
                value = value.squeeze(-1)
            return value

        target.score = types.MethodType(_score, target)

    _attach_score_method(policy, policy)
    _attach_score_method(base, policy)

    def _safe_set_cfg(cfg, key, value):
        try:
            setattr(cfg, key, value)
            return True
        except Exception:
            pass
        try:
            # PretrainedConfig は update をサポート
            cfg.update({key: value})
            return True
        except Exception:
            pass
        try:
            # どうしてもダメなら __dict__ を直接更新（読み取り専用のときはスキップ）
            cfg.__dict__[key] = value
            return True
        except Exception:
            return False

    def _enforce_return_dict(m):
        if m is None:
            return
        cfg = getattr(m, "config", None)
        if cfg is None:
            return
        # tuple ではなく ModelOutput を返すように
        _safe_set_cfg(cfg, "return_dict", True)
        # 学習時の安定化と省メモリ
        _safe_set_cfg(cfg, "use_cache", False)

    _enforce_return_dict(policy)
    _enforce_return_dict(base)

    if bool(getattr(ppo_cfg, "gradient_checkpointing", True)):
        for module in {policy, base}:
            if module is None:
                continue
            if hasattr(module, "gradient_checkpointing_enable"):
                try:
                    module.gradient_checkpointing_enable()
                except Exception as exc:
                    print(f"[warn] gradient checkpointing enable failed for {module.__class__.__name__}: {exc}")
        if hasattr(base, "enable_input_require_grads"):
            try:
                base.enable_input_require_grads()
            except Exception:
                pass

    # 依然として tuple を返すモデル向けの最終手段（必要時のみ）
    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
        _base = getattr(policy, "pretrained_model", policy)
        _orig_forward = _base.forward
        def _forward_wrap(*args, **kwargs):
            kwargs.setdefault("return_dict", True)
            out = _orig_forward(*args, **kwargs)
            if isinstance(out, tuple):
                # 慣例的に out[0] が logits
                return CausalLMOutputWithPast(logits=out[0])
            return out
        _base.forward = _forward_wrap
    except Exception:
        pass

    def print_trainable_params(model):
        total, trainable = 0, 0
        for _, p in model.named_parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        print(f"trainable params: {trainable:,} / {total:,} ({trainable/total:.2%})")
    print_trainable_params(policy)

    # === ここからデバイス配置とGPU使用量の可視化 ===
    try:
        from accelerate.utils import find_device  # importだけでOK（未使用でも可）
        dm = getattr(policy, "hf_device_map", getattr(policy.pretrained_model, "hf_device_map", None))
        print("[device_map]", dm)
    except Exception as e:
        print("[device_map] not available:", e)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            used = total - free
            print(f"[GPU{i}] used={used/1024**3:.2f}GiB / total={total/1024**3:.2f}GiB")

    # base 側の代表的な属性を、ラッパ側にも生やす（AttributeError回避）
    # （ないと PolicyAndValueWrapper 内で AttributeError）
    for attr, default in [
        ("is_gradient_checkpointing", False),
        ("config", None),
        ("base_model_prefix", None),
    ]:
        if not hasattr(policy, attr):
            setattr(policy, attr, getattr(base, attr, default))

    # ---- ここから return_dict を強制するためのヘルパ ----
    def patch_return_dict(model):
        """
        transformers が tuple を返す旧実装に対して、
        常に `return_dict=True` にし、tuple なら logits を包んで返す。
        """
        try:
            from transformers.modeling_outputs import CausalLMOutputWithPast
        except Exception:
            CausalLMOutputWithPast = None

        m = getattr(model, "pretrained_model", model)
        # config 側で可能ならフラグを立てる（古い config は 'return_dict'）
        cfg = getattr(m, "config", None)
        if cfg is not None:
            # Config 側は update 経由（凍結でも通る可能性が高い）
            try:
                cfg.update({"return_dict": True, "use_cache": False})
            except Exception:
                try:
                    cfg.__dict__.update({"return_dict": True, "use_cache": False})
                except Exception:
                    pass

        # forward をラップ
        orig_forward = m.forward
        def _forward_wrap(*args, **kwargs):
            kwargs.setdefault("return_dict", True)
            out = orig_forward(*args, **kwargs)
            if isinstance(out, tuple):
                # 慣例的に out[0] が logits
                if CausalLMOutputWithPast is not None:
                    return CausalLMOutputWithPast(logits=out[0])
            return out
        m.forward = _forward_wrap

    # policy 本体に適用
    patch_return_dict(policy)
    patch_return_dict(base)


    kl_coef_raw = getattr(ppo_cfg, "kl_coef", 0.0)
    try:
        kl_coef_value = float(kl_coef_raw)
    except Exception:
        kl_coef_value = 0.0

    ref_policy = None
    ref_device_map_cfg = getattr(ppo_cfg, "ref_device_map", None)
    ref_device_cfg = getattr(ppo_cfg, "ref_device", None)
    ref_device_map = None
    if isinstance(ref_device_map_cfg, dict):
        ref_device_map = {}
        for key, val in ref_device_map_cfg.items():
            ref_device_map[str(key)] = _resolve_device_id(val)
    elif ref_device_cfg is not None:
        resolved = _resolve_device_id(ref_device_cfg)
        if resolved is not None:
            ref_device_map = {"": resolved}
    if kl_coef_value > 0.0:
        # === 参照モデル用 Config（setattr禁止でも通るやり方） ===
        _raw = AutoConfig.from_pretrained(model_id, revision=getattr(ppo_cfg, "revision", None))
        # dict化して書き換え → from_dict で再生成（凍結属性でもOK）
        _d = _raw.to_dict()
        # LlamaConfig は use_return_dict を受け付けないので入れない
        _d.pop("use_return_dict", None)
        _d["return_dict"] = True
        _d["use_cache"]  = False
        ref_cfg = type(_raw).from_dict(_d)

        # 参照モデル本体（LoRA なし・勾配停止・4bit/8bit はお好み）
        if ref_device_map is None:
            if isinstance(device_map, dict):
                used_devices = {dev for dev in device_map.values() if isinstance(dev, int)}
                if len(used_devices) == 1:
                    sole = next(iter(used_devices))
                    ref_device_map = {"": sole}
                else:
                    primary = (sorted(used_devices)[0] if used_devices else (gpu_indices[0] if gpu_indices else 0))
                    ref_device_map = {"": primary}
            else:
                primary = gpu_indices[0] if gpu_indices else 0
                ref_device_map = {"": primary}

        ref_policy = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=getattr(ppo_cfg, "revision", None),
            config=ref_cfg,
            device_map=ref_device_map,
            quantization_config=quant,        # 省メモリしたいなら 4bit でもOK
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    # YAMLでクォートされていると文字列になることがあるので安全にキャスト
    def _to_int(x, d): 
        try: return int(x)
        except: return d
    def _to_float(x, d):
        try: return float(x)
        except: return d

    _bs    = _to_int(getattr(ppo_cfg, "batch_size", 8), 8)
    _mbs   = max(1, _bs // 2)
    _ga    = _to_int(getattr(ppo_cfg, "grad_accum_steps", 1), 1)
    _epochs= _to_int(getattr(ppo_cfg, "ppo_epochs", 1), 1)  # ←回数を外部（データ件数）で制御するので 1 がラク
    _lr    = _to_float(getattr(ppo_cfg, "lr", 1e-5), 1e-5)
    _kl    = max(0.0, kl_coef_value)
    _target_kl = max(0.0, _to_float(getattr(ppo_cfg, "target_kl", 0.02), 0.02))
    _kl_adjust_up = _to_float(getattr(ppo_cfg, "kl_adjust_up", 1.3), 1.3)
    if (not math.isfinite(_kl_adjust_up)) or _kl_adjust_up <= 1.0:
        _kl_adjust_up = 1.3
    _kl_adjust_down = _to_float(getattr(ppo_cfg, "kl_adjust_down", 0.9), 0.9)
    if (not math.isfinite(_kl_adjust_down)) or not (0.0 < _kl_adjust_down < 1.0):
        _kl_adjust_down = 0.9
    _cliprange = max(0.0, _to_float(getattr(ppo_cfg, "cliprange", 0.2), 0.2))
    _cliprange_value = max(0.0, _to_float(getattr(ppo_cfg, "cliprange_value", 0.2), 0.2))

    # === PPOConfig（TRL 0.23 互換の安全パラメータ）====
    ppo_config = PPOConfig(
        learning_rate=_lr,
        batch_size=max(1, _bs),                # 収集するサンプル数/更新
        mini_batch_size=max(1, _mbs),          # ミニバッチ
        gradient_accumulation_steps=max(1, _ga),
        num_ppo_epochs=_epochs,
        kl_coef=_kl,
        kl_estimator="k1",
        cliprange=_cliprange,
        vf_coef=0.1,
        cliprange_value=_cliprange_value,
        gamma=1.0,
        lam=0.95,
        ds3_gather_for_generation=True,        # ZeRO-3 を使う場合の最適化（使ってなければ無害）
    )
    ppo_config.target_kl = _target_kl
    ppo_config.kl_adjust_up = _kl_adjust_up
    ppo_config.kl_adjust_down = _kl_adjust_down
    # 生成の終端
    ppo_config.stop_token_id = tokenizer.eos_token_id
    # 1 デバイス当たりのバッチサイズ（Accelerate の world size 計算と整合させる）
    ppo_config.per_device_train_batch_size = max(
        1,
        _to_int(getattr(ppo_cfg, "per_device_train_batch_size", _bs), 1),
    )
    ppo_config.per_device_eval_batch_size = max(
        1,
        _to_int(getattr(ppo_cfg, "per_device_eval_batch_size", ppo_config.per_device_train_batch_size), 1),
    )

    # === 3) PPOTrainer 準備（環境ベース報酬を差し込む） ===
    # 会話環境
    env = ConversationEnv(
        max_steps=cfg.env.max_steps,
        personas=cfg.env.personas,
        include_robot=cfg.env.include_robot,
        ema_alpha=cfg.scorer.ema_alpha,
        decay_factor=cfg.scorer.decay_factor,
        backend=cfg.scorer.backend,
        debug=cfg.env.debug,
        prompt_debug=getattr(ppo_cfg, "prompt_feed_debug", None),
        max_history=cfg.env.max_history,
        reward_backend=getattr(cfg.env, "reward_backend", None),
    )

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_log_dir = Path("./logs") / f"ppo_run-{run_id}"
    training_logger = TrainingRunLogger(run_log_dir)
    training_logger.write_json("config.json", dataclasses.asdict(cfg))
    training_logger.log_event(event="run_start", run_id=run_id)
    training_logger.log_event(
        event="episode_start",
        episode=env.episode,
        state=env._state(),
        initial_humans=env.get_initial_humans(),
        personas=list(env.personas),
    )
    print(f"[LOG] training artifacts => {run_log_dir}")
    interaction_counter = itertools.count(start=1)

    # （当面の起動確認用）ダミーの学習データセット
    _dummy_ds = Dataset.from_dict({"prompt": ["dummy"]})

    # 合計ステップ数＝ total_updates * batch_size に合わせた有限データセットを作る
    total_updates = _to_int(getattr(ppo_cfg, "total_updates", 50), 50)
    total_samples = total_updates * max(1, _bs)
    train_ds = FiniteOnlineDataset(tokenizer, env, build_robot_messages, total_samples=total_samples)
    eval_samples = max(1, _bs)
    eval_ds = FiniteOnlineDataset(tokenizer, env, build_robot_messages, total_samples=eval_samples)

    # TRL 側で要求されるバッチ回数を total_updates に合わせる
    ppo_config.total_episodes = total_samples
    ppo_config.num_train_epochs = 1

    # 生成ハイパパラメータ（参考：必要なら trainer 側に渡す）
    gen_kwargs = dict(
        do_sample=True,
        temperature=_to_float(getattr(ppo_cfg, "temperature", 0.7), 0.7),
        top_p=_to_float(getattr(ppo_cfg, "top_p", 0.9), 0.9),
        max_new_tokens=_to_int(getattr(ppo_cfg, "max_new_tokens", 64), 64),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # TRL の PPOTrainer は args.response_length / args.temperature を参照して
    # GenerationConfig を構成するため、ここで揃えておく
    ppo_config.response_length = gen_kwargs["max_new_tokens"]
    ppo_config.temperature = gen_kwargs["temperature"]
    ppo_config.stop_token_id = gen_kwargs["eos_token_id"]
    ppo_config.num_sample_generations = 0
    ppo_config.per_device_eval_batch_size = max(1, min(eval_samples, getattr(ppo_cfg, "per_device_eval_batch_size", eval_samples)))
    rollout_forward = getattr(ppo_cfg, "local_rollout_forward_batch_size", None)
    if rollout_forward is not None:
        ppo_config.local_rollout_forward_batch_size = max(1, _to_int(rollout_forward, 1))

    entropy_floor_value = float(getattr(ppo_cfg, "entropy_floor", 0.0) or 0.0)
    entropy_patience_value = max(1, int(getattr(ppo_cfg, "entropy_patience", 1) or 1))
    entropy_monitor_warmup = max(0, int(getattr(ppo_cfg, "entropy_monitor_warmup", 0) or 0))

    # 報酬関数：生成テキストを環境に流し込み、Before→After のスコア差を返す
    # PPOTrainer 0.23 では `reward_model` 経由の計算が前提のため、後段で monkeypatch して利用する
    def reward_fn(samples, **kwargs):
        rewards = []
        fallback_penalty_value = float(
            getattr(ppo_cfg, "fallback_penalty", getattr(ppo_cfg, "non_japanese_penalty", 0.0)) or 0.0
        )
        repetition_penalty_value = float(getattr(ppo_cfg, "repetition_penalty", 0.0) or 0.0)
        repetition_lookback = int(getattr(ppo_cfg, "repetition_lookback", 1) or 1)
        repetition_semantic_penalty_value = float(getattr(ppo_cfg, "repetition_semantic_penalty", 0.0) or 0.0)
        repetition_semantic_threshold = float(getattr(ppo_cfg, "repetition_semantic_threshold", 0.9) or 0.9)
        if repetition_semantic_threshold <= 0.0:
            repetition_semantic_threshold = 0.9
        repetition_semantic_threshold = min(max(repetition_semantic_threshold, 0.0), 0.999)
        topic_similarity_weight = float(getattr(ppo_cfg, "topic_overlap_weight", 0.0) or 0.0)
        topic_miss_penalty_value = float(getattr(ppo_cfg, "topic_miss_penalty", 0.0) or 0.0)
        _threshold_raw = getattr(ppo_cfg, "topic_similarity_threshold", None)
        if _threshold_raw is None:
            _threshold_raw = getattr(ppo_cfg, "topic_overlap_min_tokens", 0.0)
        topic_similarity_threshold = float(_threshold_raw or 0.0)
        topic_overlap_horizon = max(0, int(getattr(ppo_cfg, "topic_overlap_horizon", 1) or 0))
        azure_endpoint = getattr(cfg.openai, "azure_endpoint", None)
        azure_embed_deployment = getattr(
            cfg.openai,
            "embedding_deployment",
            getattr(ppo_cfg, "topic_embed_model_name", None),
        )
        azure_embed_api_version = (
            getattr(cfg.openai, "embedding_api_version", None)
            or getattr(cfg.openai, "azure_api_version", None)
            or getattr(cfg.openai, "api_version", None)
            or "2024-02-01"
        )
        azure_api_key = getattr(cfg.openai, "azure_api_key", None) or ""
        brevity_penalty_value = float(getattr(ppo_cfg, "brevity_penalty", 0.0) or 0.0)
        brevity_min_chars = max(0, int(getattr(ppo_cfg, "brevity_min_chars", 0) or 0))
        topic_similarity_enabled = (
            topic_overlap_horizon > 0
            and (topic_similarity_weight != 0.0 or topic_miss_penalty_value != 0.0 or topic_similarity_threshold > 0.0)
            and azure_endpoint
            and azure_embed_deployment
            and azure_api_key
        )
        for resp in samples:
            raw_resp = resp or ""
            visible_resp = _strip_think_tags(raw_resp)
            primary_candidate = _extract_primary_japanese_sentence(visible_resp)
            preprocessed_candidate = _enforce_single_sentence(primary_candidate or "")
            robot_response = _prepare_robot_response(raw_resp)
            fallback_used = robot_response == _FALLBACK_SENTENCE and robot_response != preprocessed_candidate
            history_before = [dict(l) for l in env.logs]
            recent_robot_utts: list[str] = []
            if repetition_penalty_value and repetition_lookback > 0:
                for prev in reversed(history_before):
                    if prev.get("speaker") != "ロボット":
                        continue
                    normalized = _prepare_robot_response(prev.get("utterance", ""))
                    if normalized:
                        recent_robot_utts.append(normalized)
                    if len(recent_robot_utts) >= repetition_lookback:
                        break
            recent_human_utts: list[str] = []
            if topic_similarity_enabled:
                for prev in reversed(history_before):
                    if prev.get("speaker") == "ロボット":
                        continue
                    utt = (prev.get("utterance") or "").strip()
                    if utt:
                        recent_human_utts.append(utt)
                    if len(recent_human_utts) >= topic_overlap_horizon:
                        break
                recent_human_utts = list(reversed(recent_human_utts))
            interaction_id = next(interaction_counter)

            if training_logger is not None:
                training_logger.log_llm_response(
                    interaction_id=interaction_id,
                    episode=env.episode,
                    turn=env.t,
                    draft_response=raw_resp,
                    sanitized_response=preprocessed_candidate,
                    fallback_used=fallback_used,
                    history_before=history_before,
                    personas=list(env.personas),
                    state_before=env._state(),
                )

            if fallback_used:
                base_breakdown = {"fallback_penalty": fallback_penalty_value, "language_penalty": 0.0}
                components = _extract_reward_components(base_breakdown)
                total_reward = sum(components.values()) if components else float(fallback_penalty_value)
                reward_summary = {
                    "total": total_reward,
                    "components": components,
                }
                triangle_summary = ""
                if hasattr(env, "relation_snapshot"):
                    try:
                        triangle_summary = _format_triangle_summary(env.relation_snapshot())
                    except Exception:
                        triangle_summary = ""
                log_payload = {
                    "interaction_id": interaction_id,
                    "episode": env.episode,
                    "turn": env.t,
                    "robot": {
                        "utterance": robot_response,
                        "flags": ["fallback"],
                    },
                    "humans": [],
                    "reward": reward_summary,
                    "done": False,
                }
                if triangle_summary:
                    log_payload["triangle_summary"] = triangle_summary
                training_logger.log_interaction(**log_payload)
                rewards.append(total_reward)
                continue

            # 環境1ステップ進める（ロボット発話 = 生成テキスト）
            state_after, r, done, info = env.step(robot_response)
            history_after = [dict(l) for l in env.logs]
            base_breakdown = dict(info.get("reward_breakdown", {}) or {})
            language_penalty = 0.0
            if repetition_penalty_value and recent_robot_utts:
                if robot_response in recent_robot_utts:
                    language_penalty += repetition_penalty_value
                    base_breakdown["repeat_penalty"] = repetition_penalty_value
                    base_breakdown["repeat_match_count"] = recent_robot_utts.count(robot_response)
                    base_breakdown["repeat_lookback"] = len(recent_robot_utts)
            if repetition_semantic_penalty_value > 0.0 and recent_robot_utts:
                best_similarity = 0.0
                for prev_resp in recent_robot_utts:
                    if not prev_resp:
                        continue
                    best_similarity = max(best_similarity, _text_similarity(robot_response, prev_resp))
                if best_similarity >= repetition_semantic_threshold:
                    semantic_delta = max(0.0, best_similarity - repetition_semantic_threshold)
                    semantic_penalty = -repetition_semantic_penalty_value * semantic_delta
                    language_penalty += semantic_penalty
                    base_breakdown["repeat_semantic_penalty"] = semantic_penalty
                    base_breakdown["repeat_semantic_similarity"] = best_similarity
                    base_breakdown["repeat_semantic_threshold"] = repetition_semantic_threshold
            base_breakdown["language_penalty"] = language_penalty
            components = _extract_reward_components(base_breakdown)
            total_reward = sum(components.values()) if components else (float(r) + language_penalty)
            reward_summary = {
                "total": total_reward,
                "components": components,
            }

            topic_similarity_log = None
            if topic_similarity_enabled and recent_human_utts and robot_response:
                helper_key = (str(azure_endpoint), str(azure_embed_deployment), str(azure_embed_api_version))
                helper: Dict[str, Any] | None = None
                try:
                    helper = _get_azure_embedding_helper(
                        endpoint=str(azure_endpoint),
                        deployment=str(azure_embed_deployment),
                        api_version=str(azure_embed_api_version),
                        api_key=str(azure_api_key),
                    )
                    embeddings = _encode_embeddings_azure(
                        [robot_response] + recent_human_utts,
                        helper,
                    )
                except Exception as exc:
                    if helper_key not in _AZURE_EMBED_FAILURES:
                        _AZURE_EMBED_FAILURES.add(helper_key)
                        print(f"[warn] semantic embedding failed for {helper_key}: {exc}")
                        if training_logger is not None:
                            training_logger.log_event(
                                event="semantic_embedding_error",
                                error=str(exc),
                                endpoint=str(azure_endpoint),
                                deployment=str(azure_embed_deployment),
                            )
                    embeddings = None
                if embeddings is not None and embeddings.size(0) >= 2:
                    robot_vec = embeddings[0]
                    human_vecs = embeddings[1:]
                    if human_vecs.size(0) > 0:
                        similarities = torch.matmul(human_vecs, robot_vec)
                        max_similarity = float(torch.max(similarities).item()) if similarities.numel() > 0 else 0.0
                        mean_similarity = float(torch.mean(similarities).item()) if similarities.numel() > 0 else 0.0
                        similarity_list = similarities.tolist()
                    else:
                        similarities = None
                        max_similarity = 0.0
                        mean_similarity = 0.0
                        similarity_list = []

                    topic_bonus = 0.0
                    if topic_similarity_weight != 0.0 and similarities is not None and similarities.numel() > 0:
                        topic_bonus = topic_similarity_weight * max_similarity
                        if topic_bonus != 0.0:
                            total_reward += topic_bonus
                            base_breakdown["topic_similarity_bonus"] = topic_bonus
                            base_breakdown["topic_similarity_max"] = max_similarity
                            base_breakdown["topic_similarity_mean"] = mean_similarity

                    topic_penalty_applied = 0.0
                    if (
                        topic_miss_penalty_value != 0.0
                        and similarities is not None
                        and similarities.numel() > 0
                        and max_similarity < topic_similarity_threshold
                    ):
                        topic_penalty_applied = topic_miss_penalty_value
                        total_reward += topic_penalty_applied
                        base_breakdown["topic_similarity_penalty"] = topic_penalty_applied
                        base_breakdown["topic_similarity_threshold"] = topic_similarity_threshold

                    if (similarities is not None and similarities.numel() > 0) or topic_bonus != 0.0 or topic_penalty_applied != 0.0:
                        topic_similarity_log = {
                            "endpoint": str(azure_endpoint),
                            "deployment": str(azure_embed_deployment),
                            "api_version": str(azure_embed_api_version),
                            "max_similarity": max_similarity,
                            "mean_similarity": mean_similarity,
                            "threshold": topic_similarity_threshold,
                            "bonus": topic_bonus,
                            "penalty": topic_penalty_applied,
                            "horizon": topic_overlap_horizon,
                            "per_human": [
                                {
                                    "utterance": utt,
                                    "similarity": float(sim),
                                }
                                for utt, sim in zip(recent_human_utts, similarity_list)
                            ],
                        }

            brevity_log = None
            if brevity_penalty_value != 0.0 and brevity_min_chars > 0:
                visible_chars = len(re.sub(r"\s+", "", robot_response))
                if visible_chars < brevity_min_chars:
                    total_reward += brevity_penalty_value
                    base_breakdown["brevity_penalty"] = brevity_penalty_value
                    base_breakdown["brevity_min_chars"] = brevity_min_chars
                    base_breakdown["brevity_char_count"] = visible_chars
                    brevity_log = {
                        "applied": True,
                        "value": brevity_penalty_value,
                        "char_count": visible_chars,
                        "min_required": brevity_min_chars,
                    }
                else:
                    brevity_log = {
                        "applied": False,
                        "value": 0.0,
                        "char_count": visible_chars,
                        "min_required": brevity_min_chars,
                    }

            log_payload = {
                "interaction_id": interaction_id,
                "episode": env.episode,
                "turn": env.t,
                "robot": {
                    "utterance": robot_response,
                },
                "humans": info.get("replies", []),
                "reward": reward_summary,
                "done": bool(done),
            }
            triangle_summary = _format_triangle_summary(info.get("rel"))
            if triangle_summary:
                log_payload["triangle_summary"] = triangle_summary
            if fallback_used:
                log_payload.setdefault("robot", {}).setdefault("flags", []).append("fallback")
            training_logger.log_interaction(**log_payload)
            rewards.append(total_reward)
            if done:
                current_episode = env.episode
                training_logger.log_event(
                    event="episode_end",
                    episode=current_episode,
                    reward=total_reward,
                    env_metrics=info.get("rel", {}),
                    reward_breakdown=base_breakdown,
                    personas=list(env.personas),
                    total_steps=env.total_steps,
                )
                next_state = env.reset()
                training_logger.log_event(
                    event="episode_start",
                    episode=env.episode,
                    state=next_state,
                    initial_humans=env.get_initial_humans(),
                    personas=list(env.personas),
                )
        return rewards

    # === DataLoader の shuffle パッチ（IterableDataset との互換性確保） ===
    if not getattr(trl_ppo_trainer, "_iterable_shuffle_patched", False):
        original_dataloader = TorchDataLoader
        iterable_types = (TorchIterableDataset,)
        if HFIterableDataset is not None:
            iterable_types = (HFIterableDataset,) + iterable_types

        def _safe_dataloader(*dl_args, **dl_kwargs):
            args = list(dl_args)
            dataset = args[0] if args else dl_kwargs.get("dataset")
            if isinstance(dataset, iterable_types):
                dl_kwargs.pop("shuffle", None)
                if len(args) >= 3 and isinstance(args[2], bool):
                    args[2] = False
                else:
                    dl_kwargs["shuffle"] = False
            return original_dataloader(*args, **dl_kwargs)

        trl_ppo_trainer.DataLoader = _safe_dataloader
        trl_ppo_trainer._iterable_shuffle_patched = True

    if not getattr(trl_ppo_trainer, "_forward_tuple_patched", False):
        _orig_forward_fn = trl_ppo_trainer.forward

        def _forward_tuple_safe(model, *args, **kwargs):
            out = _orig_forward_fn(model, *args, **kwargs)
            if isinstance(out, tuple):
                if len(out) == 2 and hasattr(out[0], "logits"):
                    return out
                first = out[0] if out else None
                if hasattr(first, "logits"):
                    return first
                if isinstance(first, torch.Tensor):
                    try:
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        return CausalLMOutputWithPast(logits=first)
                    except Exception:
                        return types.SimpleNamespace(logits=first)
            return out

        trl_ppo_trainer.forward = _forward_tuple_safe
        trl_ppo_trainer._forward_tuple_patched = True

    if not getattr(trl_utils, "_forward_tuple_safe", False):
        _orig_utils_forward = trl_utils.forward

        def _utils_forward_safe(model, query_responses, pad_token_id):
            out = _orig_utils_forward(model, query_responses, pad_token_id)
            if isinstance(out, tuple):
                first = out[0] if out else None
                if hasattr(first, "logits"):
                    return first
                if isinstance(first, torch.Tensor):
                    try:
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        return CausalLMOutputWithPast(logits=first)
                    except Exception:
                        return types.SimpleNamespace(logits=first)
            return out

        trl_utils.forward = _utils_forward_safe
        trl_utils._forward_tuple_safe = True

    if not getattr(PolicyAndValueWrapper, "_forward_tuple_safe", False):
        _orig_policy_value_forward = PolicyAndValueWrapper.forward

        def _policy_value_forward_safe(self, **kwargs):
            policy_output, value_logits = _orig_policy_value_forward(self, **kwargs)
            if isinstance(policy_output, tuple):
                first = policy_output[0] if len(policy_output) > 0 else None
                if hasattr(first, "logits"):
                    policy_output = first
                elif isinstance(first, torch.Tensor):
                    try:
                        from transformers.modeling_outputs import CausalLMOutputWithPast
                        policy_output = CausalLMOutputWithPast(logits=first)
                    except Exception:
                        policy_output = types.SimpleNamespace(logits=first)
                elif first is not None and hasattr(first, "__dict__") and hasattr(first, "logits"):
                    policy_output = first
                else:
                    policy_output = types.SimpleNamespace(logits=first)
            elif not hasattr(policy_output, "logits"):
                raise AttributeError("Policy output lacks logits attribute after patching")
            return policy_output, value_logits

        PolicyAndValueWrapper.forward = _policy_value_forward_safe
        PolicyAndValueWrapper._forward_tuple_safe = True

    def _collect_gc_targets(wrapper: PolicyAndValueWrapper):
        attrs = (
            "policy_model",
            "model",
            "policy",
            "pretrained_model",
        )
        seen = []
        for name in attrs:
            obj = getattr(wrapper, name, None)
            if obj is not None and obj not in seen:
                seen.append(obj)
        return seen

    if not hasattr(PolicyAndValueWrapper, "gradient_checkpointing_enable"):

        def _policy_gc_enable(self, use_cache: bool = False):
            for target in _collect_gc_targets(self):
                if hasattr(target, "gradient_checkpointing_enable"):
                    target.gradient_checkpointing_enable()
                if hasattr(target, "config") and target.config is not None:
                    try:
                        target.config.use_cache = use_cache
                    except Exception:
                        pass

        PolicyAndValueWrapper.gradient_checkpointing_enable = _policy_gc_enable

    if not hasattr(PolicyAndValueWrapper, "gradient_checkpointing_disable"):

        def _policy_gc_disable(self):
            for target in _collect_gc_targets(self):
                if hasattr(target, "gradient_checkpointing_disable"):
                    target.gradient_checkpointing_disable()

        PolicyAndValueWrapper.gradient_checkpointing_disable = _policy_gc_disable

    dummy_reward_model = DummyRewardModel()

    def _env_get_reward(model, query_responses, pad_token_id, context_length):
        device = query_responses.device
        response_tokens = query_responses[:, context_length:]

        special_remove = {pad_token_id, tokenizer.eos_token_id, getattr(tokenizer, "bos_token_id", None)}
        special_remove.discard(None)

        texts = []
        for row in response_tokens:
            ids = [int(t) for t in row.tolist() if int(t) not in special_remove]
            if not ids:
                texts.append("")
            else:
                texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

        reward_values = reward_fn(texts)
        rewards_tensor = torch.as_tensor(reward_values, dtype=torch.float32, device=device)

        if pad_token_id is None:
            pad_mask = torch.zeros_like(response_tokens, dtype=torch.bool)
        else:
            pad_mask = response_tokens == pad_token_id

        offsets = first_true_indices(pad_mask, dtype=torch.long) - 1
        offsets = torch.clamp(offsets, min=0)
        sequence_lengths = torch.clamp(offsets + context_length, max=query_responses.size(1) - 1)

        reward_logits = torch.zeros(
            query_responses.size(0),
            query_responses.size(1),
            dtype=torch.float32,
            device=device,
        )
        reward_logits[torch.arange(reward_logits.size(0), device=device), sequence_lengths] = rewards_tensor
        return reward_logits, rewards_tensor, sequence_lengths

    trl_ppo_trainer.get_reward = _env_get_reward

    # TRL 0.23: train() が data["input_ids"] を参照するため
    # collator でトークナイズして渡す
    def ppo_collator(features):
        """Torch IterableDataset からの list[dict] をトークナイズする。"""
        if not isinstance(features, list) or not all(isinstance(x, dict) for x in features):
            raise TypeError("ppo_collator expects a list[dict] from a torch IterableDataset")

        texts = [ex.get("query", "") for ex in features]

        if not texts or all((t is None) or (str(t).strip() == "") for t in texts):
            texts = ["(empty)"]

        toks = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(2048, getattr(tokenizer, "model_max_length", 2048)),
        )
        return toks

    # PPOTrainer 内部で生成設定を上書きできるよう GenerationConfig をラップ
    if gen_kwargs and not hasattr(trl_ppo_trainer, "_generation_config_patched"):
        from transformers import GenerationConfig as _HFGenerationConfig

        class _GenerationConfigWithDefaults(_HFGenerationConfig):
            def __init__(self, *args, **kwargs):
                merged = {**gen_kwargs, **kwargs}
                super().__init__(*args, **merged)

        trl_ppo_trainer.GenerationConfig = _GenerationConfigWithDefaults
        trl_utils.GenerationConfig = _GenerationConfigWithDefaults
        trl_ppo_trainer._generation_config_patched = True

    trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,            # = tokenizer
        model=policy,
        ref_model=ref_policy,
        reward_model=dummy_reward_model,
        train_dataset=train_ds,                # IterableDataset で "query" を供給
        eval_dataset=eval_ds,
        value_model=getattr(policy, "pretrained_model", policy),
        data_collator=ppo_collator,           # ← これが重要
    )

    class AdaptiveKLCallback(TrainerCallback):
        def __init__(
            self,
            trainer_obj,
            target: float,
            adjust_up: float,
            adjust_down: float,
            logger: TrainingRunLogger | None = None,
            increase_margin: float = 1.1,
            decrease_margin: float = 0.6,
            min_coef: float = 1e-5,
            max_coef: float = 10.0,
        ) -> None:
            self.trainer = trainer_obj
            self.target = max(0.0, float(target))
            self.adjust_up = max(1.0, float(adjust_up))
            self.adjust_down = float(adjust_down)
            if not (0.0 < self.adjust_down < 1.0):
                self.adjust_down = 0.9
            self.logger = logger
            self.increase_margin = max(1.0, float(increase_margin))
            self.decrease_margin = max(0.0, float(decrease_margin))
            self.min_coef = max(0.0, float(min_coef))
            self.max_coef = max(self.min_coef + 1e-6, float(max_coef))
            self._last_step = None

        def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
            if logs is None or self.target <= 0.0:
                return control
            approx = logs.get("policy/approxkl_avg")
            if approx is None:
                approx = logs.get("objective/kl")
            if approx is None:
                return control
            try:
                approx_val = float(approx)
            except Exception:
                return control
            if not math.isfinite(approx_val):
                return control

            current_coef = float(getattr(self.trainer.args, "kl_coef", 0.0))
            if current_coef <= 0.0:
                return control

            step = int(getattr(state, "global_step", 0) or 0)
            if self._last_step == step:
                return control

            new_coef = current_coef
            action = None
            if approx_val > self.target * self.increase_margin:
                new_coef = min(self.max_coef, max(self.min_coef, current_coef * self.adjust_up))
                action = "increase"
            elif approx_val < self.target * self.decrease_margin:
                new_coef = max(self.min_coef, min(self.max_coef, current_coef * self.adjust_down))
                action = "decrease"

            if action is None or abs(new_coef - current_coef) < 1e-12:
                return control

            self._last_step = step
            setattr(self.trainer.args, "kl_coef", new_coef)
            if hasattr(self.trainer, "config"):
                setattr(self.trainer.config, "kl_coef", new_coef)

            print(
                f"[adaptive_kl] step={step} approxkl={approx_val:.4f} target={self.target:.4f} kl_coef: {current_coef:.5f} -> {new_coef:.5f} ({action})"
            )
            if self.logger is not None:
                try:
                    self.logger.log_event(
                        event="adaptive_kl_update",
                        step=step,
                        approxkl=approx_val,
                        target=self.target,
                        action=action,
                        old_coef=current_coef,
                        new_coef=new_coef,
                    )
                except Exception:
                    pass
            return control

    adaptive_kl_callback = None
    if _target_kl > 0.0 and _kl > 0.0:
        adaptive_kl_callback = AdaptiveKLCallback(
            trainer_obj=trainer,
            target=_target_kl,
            adjust_up=_kl_adjust_up,
            adjust_down=_kl_adjust_down,
            logger=training_logger,
        )
        trainer.add_callback(adaptive_kl_callback)
        if training_logger is not None:
            training_logger.log_event(
                event="adaptive_kl_enabled",
                target=_target_kl,
                adjust_up=_kl_adjust_up,
                adjust_down=_kl_adjust_down,
                initial_kl_coef=_kl,
            )

    if entropy_floor_value > 0.0 and entropy_patience_value > 0:
        class _EntropyMonitorCallback(TrainerCallback):
            def __init__(self, floor: float, patience: int, warmup: int, logger: TrainingRunLogger | None = None):
                self.floor = float(floor)
                self.patience = max(1, int(patience))
                self.warmup = max(0, int(warmup))
                self.logger = logger
                self._streak = 0
                self._triggered = False

            def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
                if logs is None:
                    return control
                if self._triggered:
                    return control
                step = getattr(state, "global_step", 0) or 0
                if step <= self.warmup:
                    self._streak = 0
                    return control
                entropy_val = logs.get("policy/entropy_avg")
                if entropy_val is None:
                    return control
                try:
                    entropy_float = float(entropy_val)
                except Exception:
                    return control
                if entropy_float < self.floor:
                    self._streak += 1
                    if self._streak == 1 and self.logger is not None:
                        self.logger.log_event(
                            event="entropy_floor_warning",
                            step=step,
                            entropy=entropy_float,
                            floor=self.floor,
                        )

                    if self._streak >= self.patience:
                        control.should_training_stop = True
                        self._triggered = True
                        if self.logger is not None:
                            self.logger.log_event(
                                event="entropy_floor_triggered",
                                step=step,
                                entropy=entropy_float,
                                floor=self.floor,
                                patience=self.patience,
                            )
                        print(
                            f"[monitor] policy/entropy_avg {entropy_float:.4f} below floor {self.floor:.4f}; early stopping triggered."
                        )
                else:
                    if self._streak > 0 and self.logger is not None:
                        self.logger.log_event(
                            event="entropy_floor_recovered",
                            step=step,
                            entropy=entropy_float,
                            floor=self.floor,
                            streak=self._streak,
                        )
                    self._streak = 0
                return control

        entropy_callback = _EntropyMonitorCallback(
            floor=entropy_floor_value,
            patience=entropy_patience_value,
            warmup=entropy_monitor_warmup,
            logger=training_logger,
        )
        trainer.add_callback(entropy_callback)
        trainer._entropy_monitor_callback = entropy_callback  # type: ignore[attr-defined]
        if training_logger is not None:
            training_logger.log_event(
                event="entropy_monitor_enabled",
                floor=entropy_floor_value,
                patience=entropy_patience_value,
                warmup=entropy_monitor_warmup,
            )

    if training_logger is not None:
        _orig_trainer_log = trainer.log

        def _logger_log(self, logs, start_time=None):
            try:
                training_logger.log_metrics(
                    split="train",
                    metrics=logs,
                    step=self.state.global_step,
                    epoch=self.state.epoch,
                )
            except Exception:
                pass
            return _orig_trainer_log(logs, start_time=start_time)

        trainer.log = types.MethodType(_logger_log, trainer)

        _orig_trainer_log_metrics = trainer.log_metrics

        def _logger_log_metrics(self, split, metrics):
            try:
                training_logger.log_metrics(
                    split=split,
                    metrics=metrics,
                    step=self.state.global_step,
                    epoch=self.state.epoch,
                )
            except Exception:
                pass
            return _orig_trainer_log_metrics(split, metrics)

        trainer.log_metrics = types.MethodType(_logger_log_metrics, trainer)

    # # 参照側は勾配を止めておく（KL 用の固定モデル扱い）
    # if getattr(trainer, "ref_model", None) is not None:
    #     for p in trainer.ref_model.parameters():
    #         p.requires_grad = False

    # ★★ 最重要：Accelerate 準備後の“実体”を直接パッチして tuple→ModelOutput を保証
    for _name in ("policy_model", "ref_model", "value_model"):
        _m = getattr(trainer, _name, None)
        if _m is not None:
            _force_model_output_tuple_safe(_m)

    # --- 学習実行（内部で: 生成→報酬→PPO更新） ---
    total_updates  = _to_int(getattr(ppo_cfg, "total_updates", 50), 50)
    out_root = Path(getattr(ppo_cfg, "output_dir", "../models/ppo_robot"))
    out_root.mkdir(parents=True, exist_ok=True)
    model_run_dir = out_root / f"ppo_run-{run_id}"
    model_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[MODEL] training artifacts => {model_run_dir}")

    # TRL 0.23: train() は引数なし。データセットが尽きるまで学習。
    # ちょうど total_updates 回の PPO 更新にしたいので、上で total_samples を調整済み。
    run_status = "completed"
    try:
        trainer.train()
        final_dir = model_run_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_dir))
        if training_logger is not None:
            training_logger.log_event(event="model_saved", path=str(final_dir))

        try:
            policy_to_save = _unwrap_module(trainer.model)
            if hasattr(policy_to_save, "save_pretrained"):
                policy_to_save.save_pretrained(str(final_dir))
                if training_logger is not None:
                    training_logger.log_event(event="adapter_saved", path=str(final_dir))
                print(f"[save] LoRA adapters saved to {final_dir}")
            else:
                print("[save] No save_pretrained method on policy; skipping adapter export.")
        except Exception as exc:
            print(f"[warn] Failed to save LoRA adapters: {exc}")
            if training_logger is not None:
                training_logger.log_event(event="adapter_save_failed", error=str(exc), path=str(final_dir))
        print("Done.")
    except Exception as exc:
        run_status = "failed"
        if training_logger is not None:
            training_logger.log_event(event="run_exception", error=str(exc))
        raise
    finally:
        if training_logger is not None:
            training_logger.log_event(event="run_end", status=run_status)
            training_logger.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="モデルと環境の初期化のみ実行")
    args = p.parse_args()
    if args.dry_run:
        cfg = get_config()
        print("[CFG]", dataclasses.asdict(cfg.ppo))
        print("Dry run OK (no training).")
    else:
        main()
