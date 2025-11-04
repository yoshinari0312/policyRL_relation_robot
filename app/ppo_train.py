# 強化学習(PPO)で「ロボットの発言」方策を更新する最小スケルトン
# - TRL PPO + LoRA (PEFT)
# - 既存 env.ConversationEnv を 1 ステップ環境として用い、
#   報酬 = 不安定三角形の減少量 (Before-After) をそのまま使用
# - 収集: 1 ステップ × (per_device_train_batch_size × grad_accum_steps × 2) サンプルで1回の PPO 更新
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
from typing import List, Tuple, Dict, Any, Optional
from functools import lru_cache
from collections import defaultdict
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
from batch_summary import BatchSummaryCollector

# wandb for visualization (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


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


def _validate_plan_json(text: str) -> tuple[dict | None, bool]:
    """
    介入判定JSONの妥当性をチェック

    Args:
        text: 検証対象のテキスト

    Returns:
        (parsed_plan_dict, is_valid): パースされた辞書とバリデーション結果
    """
    if not text or not isinstance(text, str):
        return None, False

    text = text.strip()
    if not text:
        return None, False

    # JSONパース試行
    try:
        # JSONっぽい部分を抜き出す（既存ロジック）
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed_obj = json.loads(snippet)
            except json.JSONDecodeError:
                parsed_obj = json.loads(text)
        else:
            parsed_obj = json.loads(text)
    except json.JSONDecodeError:
        return None, False

    # 辞書型かチェック
    if not isinstance(parsed_obj, dict):
        return None, False

    # 必須フィールドのチェック
    required_fields = ["intervene_now"]
    for field in required_fields:
        if field not in parsed_obj:
            return None, False

    # intervene_nowがtrueの場合、追加フィールドが必要
    intervene_now = parsed_obj.get("intervene_now")
    if intervene_now:
        additional_required = ["strategy", "edge_to_change", "target_speaker"]
        for field in additional_required:
            if field not in parsed_obj:
                return None, False

        # 値の妥当性チェック
        strategy = parsed_obj.get("strategy")
        if strategy not in ["plan", "validate", "bridge"]:
            return None, False

        edge_to_change = str(parsed_obj.get("edge_to_change", "")).upper()
        if edge_to_change not in ["AB", "BC", "CA"]:
            return None, False

        target_speaker = str(parsed_obj.get("target_speaker", "")).upper()
        if target_speaker not in ["A", "B", "C"]:
            return None, False

        # edge_to_changeとtarget_speakerの整合性チェック
        valid_targets = {
            "AB": ["A", "B"],
            "BC": ["B", "C"],
            "CA": ["C", "A"]
        }

        if edge_to_change in valid_targets:
            if target_speaker not in valid_targets[edge_to_change]:
                # デバッグ用: 不整合の詳細をログ出力
                if hasattr(get_config(), 'env') and getattr(get_config().env, 'debug', False):
                    print(f"[JSON validation] 不整合検出: edge_to_change='{edge_to_change}' target_speaker='{target_speaker}' (期待値: {valid_targets[edge_to_change]})")
                return None, False

    return parsed_obj, True


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
    """<think> タグで囲まれた内側のテキストを削除する。"""
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

    def __init__(self, run_dir: Path, enable_wandb: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._convo_path = self.run_dir / "conversation_summary.jsonl"
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._events_path = self.run_dir / "run_events.jsonl"
        self._batch_path = self.run_dir / "batch_summary.jsonl"  # バッチ統計専用ファイル
        self._convo_fp = open(self._convo_path, "a", encoding="utf-8")
        self._metrics_fp = open(self._metrics_path, "a", encoding="utf-8")
        self._events_fp = open(self._events_path, "a", encoding="utf-8")
        self._batch_fp = open(self._batch_path, "a", encoding="utf-8")  # バッチ統計ファイル
        atexit.register(self.close)

        # wandb初期化
        self.use_wandb = enable_wandb and WANDB_AVAILABLE
        self._wandb_run = None
        self._strategy_table_data = []  # 戦略テーブルのデータ
        self._table_log_frequency = 50  # N回ごとにwandb.logを呼ぶ
        self._interaction_count = 0

        # Rolling window for intervention rate tracking
        self._intervention_window = []  # 介入判定の履歴 (True/False)
        self._window_size = 50  # 直近50回の決定で介入率を計算

        if self.use_wandb:
            try:
                # wandbが既に初期化されているかチェック
                if wandb.run is None:
                    # プロジェクト名を設定
                    run_name = self.run_dir.name
                    self._wandb_run = wandb.init(
                        project=get_config().wandb.project,
                        entity=get_config().wandb.entity,
                        name=run_name,
                        dir=str(self.run_dir.parent),
                        reinit=True,
                    )
                else:
                    self._wandb_run = wandb.run
                print(f"✓ wandb initialized: {self._wandb_run.name}")
            except Exception as e:
                print(f"⚠ wandb initialization failed: {e}")
                self.use_wandb = False

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

    def _sanitize(self, obj, parent_key=None):
        if isinstance(obj, dict):
            safe = {}
            for k, v in obj.items():
                if isinstance(k, (str, int, float, bool)) or k is None:
                    safe_key = k
                elif isinstance(k, (tuple, list, set)):
                    safe_key = ",".join(str(item) for item in k)
                else:
                    safe_key = str(k)
                safe[safe_key] = self._sanitize(v, parent_key=safe_key)
            return safe
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize(v, parent_key=parent_key) for v in obj]
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            try:
                return self._sanitize(obj.item(), parent_key=parent_key)
            except Exception:
                pass
        if isinstance(obj, float):
            # 学習率などの小さい値は丸めずに保持
            if parent_key in ('lr', 'learning_rate', 'eps', 'lr_scheduler_lr'):
                return float(obj)
            return self._round_float(obj)
        if isinstance(obj, (str, int, bool)) or obj is None:
            return obj
        return str(obj)

    def _write_jsonl(self, fp, payload: dict):
        record = {
            **payload,
            "timestamp": payload.get("timestamp") or datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        # インデント付きで見やすく整形（1レコード=複数行）
        fp.write(json.dumps(self._sanitize(record), ensure_ascii=False, indent=2))
        fp.write("\n" + "="*80 + "\n")  # レコード区切り線
        fp.flush()

    def log_interaction(self, **payload):
        self._write_jsonl(self._convo_fp, payload)

        # wandb.Tableにデータを追加
        if self.use_wandb:
            self._interaction_count += 1

            # 介入判定があった場合は常に記録（介入する・しないの両方）
            intervention = payload.get("intervention")
            if intervention and isinstance(intervention, dict):
                # 介入した場合
                if intervention.get("intervened"):
                    strategy = intervention.get("strategy", "unknown")
                    edge = intervention.get("edge_to_change", "unknown")
                    robot_utterance = intervention.get("robot_utterance", "")
                    target_speaker = intervention.get("target_speaker", "")
                    decision_type = "intervention"
                else:
                    # 介入しなかった場合
                    strategy = "no_intervention"
                    edge = "none"
                    robot_utterance = ""
                    target_speaker = ""
                    decision_type = "no_intervention"

                # 関係性情報
                relation = payload.get("relation", {})
                is_stable_before = relation.get("is_stable", False)
                edges_before = relation.get("edges", {})

                # horizon後の関係性（介入した場合のみ）
                if intervention.get("intervened"):
                    relation_after = payload.get("relations_after_horizon", {})
                    is_stable_after = relation_after.get("is_stable", False)
                    edges_after = relation_after.get("edges", {})
                else:
                    # 介入しなかった場合は関係性変化なし
                    is_stable_after = is_stable_before
                    edges_after = edges_before

                # 報酬情報
                reward_info = payload.get("reward", {})
                total_reward = reward_info.get("total", 0.0)

                # 戦略テーブルに追加（target_speakerとutteranceを追加）
                self._strategy_table_data.append([
                    self._interaction_count,  # step
                    payload.get("episode", 0),  # episode
                    payload.get("turn", 0),  # turn
                    strategy,  # strategy
                    edge,  # target_edge
                    target_speaker,  # target_speaker
                    robot_utterance[:200] if robot_utterance else "",  # utterance (truncated)
                    is_stable_before,  # stable_before
                    is_stable_after,  # stable_after
                    total_reward,  # reward
                    decision_type,  # intervention_type (intervention/no_intervention)
                ])

                # Rolling window介入率の更新
                intervened = intervention.get("intervened", False)
                self._intervention_window.append(intervened)

                # Window sizeを超えた場合は古いデータを削除
                if len(self._intervention_window) > self._window_size:
                    self._intervention_window.pop(0)

                # リアルタイム介入率を計算してログ
                if len(self._intervention_window) >= 5:  # 最低5回の判定があった場合
                    window_intervention_rate = sum(self._intervention_window) / len(self._intervention_window)

                    # 個別interactionごとにwandبログ
                    if self._wandb_run:
                        try:
                            wandb.log({
                                "intervention/rolling_rate": window_intervention_rate,
                                "intervention/rolling_window_size": len(self._intervention_window),
                                "intervention/current_decision": "intervention" if intervened else "no_intervention",
                                "step": self._interaction_count,
                            })
                        except Exception as e:
                            if getattr(env, "debug", False):
                                print(f"[wandb] rolling rate log failed: {e}")

            # 定期的にwandب.logを呼ぶ
            if self._interaction_count % self._table_log_frequency == 0:
                self._flush_wandb_tables()

    def log_metrics(self, split: str, metrics: dict, step: int | None = None, epoch: float | None = None):
        clean = self._sanitize(metrics or {})
        payload = {"split": split, "metrics": clean}
        if step is not None:
            payload["step"] = step
        if epoch is not None:
            payload["epoch"] = epoch
        self._write_jsonl(self._metrics_fp, payload)

    def log_batch_summary(self, batch_summary: dict, episode: int | None = None, step: int | None = None):
        """バッチサマリーを専用ファイルに記録"""
        payload = {
            "batch_summary": batch_summary,
        }
        if episode is not None:
            payload["episode"] = episode
        if step is not None:
            payload["step"] = step
        self._write_jsonl(self._batch_fp, payload)

    def log_event(self, **payload):
        event = payload.pop("event", "info")
        record: Dict[str, Any] = {"event": event}
        for key in ("run_id", "episode", "step", "total_steps"):
            if key in payload:
                record[key] = payload[key]

        if event == "run_start" and "state" in payload:
            record["state"] = payload["state"]

        if event == "episode_start":
            if payload.get("topic"):
                record["topic"] = payload["topic"]
        elif event == "episode_end":
            record["reward"] = {
                "total": payload.get("reward"),
                "components": _extract_reward_components(payload.get("reward_breakdown")),
            }
            if payload.get("env_metrics"):
                unstable_count = payload["env_metrics"].get("unstable_triads", 0)
                record["env_summary"] = {
                    "is_stable": unstable_count == 0,
                }
            # エピソード統計を追加
            if "episode_total_reward" in payload:
                record["episode_total_reward"] = payload["episode_total_reward"]
            if "episode_intervention_count" in payload:
                record["episode_intervention_count"] = payload["episode_intervention_count"]

        other_keys = {k: v for k, v in payload.items() if k not in {"state", "personas", "reward", "reward_breakdown", "env_metrics", "episode_total_reward", "episode_intervention_count"}}
        if other_keys:
            record["details"] = other_keys

        self._write_jsonl(self._events_fp, record)

    def write_json(self, name: str, data: dict):
        target = self.run_dir / name
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self._sanitize(data), f, ensure_ascii=False, indent=2)

    def _flush_wandb_tables(self):
        """wandb.Tableをログに送信"""
        if not self.use_wandb or not self._wandb_run:
            return

        try:
            # 戦略テーブルを作成してログ
            if self._strategy_table_data:
                strategy_table = wandb.Table(
                    columns=["step", "episode", "turn", "strategy", "target_edge", "target_speaker",
                            "utterance", "stable_before", "stable_after", "reward", "decision_type"],
                    data=self._strategy_table_data
                )
                try:
                    wandb.log({
                        "strategy_table": strategy_table,
                        "step": self._interaction_count,
                    })
                except Exception as log_error:
                    print(f"⚠ wandb strategy table log failed: {log_error}")

                # 戦略の分布を集計してログ
                from collections import Counter
                strategy_counts = Counter([row[3] for row in self._strategy_table_data])
                decision_counts = Counter([row[10] for row in self._strategy_table_data])  # decision_type列
                total = sum(strategy_counts.values())

                if total > 0:
                    # 介入率の計算
                    intervention_count = decision_counts.get("intervention", 0)
                    no_intervention_count = decision_counts.get("no_intervention", 0)
                    intervention_rate = intervention_count / total if total > 0 else 0.0

                    try:
                        wandb.log({
                            # 戦略分布（介入した場合のみ）
                            "strategy/validate_ratio": strategy_counts.get("validate", 0) / total,
                            "strategy/bridge_ratio": strategy_counts.get("bridge", 0) / total,
                            "strategy/plan_ratio": strategy_counts.get("plan", 0) / total,
                            "strategy/no_intervention_ratio": strategy_counts.get("no_intervention", 0) / total,

                            # 戦略カウント
                            "strategy/validate_count": strategy_counts.get("validate", 0),
                            "strategy/bridge_count": strategy_counts.get("bridge", 0),
                            "strategy/plan_count": strategy_counts.get("plan", 0),
                            "strategy/no_intervention_count": strategy_counts.get("no_intervention", 0),

                            # 介入率メトリクス
                            "intervention/rate": intervention_rate,
                            "intervention/count": intervention_count,
                            "intervention/no_intervention_count": no_intervention_count,
                            "intervention/total_decisions": total,

                            "step": self._interaction_count,
                        })
                    except Exception as log_error:
                        print(f"⚠ wandb strategy metrics log failed: {log_error}")

            print(f"✓ wandb tables logged (interaction {self._interaction_count})")
        except Exception as e:
            print(f"⚠ wandb table logging failed: {e}")

    def close(self):
        # 最後にwandbテーブルをフラッシュ
        if self.use_wandb:
            self._flush_wandb_tables()
            # wandbをfinish
            if self._wandb_run:
                try:
                    self._wandb_run.finish()
                    print("✓ wandb run finished")
                except Exception as e:
                    print(f"⚠ wandb finish failed: {e}")

        for fp in (
            getattr(self, attr, None)
            for attr in ("_convo_fp", "_metrics_fp", "_events_fp", "_batch_fp")
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


_PLANNER_STRATEGIES = ("plan", "validate", "bridge")


def build_robot_messages(
    logs: List[dict],
    personas: List[str],
    env=None,
    k: int | None = None,
    model_name: str | None = None,
):
    """会話履歴と環境情報から介入計画を作成する"""

    prompt_text: Optional[str] = None
    if env is not None and hasattr(env, "_make_observation"):
        try:
            prompt_text = env._make_observation()
        except Exception:
            prompt_text = None

    # プロンプト最適化: 固定説明をシステムプロンプトに集約
    system_content = """あなたは関係性を安定させるロボットの介入計画を提案するAIです。

三者会話（話者 A/B/C）の関係を安定化するため、ロボットが適切なタイミングで一言介入します。
あなたの役割は、会話履歴と各ペアの関係スコア（-1..1）を受け取り、
「できるだけ早く関係性を安定状態（+++,+--,-+-,--+）にする」ための介入方法を提案することです。
※ロボットの実際の発話文は別LLMが生成します。あなたは介入方法だけを出力します。

制約:
- 出力は JSON のみ。説明や装飾は禁止。
- intervene_now は今すぐ介入すべきかを表す。 true|false のいずれか。
- edge_to_change はどの関係を変更するかを表す。 "AB" | "BC" | "CA" のいずれか。
- strategy は介入戦略を表す。 "plan" | "validate" | "bridge" のいずれか。
  - "plan": 対象者に具体的な行動計画を提案し、関係改善を促進
  - "validate": 対象者の感情・意見を承認し、心理的安全性を構築
  - "bridge": 対立する者の共通点・目標を明示し、協力関係を構築
  ※どの戦略をいつ使うかは会話文脈や関係スコアから判断してください。
- target_speaker はその介入を誰に向けるかを表す。 edge_to_changeで選んだ人物のいずれか。（"AB"→"A" | "B"、"BC"→"B" | "C"、"CA"→"C" | "A"） 。

出力例(あくまで例です。36通りの組み合わせがあります):
{"intervene_now": true, "edge_to_change": "AB", "strategy": "plan", "target_speaker": "A"}
{"intervene_now": true, "edge_to_change": "BC", "strategy": "validate", "target_speaker": "B"}
{"intervene_now": true, "edge_to_change": "CA", "strategy": "bridge", "target_speaker": "C"}
{"intervene_now": false}
"""

    if model_name is None:
        model_name = _default_model_name()
    if model_name and "Qwen3" in model_name:
        system_content = "\\no_think\n" + system_content

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt_text},
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
    def __init__(self, tokenizer, env, build_messages_fn, total_samples: int, logger=None, interaction_counter=None):
        self.tokenizer = tokenizer
        self.env = env
        self.build_messages_fn = build_messages_fn
        self.total_samples = int(total_samples)
        self.logger = logger
        self.interaction_counter = interaction_counter
        self._epoch = 0
    # accelerate.DataLoader から呼ばれる
    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
    def __iter__(self):
        # 不安定な状態でのみプロンプトを生成し、訓練データとする
        # 安定状態ではconvo_env内部でmax_auto_skipループが回り、エピソード終了する
        generated = 0

        while generated < self.total_samples:
            # 現在の環境の関係性状態をチェック
            current_rel = self.env.relation_snapshot()
            if isinstance(current_rel, dict):
                metrics = current_rel.get("metrics", current_rel)
                unstable_count = metrics.get("unstable_triads", 0)
            else:
                unstable_count = 0

            # 不安定ならプロンプトを生成
            if unstable_count > 0:
                # 介入判定プロンプトを生成
                persona_list = list(getattr(self.env, "persona_pool", getattr(self.env, "personas", [])))
                messages = self.build_messages_fn(self.env.logs, persona_list, env=self.env)

                if getattr(self.env, "debug_prompts", getattr(self.env, "debug", False)):
                    print("\n[ppo_train] === Prompt feed ===")
                    print(f"episode={self.env.episode}, turn={self.env.t}, total_steps={self.env.total_steps}")
                    print(f"unstable_triads={unstable_count}")
                    # print("[ppo_train] conversation logs (env.logs):")
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
                generated += 1
                yield {"query": query}
            else:
                # 安定 → step()で人間発話を生成
                # convo_env内部でmax_auto_skipループが回る
                if getattr(self.env, "debug", False):
                    print(f"[FiniteOnlineDataset] 安定状態を検出、step()を実行")

                _, _, done, info = self.env.step('{"intervene_now": false}')

                if done:
                    # max_auto_skip到達でエピソード終了 → リセット
                    if getattr(self.env, "debug", False):
                        print(f"[FiniteOnlineDataset] エピソード終了、reset()を実行")

                    # エピソード終了時のログ出力（安定状態で生成された全発話をまとめて表示）
                    if self.logger is not None and self.interaction_counter is not None:
                        interaction_id = next(self.interaction_counter)
                        human_utterances = info.get("human_utterance_before_relation", [])
                        rel_after = info.get("rel", {})
                        log_payload = {
                            "interaction_id": interaction_id,
                            "episode": self.env.episode,
                            "turn": self.env.t,
                            "skipped": True,
                            "reason": "stable_state_max_auto_skip_reached",
                            "done": True,
                        }
                        if human_utterances:
                            log_payload["human_utterances"] = [
                                {"speaker": u.get("speaker"), "utterance": u.get("utterance")}
                                for u in human_utterances
                            ]
                        if rel_after:
                            log_payload["relation"] = {
                                "is_stable": rel_after.get("unstable_triads", 0) == 0,
                                "edges": rel_after.get("edges", {}),
                            }
                        self.logger.log_interaction(**log_payload)

                    self.env.reset()
    def __len__(self):
        return self.total_samples


class DummyRewardModel(nn.Module):
    """環境ベース報酬を使うためのダミー報酬モデル。

    TRL 0.23 の PPOTrainer は torch.nn.Module を期待し `.to()` を呼ぶため、
    forward が呼ばれない最小実装を用意する。
    """
    
    # transformersモデルが持つ標準的な属性
    base_model_prefix = "transformer"

    def __init__(self):
        super().__init__()
        # ダミーのバックボーンモデルを作成（get_rewardで使用される）
        # 実際には呼ばれないが、属性アクセスを満たすため
        class DummyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, *args, **kwargs):
                # get_rewardから呼ばれる可能性があるので、適切なダミー出力を返す
                batch_size = kwargs.get('input_ids', args[0] if args else torch.zeros(1, 1)).shape[0]
                seq_length = kwargs.get('input_ids', args[0] if args else torch.zeros(1, 1)).shape[1]
                hidden_size = 4096  # Qwen2.5の隠れ層サイズ
                
                # ダミーの出力を作成
                class DummyOutput:
                    def __init__(self):
                        self.hidden_states = (torch.zeros(batch_size, seq_length, hidden_size),)
                        self.last_hidden_state = torch.zeros(batch_size, seq_length, hidden_size)
                
                return DummyOutput()
        
        self.transformer = DummyBackbone()

    def forward(self, *args, **kwargs):  # pragma: no cover - 呼ばれない想定
        # get_rewardから呼ばれる可能性があるので、ダミー出力を返す
        return self.transformer.forward(*args, **kwargs)

# === カスタムPPOTrainer: エントロピーボーナスを追加 ===
# TRL PPOTrainer.train()メソッドをオーバーライドして、
# 損失計算時にエントロピーボーナスを追加します（理論的に正しい実装）

def _create_train_method_with_entropy_bonus(get_reward_fn=None):
    """
    TRL PPOTrainer.train()メソッドにエントロピーボーナスを追加したバージョンを生成
    
    実行時に元のtrain()メソッドを取得し、損失計算部分を修正してエントロピーボーナスを追加します。
    
    Args:
        get_reward_fn: カスタムget_reward関数（環境ベースの報酬用）。Noneの場合はデフォルトを使用
    """
    import inspect
    import re
    from trl import PPOTrainer
    import textwrap
    

    cfg = get_config()
    ppo_cfg = cfg.ppo

    # 元のtrain()メソッドのソースコードを取得
    original_code = inspect.getsource(PPOTrainer.train)
    
    # インデントを除去（def train(self): の部分だけ残す）
    original_code = textwrap.dedent(original_code)
    
    # 損失計算部分を修正
    # 元の行（237行目付近）:
    #     pg_loss = masked_mean(pg_loss_max, ~padding_mask[micro_batch_inds])
    #     loss = pg_loss + args.vf_coef * vf_loss
    
    # パターンマッチング（より緩い条件）
    pattern = r'(pg_loss = masked_mean\(pg_loss_max,)'
    
    # まず該当箇所があるか確認
    if not re.search(pattern, original_code):
        print("[ERROR] Could not find pg_loss calculation in train() method", flush=True)
        return None
    
    # loss計算の直後にエントロピーボーナスを追加
    loss_pattern = r'(loss = pg_loss \+ args\.vf_coef \* vf_loss)'
    
    replacement = r'''\1
                        # [ENTROPY_BONUS] エントロピーボーナスを損失から引く
                        if hasattr(self, 'custom_entropy_coef') and self.custom_entropy_coef > 0:
                            try:
                                # logitsのデバイスを基準として使用
                                target_device = logits.device
                                
                                # padding_maskを同じデバイスに同期転送
                                padding_mask_moved = padding_mask.to(target_device, non_blocking=False)
                                
                                # エントロピー計算
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                
                                # micro_batch_indsをテンソルに変換して同じデバイスに配置
                                if isinstance(micro_batch_inds, np.ndarray):
                                    inds = torch.from_numpy(micro_batch_inds).long().to(target_device, non_blocking=False)
                                elif isinstance(micro_batch_inds, torch.Tensor):
                                    inds = micro_batch_inds.to(target_device, non_blocking=False)
                                else:
                                    inds = torch.tensor(micro_batch_inds, dtype=torch.long, device=target_device)
                                
                                # マスク作成（全て同じデバイス上で実行）
                                mask = ~padding_mask_moved[inds]
                                
                                # masked_meanを直接計算（デバイス問題を回避）
                                mask_sum = mask.sum()
                                if mask_sum > 0:
                                    mean_entropy = (entropy * mask).sum() / mask_sum
                                else:
                                    mean_entropy = torch.tensor(0.0, device=target_device)
                                
                                # エントロピー係数のスケジューリング（update数に応じて減衰）
                                if update <= getattr(self, 'entropy_transition_updates', [20, 60])[0]:
                                    current_entropy_coef = getattr(self, 'entropy_coef_phase1', 0.03)
                                elif update <= getattr(self, 'entropy_transition_updates', [20, 60])[1]:
                                    current_entropy_coef = getattr(self, 'entropy_coef_phase2', 0.02)
                                else:
                                    current_entropy_coef = getattr(self, 'entropy_coef_phase3', 0.01)

                                entropy_bonus = current_entropy_coef * mean_entropy
                                
                                # エントロピーボーナスの上限クリップ（yaml設定から読み込み）
                                max_entropy_bonus = getattr(self, 'max_entropy_bonus', 0.5)
                                if entropy_bonus > max_entropy_bonus:
                                    entropy_bonus = torch.tensor(max_entropy_bonus, device=target_device)
                                    if gradient_accumulation_idx == 0 and minibatch_idx == 0 and ppo_epoch_idx == 0:
                                        accelerator.print(f"[ENTROPY_BONUS] クリップ適用: {current_entropy_coef * mean_entropy.item():.4f} → {max_entropy_bonus:.4f}")
                                
                                # lossをtarget_deviceに移動（もし異なるデバイスにあれば）
                                if loss.device != target_device:
                                    loss = loss.to(target_device)
                                
                                loss_before = loss.item()
                                loss = loss - entropy_bonus
                                
                                # メトリクスを記録（wandbとlogger）
                                if gradient_accumulation_idx == 0 and minibatch_idx == 0 and ppo_epoch_idx == 0:
                                    mean_entropy_val = mean_entropy.item()
                                    entropy_bonus_val = entropy_bonus.item()
                                    loss_after_val = loss.item()
                                    
                                    # コンソール出力
                                    accelerator.print(f"[ENTROPY_BONUS] update={update} mean_entropy={mean_entropy_val:.4f} bonus={entropy_bonus_val:.4f} (coef={current_entropy_coef:.3f}) loss: {loss_before:.4f} -> {loss_after_val:.4f}")
                                    
                                    # Trainerのログ経路に記録（metrics.jsonl + wandb自動連携）
                                    try:
                                        self.log({
                                            "train/entropy/mean": mean_entropy_val,
                                            "train/entropy/bonus": entropy_bonus_val,
                                            "train/entropy/coef": current_entropy_coef,
                                            "train/loss/before_entropy": loss_before,
                                            "train/loss/after_entropy": loss_after_val,
                                        })
                                    except Exception as log_err:
                                        accelerator.print(f"[ENTROPY_BONUS] self.log failed: {log_err}")
                                    
                                    # wandbに直接記録（フォールバック）
                                    try:
                                        import wandb
                                        if wandb.run is not None:
                                            wandb.log({
                                                "entropy/mean_entropy": mean_entropy_val,
                                                "entropy/bonus": entropy_bonus_val,
                                                "entropy/coef": current_entropy_coef,
                                                "entropy/loss_before": loss_before,
                                                "entropy/loss_after": loss_after_val,
                                                "entropy/loss_reduction": loss_before - loss_after_val,
                                                "ppo/update": update,
                                            })  # stepを指定しない → Trainerの現在stepに自動追従
                                    except Exception as wandb_err:
                                        accelerator.print(f"[ENTROPY_BONUS] wandb log failed: {wandb_err}")
                                    
                                    # loggerに記録
                                    if hasattr(self, 'logger') and self.logger is not None:
                                        try:
                                            self.logger.log_event(
                                                event="entropy_bonus",
                                                update=update,
                                                mean_entropy=mean_entropy_val,
                                                entropy_bonus=entropy_bonus_val,
                                                loss_before=loss_before,
                                                loss_after=loss_after_val,
                                                loss_reduction=loss_before - loss_after_val,
                                            )
                                        except Exception as logger_err:
                                            accelerator.print(f"[ENTROPY_BONUS] logger failed: {logger_err}")
                            except Exception as e:
                                if gradient_accumulation_idx == 0 and minibatch_idx == 0 and ppo_epoch_idx == 0:
                                    import traceback
                                    accelerator.print(f"[ENTROPY_BONUS ERROR] {type(e).__name__}: {e}")
                                    tb = traceback.format_exc()
                                    accelerator.print("[ENTROPY_BONUS ERROR] Traceback:")
                                    accelerator.print(tb)
                                # エラー時はボーナスなしで続行
                                pass'''
    
    modified_code = re.sub(loss_pattern, replacement, original_code)
    
    if modified_code == original_code:
        print("[WARNING] Failed to patch train() method - pattern not found", flush=True)
        return None
    else:
        print("[SUCCESS] train() method patched with entropy bonus", flush=True)
    
    # 修正したコードをコンパイル
    namespace = {}
    try:
        # 必要なインポートを追加
        import torch
        import numpy as np
        import time
        import math
        import gc
        from transformers import GenerationConfig
        from trl.trainer.ppo_trainer import (
            batch_generation,
            forward,
            get_reward,
            masked_mean,
            masked_whiten,
            selective_log_softmax,
            unwrap_model_for_generation,
            INVALID_LOGPROB,
        )
        from trl.trainer.utils import (
            truncate_response,
            first_true_indices,
        )
        from contextlib import nullcontext
        from torch.nn.utils import clip_grad_norm_
        from accelerate.utils import release_memory
        
        # empty_cache は release_memory のエイリアス
        def empty_cache():
            release_memory()
        
        # カスタムget_reward関数が渡された場合は使用
        if get_reward_fn is not None:
            get_reward_to_use = get_reward_fn
        else:
            get_reward_to_use = get_reward
        
        # namespaceに追加
        namespace.update({
            'torch': torch,
            'np': np,
            'time': time,
            'math': math,
            'gc': gc,
            'GenerationConfig': GenerationConfig,
            'batch_generation': batch_generation,
            'forward': forward,
            'get_reward': get_reward_to_use,  # パッチされた関数を使用
            'masked_mean': masked_mean,
            'masked_whiten': masked_whiten,
            'selective_log_softmax': selective_log_softmax,
            'unwrap_model_for_generation': unwrap_model_for_generation,
            'INVALID_LOGPROB': INVALID_LOGPROB,
            'truncate_response': truncate_response,
            'first_true_indices': first_true_indices,
            'nullcontext': nullcontext,
            'clip_grad_norm_': clip_grad_norm_,
            'release_memory': release_memory,
            'empty_cache': empty_cache,
        })
        
        exec(modified_code, namespace)
        return namespace['train']
    except SyntaxError as e:
        print(f"[ERROR] Syntax error in modified train(): {e}", flush=True)
        # デバッグ用に問題のある行を表示
        lines = modified_code.split('\n')
        if hasattr(e, 'lineno') and e.lineno:
            start = max(0, e.lineno - 3)
            end = min(len(lines), e.lineno + 2)
            print(f"[DEBUG] Lines {start}-{end}:")
            for i in range(start, end):
                marker = '>>>' if i == e.lineno - 1 else '   '
                print(f"{marker} {i+1:3d}: {lines[i]}")
        return None

class PPOTrainerWithEntropyBonus(PPOTrainer):
    """
    エントロピーボーナスを損失に追加するカスタムPPOTrainer
    
    標準的なPPO実装:
        loss = policy_loss + vf_coef * value_loss
    
    この実装:
        loss = policy_loss + vf_coef * value_loss - entropy_coef * entropy
        
    これにより、方策が探索的になることを奨励します。
    """
    
    def __init__(self, *args, entropy_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_entropy_coef = float(entropy_coef)
        if self.custom_entropy_coef > 0:
            print(f"[PPOTrainerWithEntropyBonus] custom_entropy_coef={self.custom_entropy_coef} ENABLED", flush=True)
            print(f"[PPOTrainerWithEntropyBonus] Will subtract entropy_coef * entropy from loss", flush=True)
        else:
            print(f"[PPOTrainerWithEntropyBonus] custom_entropy_coef={self.custom_entropy_coef} DISABLED", flush=True)

# train()メソッドは main() 内でパッチ（get_rewardのパッチ後に適用する必要があるため）
# try:
#     PPOTrainerWithEntropyBonus.train = _create_train_method_with_entropy_bonus()
# except Exception as e:
#     print(f"[ERROR] Failed to create custom train() method: {e}", flush=True)
#     print("[FALLBACK] Using parent PPOTrainer.train() without entropy bonus", flush=True)

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

    # バッチサイズの計算:
    # 実効バッチサイズ = per_device_train_batch_size × grad_accum_steps × 2
    _per_device_bs = _to_int(getattr(ppo_cfg, "per_device_train_batch_size", 1), 1)
    _ga    = _to_int(getattr(ppo_cfg, "grad_accum_steps", 1), 1)
    _epochs= _to_int(getattr(ppo_cfg, "ppo_epochs", 1), 1)  # ←回数を外部（データ件数）で制御するので 1 がラク
    _lr    = _to_float(getattr(ppo_cfg, "lr", 1e-5), 1e-5)
    
    # [DEBUG] 設定値をログ出力（バッチサイズ問題の診断用）
    print(f"[PPO_CONFIG] per_device_train_batch_size (YAML) = {_per_device_bs}")
    print(f"[PPO_CONFIG] grad_accum_steps (YAML)    = {_ga}")
    print(f"[PPO_CONFIG] ppo_epochs (YAML)          = {_epochs}")
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
    _max_grad_norm = max(0.0, _to_float(getattr(ppo_cfg, "max_grad_norm", 1.0), 1.0))
    _whiten_rewards = bool(getattr(ppo_cfg, "whiten_rewards", False))
    _vf_coef = max(0.0, _to_float(getattr(ppo_cfg, "vf_coef", 0.1), 0.1))
    _gamma = max(0.0, min(1.0, _to_float(getattr(ppo_cfg, "gamma", 1.0), 1.0)))
    _lam = max(0.0, min(1.0, _to_float(getattr(ppo_cfg, "lam", 0.95), 0.95)))
    _kl_estimator = str(getattr(ppo_cfg, "kl_estimator", "k1")).lower()
    if _kl_estimator not in ["k1", "k3"]:
        _kl_estimator = "k1"
    _num_mini_batches = _to_int(getattr(ppo_cfg, "num_mini_batches", 1), 1)
    _entropy_coef = max(0.0, _to_float(getattr(ppo_cfg, "entropy_coef", 0.01), 0.01))

    # === PPOConfig（TRL 0.23 互換の安全パラメータ）====
    # TRLのバージョンによってはent_coefが存在しない場合があるため、条件付きで設定
    ppo_config_kwargs = {
        "learning_rate": _lr,
        "gradient_accumulation_steps": max(1, _ga),
        "num_ppo_epochs": _epochs,
        "kl_coef": _kl,
        "kl_estimator": _kl_estimator,            # KL推定器（"k1" or "k3"）
        "num_mini_batches": max(1, _num_mini_batches),  # ミニバッチ数（更新の粒度）
        "cliprange": _cliprange,
        "vf_coef": _vf_coef,                      # 価値関数損失の係数
        "cliprange_value": _cliprange_value,
        "gamma": _gamma,                          # 割引率
        "lam": _lam,                              # GAE lambda
        "max_grad_norm": _max_grad_norm,          # 勾配クリッピング（安定性向上）
        "whiten_rewards": _whiten_rewards,        # 報酬の正規化（平均0、標準偏差1に変換）
        "ds3_gather_for_generation": True,        # ZeRO-3 を使う場合の最適化（使ってなければ無害）
        "lr_scheduler_type": "constant",          # 学習率を一定に保つ（デフォルトの"linear"は0に減衰する）
        "warmup_steps": 0,                        # ウォームアップなし（すぐに full lr で学習開始）
    }
    
    # TRLのバージョンによってent_coefが存在しない場合があるため、
    # PPOTrainerWithEntropyBonusのカスタム実装を使用
    # （ent_coefがサポートされている場合でも、カスタム実装を使うことで一貫性を保つ）
    
    ppo_config = PPOConfig(**ppo_config_kwargs)
    ppo_config.target_kl = _target_kl
    ppo_config.kl_adjust_up = _kl_adjust_up
    ppo_config.kl_adjust_down = _kl_adjust_down
    # 生成の終端
    ppo_config.stop_token_id = tokenizer.eos_token_id
    # 1 デバイス当たりのバッチサイズ（Accelerate の world size 計算と整合させる）
    ppo_config.per_device_train_batch_size = max(
        1,
        _to_int(getattr(ppo_cfg, "per_device_train_batch_size", _per_device_bs), 1),
    )
    ppo_config.per_device_eval_batch_size = max(
        1,
        _to_int(getattr(ppo_cfg, "per_device_eval_batch_size", ppo_config.per_device_train_batch_size), 1),
    )


    # === 3) PPOTrainer 準備（環境ベース報酬を差し込む） ===
    # 会話環境
    # 後方互換性: max_historyを渡すが、env内部で新しいパラメータを優先する
    env = ConversationEnv(
        max_steps=cfg.env.max_steps,
        personas=None,  # 環境内部でcfg.env.personasから読み込む
        include_robot=cfg.env.include_robot,
        max_history=getattr(cfg.env, "max_history", 6),  # フォールバック値
        backend=cfg.scorer.backend,
        decay_factor=cfg.scorer.decay_factor,
        debug=cfg.env.debug,
        reward_backend=getattr(cfg.env, "reward_backend", "rule"),
        evaluation_horizon=cfg.env.evaluation_horizon,
        time_penalty=cfg.env.time_penalty,
        terminal_bonus=cfg.env.terminal_bonus,
        intervention_cost=getattr(cfg.env, "intervention_cost", 0.02),
    )

    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_log_dir = Path("./logs") / f"ppo_run-{run_id}"

    # wandb設定を読み込み
    enable_wandb = getattr(cfg.wandb, "enabled", True) if hasattr(cfg, "wandb") else True
    training_logger = TrainingRunLogger(run_log_dir, enable_wandb=enable_wandb)

    # wandb設定をTrainingRunLoggerに反映
    if hasattr(cfg, "wandb") and cfg.wandb:
        if hasattr(cfg.wandb, "table_log_frequency") and cfg.wandb.table_log_frequency:
            training_logger._table_log_frequency = int(cfg.wandb.table_log_frequency)
    training_logger.write_json("config.json", dataclasses.asdict(cfg))
    training_logger.log_event(event="run_start", run_id=run_id)

    # エピソード開始時に話題を記録
    training_logger.log_event(
        event="episode_start",
        episode=env.episode,
        topic=getattr(env, "current_topic", None),
    )
    print(f"[LOG] training artifacts => {run_log_dir}")
    interaction_counter = itertools.count(start=1)

    # （当面の起動確認用）ダミーの学習データセット
    _dummy_ds = Dataset.from_dict({"prompt": ["dummy"]})

    # 合計ステップ数＝ total_updates × effective_batch_size に合わせた有限データセットを作る
    # effective_batch_size = per_device_train_batch_size × grad_accum_steps × 2
    total_updates = _to_int(getattr(ppo_cfg, "total_updates", 50), 50)
    effective_batch_size = _per_device_bs * _ga * 2
    total_samples = total_updates * max(1, effective_batch_size)
    train_ds = FiniteOnlineDataset(
        tokenizer, env, build_robot_messages,
        total_samples=total_samples,
        logger=training_logger,
        interaction_counter=interaction_counter
    )
    eval_samples = max(1, effective_batch_size)
    eval_ds = FiniteOnlineDataset(
        tokenizer, env, build_robot_messages,
        total_samples=eval_samples,
        logger=training_logger,
        interaction_counter=interaction_counter
    )

    # TRL 側で要求されるバッチ回数を total_updates に合わせる
    ppo_config.total_episodes = total_samples
    ppo_config.num_train_epochs = 1
    # max_stepsを明示的に設定（学習率スケジューラーに必要）
    # 計算: total_updates * num_ppo_epochs = トレーニングステップ数
    ppo_config.max_steps = total_updates * _epochs
    print(f"[PPO] max_steps set to {ppo_config.max_steps} (total_updates={total_updates} * ppo_epochs={_epochs})")

    # 生成ハイパパラメータ（参考：必要なら trainer 側に渡す）
    gen_kwargs = dict(
        do_sample=True,
        temperature=_to_float(getattr(ppo_cfg, "temperature", 0.7), 0.7),
        top_p=_to_float(getattr(ppo_cfg, "top_p", 0.9), 0.9),
        max_new_tokens=_to_int(getattr(ppo_cfg, "max_new_tokens", 64), 64),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(f"[PPO] Generation kwargs: temperature={gen_kwargs['temperature']}, top_p={gen_kwargs['top_p']}, max_new_tokens={gen_kwargs['max_new_tokens']}")
    
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

    # 実効バッチサイズ: TRLが実行時に per_device_train_batch_size × grad_accum_steps × 2 で計算する
    
    # [CRITICAL] TRLは実行時にこの値を計算するため、Trainer作成後に再取得が必要
    # 2 * per_device_train_batch_size * grad_accum_steps

    # エピソードごとの統計を追跡
    episode_stats: Dict[int, Dict[str, Any]] = {}

    # バッチサマリーコレクターは後でTrainer作成後に初期化（実効バッチサイズ使用）
    batch_collector = None  # 一時的にNone、Trainer作成後に初期化

    # 報酬関数：生成テキストを環境に流し込み、Before→After のスコア差を返す
    # PPOTrainer 0.23 では `reward_model` 経由の計算が前提のため、後段で monkeypatch して利用する
    def reward_fn(samples, **kwargs):
        nonlocal batch_collector
        rewards: List[float] = []

        def _extract_plan_candidate(text: str) -> str:
            # JSONっぽい部分を抜き出す
            candidate = text.strip()
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = candidate[start : end + 1]
                try:
                    json.loads(snippet)
                    return snippet
                except json.JSONDecodeError:
                    pass
            return candidate

        def _compute_choice_entropy(counts_dict: dict) -> float:
            """カテゴリ分布からエントロピーを計算（bits）"""
            if not counts_dict:
                return 0.0
            total = sum(counts_dict.values())
            if total == 0:
                return 0.0
            probs = [count / total for count in counts_dict.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            return entropy

        for resp in samples:
            # 再推論ロジック: 不正なJSONの場合は最大3回再試行
            retry_count = 0
            max_retries = 3
            plan_text = None
            plan_obj = None
            final_raw_resp = ""

            for attempt in range(max_retries + 1):
                if attempt == 0:
                    # 初回: 元のレスポンスを使用
                    current_raw_resp = resp or ""
                else:
                    # 再推論: 同じプロンプトで新しいサンプルを生成
                    try:
                        retry_count += 1
                        # プロンプト再生成
                        messages = build_robot_messages(env.logs, env.personas, env=env)
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        encoded = tokenizer(prompt, return_tensors="pt").to(policy.device)

                        # 再生成時はtemperatureを大幅に下げてJSON形式を厳格化
                        retry_gen_kwargs = gen_kwargs.copy()
                        retry_gen_kwargs["temperature"] = 0.3  # 元の temperature (2.0など) から大幅削減
                        retry_gen_kwargs["top_p"] = 0.8  # top_pも絞る
                        retry_gen_kwargs["do_sample"] = True  # サンプリングは維持（完全greedyだと同じ出力）
                        
                        # 再生成（厳格モード）
                        with torch.no_grad():
                            generated = policy.generate(
                                **encoded,
                                **retry_gen_kwargs,
                            )
                        gen_ids = generated[0, encoded["input_ids"].shape[-1]:]
                        current_raw_resp = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    except Exception as e:
                        # 再推論に失敗した場合は元のレスポンスを使用
                        current_raw_resp = resp or ""
                        if getattr(env, "debug", False):
                            print(f"[reward_fn] 再推論失敗 (attempt {attempt}): {e}")

                final_raw_resp = current_raw_resp
                visible_resp = _strip_think_tags(current_raw_resp)
                candidate_plan_text = _extract_plan_candidate(visible_resp)

                # JSON妥当性チェック
                validated_plan_obj, is_valid = _validate_plan_json(candidate_plan_text)

                if is_valid and validated_plan_obj is not None:
                    # 有効なJSONが得られた
                    plan_text = candidate_plan_text
                    plan_obj = validated_plan_obj
                    break
                else:
                    # 検証失敗の理由をログ出力（デバッグ時のみ）
                    if getattr(env, "debug", False) and attempt > 0:
                        print(f"[reward_fn] JSON検証失敗 (attempt {attempt}): {candidate_plan_text[:100]}...")
            else:
                # すべての試行が失敗した場合のフォールバック
                plan_text = '{"intervene_now": false}'
                plan_obj = {"intervene_now": False}
                json_generation_failed = True  # JSON生成失敗フラグ
                if getattr(env, "debug", False):
                    print(f"[reward_fn] JSON検証失敗、フォールバック使用 (retry_count={retry_count})")
            
            # JSON生成が成功した場合のフラグ
            if 'json_generation_failed' not in locals():
                json_generation_failed = False

            history_before = [dict(entry) for entry in env.logs]
            state_before = None
            try:
                state_before = env.planning_context()
            except Exception:
                state_before = None

            # 安定状態の場合は介入判定をスキップ（フォールバック）
            # 注: 通常は__iter__()でスキップされるため、ここに到達するのは不安定なターンのみ
            is_stable_now = False
            try:
                current_rel = env.relation_snapshot()
                if isinstance(current_rel, dict):
                    metrics = current_rel.get("metrics", current_rel)
                    is_stable_now = metrics.get("unstable_triads", 0) == 0
                else:
                    is_stable_now = False
                if is_stable_now:
                    plan_text = '{"intervene_now": false}'
                    plan_obj = {"intervene_now": False}
            except Exception:
                pass  # 安定チェックに失敗したら通常通り進める

            # 選択肢をバッチコレクターに記録
            if plan_obj and batch_collector is not None:
                batch_collector.add_decision(plan_obj)

            interaction_id = next(interaction_counter)

            next_obs, reward_value, done, info = env.step(plan_text)
            reward_details = info.get("reward_breakdown", {}) or {}
            components = _extract_reward_components(reward_details)
            total_reward = float(reward_value)
            if components:
                total_reward = sum(components.values())

            # 報酬詳細を整形（reward_detailsとcomponentsの両方から情報を取得）
            reward_breakdown_formatted = {}
            if reward_details:
                if "delta_u_flip" in reward_details:
                    reward_breakdown_formatted["Δu_flip"] = reward_details["delta_u_flip"]
                if "intervention_cost" in reward_details:
                    reward_breakdown_formatted["介入コスト"] = reward_details["intervention_cost"]
                if "time_penalty" in reward_details:
                    reward_breakdown_formatted["時間ペナルティ"] = reward_details["time_penalty"]
                if "terminal_bonus" in reward_details:
                    reward_breakdown_formatted["終了ボーナス"] = reward_details["terminal_bonus"]

            # componentsからも情報を補完（特に安定時のtime_penalty用）
            if components and not reward_breakdown_formatted:
                for key, value in components.items():
                    if "time_penalty" in key.lower() or "time" in key.lower():
                        reward_breakdown_formatted["時間ペナルティ"] = value
                    elif "intervention" in key.lower():
                        reward_breakdown_formatted["介入コスト"] = value
                    elif "delta" in key.lower() or "u_flip" in key.lower():
                        reward_breakdown_formatted["Δu_flip"] = value
                    elif "bonus" in key.lower() or "terminal" in key.lower():
                        reward_breakdown_formatted["終了ボーナス"] = value

            # エピソード統計の追跡
            current_episode_id = env.episode
            if current_episode_id not in episode_stats:
                episode_stats[current_episode_id] = {
                    "total_reward": 0.0,
                    "intervention_count": 0,
                    "unstable_no_intervention_count": 0,
                    "plan_error_count": 0,
                }
            episode_stats[current_episode_id]["total_reward"] += total_reward
            if info.get("intervened"):
                episode_stats[current_episode_id]["intervention_count"] += 1

            # 不安定で介入しなかった回数をカウント
            status = info.get("status", {})
            is_stable_before = status.get("is_stable", False)
            if not is_stable_before and not info.get("intervened"):
                episode_stats[current_episode_id]["unstable_no_intervention_count"] += 1

            # plan_errorがあればカウント
            plan_error = info.get("plan_error")
            if plan_error:
                episode_stats[current_episode_id]["plan_error_count"] += 1

            # バッチコレクターにステップを記録
            if batch_collector is not None:
                batch_collector.add_step(
                    reward=total_reward,
                    intervened=info.get("intervened", False),
                    is_stable_before=is_stable_before,
                    plan_error=bool(plan_error)
                )

            # 会話の流れを整理: 介入前 → 介入 → evaluation_horizon後 → bonus確認後
            log_payload: Dict[str, Any] = {
                "interaction_id": interaction_id,
                "episode": env.episode,
                "turn": env.t,
                "done": bool(done),
                "retry_count": retry_count,  # 再推論回数を記録
            }

            # interaction_id=1（最初のステップ）の場合のみepisode_info（話題・選択された地雷・地雷を持つペルソナ）を追加
            if interaction_id == 1:
                episode_info = {}
                if hasattr(env, "current_topic") and env.current_topic:
                    episode_info["topic"] = env.current_topic

                # デバッグ: 地雷情報を確認
                has_trigger = hasattr(env, "current_topic_trigger")
                trigger_value = getattr(env, "current_topic_trigger", None)
                has_persona_triggers = hasattr(env, "persona_triggers")
                persona_triggers_value = getattr(env, "persona_triggers", None)

                if getattr(env, "debug", False):
                    print(f"[ppo_train] DEBUG - Step 1 episode_info:")
                    print(f"  has current_topic_trigger: {has_trigger}")
                    print(f"  current_topic_trigger value: {trigger_value}")
                    print(f"  has persona_triggers: {has_persona_triggers}")
                    print(f"  persona_triggers: {persona_triggers_value}")

                if has_trigger and trigger_value:
                    selected_trigger = trigger_value
                    episode_info["selected_trigger"] = selected_trigger

                    # 選択された地雷を持っているペルソナをリストアップ
                    if has_persona_triggers and persona_triggers_value:
                        personas_with_trigger = [
                            persona for persona, triggers in persona_triggers_value.items()
                            if selected_trigger in triggers
                        ]
                        if personas_with_trigger:
                            episode_info["personas_with_trigger"] = personas_with_trigger

                if episode_info:
                    log_payload["episode_info"] = episode_info

            # 1. interaction_id=1の場合は初期発話を表示、それ以外は何も表示しない
            if interaction_id == 1:
                # 初期発話（このステップ開始時まで）
                # reset()で記録された初期会話ログを使用
                if hasattr(env, 'initial_conversation_log') and env.initial_conversation_log:
                    log_payload["initial_conversation"] = env.initial_conversation_log
                else:
                    # フォールバック: logsから抽出
                    initial_utterances = []
                    for log_entry in history_before:
                        if log_entry.get("speaker") != "ロボット":
                            initial_utterances.append({
                                "speaker": log_entry.get("speaker"),
                                "utterance": log_entry.get("utterance"),
                            })
                    if initial_utterances:
                        log_payload["initial_conversation"] = initial_utterances

            # 2. 前のステップで介入していない場合に生成された人間発話（interaction_id=1は除外）
            if interaction_id > 1:
                human_utterances_before_rel = info.get("human_utterance_before_relation", [])

                if human_utterances_before_rel:
                    log_payload["human_utterances"] = [
                        {"speaker": r.get("speaker"), "utterance": r.get("utterance")}
                        for r in human_utterances_before_rel if r.get("speaker") != "ロボット"
                    ]

            # 3. 関係性（ステップ開始時の関係性）
            status = info.get("status", {})
            is_stable_before = status.get("is_stable", False)
            edges_before = status.get("edges", {})
            log_payload["relation"] = {
                "is_stable": is_stable_before,
                "edges": edges_before,
            }
            # edgesが空の場合はデバッグ情報を追加
            if not edges_before:
                log_payload["relation"]["debug_note"] = "edges_empty_at_step_start"

            # 4. 介入判定と詳細（不安定時のみ出力）
            if not is_stable_before:
                # 不安定時は、介入の有無に関わらず介入判定結果を出力
                plan = info.get("plan")

                if info.get("intervened"):
                    # 介入した場合
                    intervention_info = {
                        "intervened": True,
                        "robot_utterance": info.get("robot_utterance"),
                    }
                    if plan and isinstance(plan, dict):
                        if "strategy" in plan:
                            intervention_info["strategy"] = plan["strategy"]
                        if "edge_to_change" in plan:
                            intervention_info["edge_to_change"] = plan["edge_to_change"]
                        if "target_speaker" in plan:
                            intervention_info["target_speaker"] = plan["target_speaker"]

                    # evaluation_horizon + terminal_bonus_duration中の人間発話を抽出
                    replies = info.get("replies", [])
                    if replies:
                        intervention_info["utterances_after_intervention"] = [
                            {"speaker": r.get("speaker"), "utterance": r.get("utterance")}
                            for r in replies if r.get("speaker") != "ロボット"
                        ]

                    log_payload["intervention"] = intervention_info

                    # 5. evaluation_horizon後の関係性
                    if info.get("rel_after_horizon"):
                        horizon_rel = info["rel_after_horizon"]
                        horizon_unstable = horizon_rel.get("unstable_triads", 0)
                        horizon_stable = horizon_unstable == 0
                        log_payload["relations_after_horizon"] = {
                            "is_stable": horizon_stable,
                            "edges": horizon_rel.get("edges", {}),
                        }

                        if getattr(env, "debug", False):
                            print(f"[ppo_train] DEBUG - relations_after_horizon:")
                            print(f"  is_stable: {horizon_stable}")
                            print(f"  edges: {horizon_rel.get('edges', {})}")

                        # 6. terminal_bonus_duration確認（該当時のみ）
                        if horizon_stable and reward_details.get("terminal_bonus"):
                            log_payload["bonus_check"] = {
                                "terminal_bonus_granted": True,
                                "duration": env.terminal_bonus_duration,
                            }
                            # terminal_bonus_duration後の関係性も出力
                            if info.get("rel_after_bonus"):
                                bonus_rel = info["rel_after_bonus"]
                                bonus_unstable = bonus_rel.get("unstable_triads", 0)
                                bonus_stable = bonus_unstable == 0
                                log_payload["relations_after_bonus"] = {
                                    "is_stable": bonus_stable,
                                    "edges": bonus_rel.get("edges", {}),
                                }

                                if getattr(env, "debug", False):
                                    print(f"[ppo_train] DEBUG - relations_after_bonus:")
                                    print(f"  is_stable: {bonus_stable}")
                                    print(f"  edges: {bonus_rel.get('edges', {})}")
                else:
                    # 介入しなかった場合（不安定だが介入判定でFalse）
                    intervention_info = {
                        "intervened": False,
                    }
                    # planが不正な場合でもintervene_nowを追加（デバッグ用）
                    if plan and isinstance(plan, dict):
                        intervention_info["intervene_now"] = plan.get("intervene_now", False)
                    else:
                        # planがNoneまたは不正な場合はFalseとして扱う
                        intervention_info["intervene_now"] = False
                        if plan is None:
                            intervention_info["plan_error"] = "plan_is_none"
                        elif not isinstance(plan, dict):
                            intervention_info["plan_error"] = "plan_not_dict"

                    # 前ステップ非介入時に生成された人間発話を表示
                    replies = info.get("replies", [])
                    if replies:
                        human_utterances = [
                            {"speaker": r.get("speaker"), "utterance": r.get("utterance")}
                            for r in replies if r.get("speaker") != "ロボット"
                        ]
                        if human_utterances:
                            intervention_info["human_utterances"] = human_utterances

                    log_payload["intervention"] = intervention_info

            # 5. 報酬
            log_payload["reward"] = {
                "total": total_reward,
                "breakdown": reward_breakdown_formatted,
            }

            # エピソード終了時は統計情報を追加
            if done:
                stats = episode_stats.get(current_episode_id, {
                    "total_reward": 0.0,
                    "intervention_count": 0,
                    "unstable_no_intervention_count": 0,
                    "plan_error_count": 0,
                })
                log_payload["episode_summary"] = {
                    "total_reward": stats["total_reward"],
                    "total_interventions": stats["intervention_count"],
                    "unstable_no_intervention": stats["unstable_no_intervention_count"],
                    "plan_errors": stats["plan_error_count"],
                }

            # JSON生成に失敗したサンプル（retry_count=3 かつ JSON無効）にペナルティを付与
            json_failure_penalty = 0.0
            if retry_count >= 3 and json_generation_failed:
                json_failure_penalty = getattr(cfg.env, "json_failure_penalty", -1.0)
                total_reward += json_failure_penalty
                # breakdownに追加
                reward_breakdown_formatted["JSON生成失敗"] = json_failure_penalty
                if getattr(env, "debug", False):
                    print(f"[reward_fn] JSON生成失敗ペナルティ: {json_failure_penalty} (retry_count={retry_count})")
            
            rewards.append(total_reward)

            # --- バッチ判定: intervention-decision ベース ---
            if batch_collector is not None and batch_collector.is_batch_ready():
                batch_summary = batch_collector.get_summary()
                training_logger.log_batch_summary(
                    batch_summary=batch_summary,
                    episode=env.episode,
                    step=interaction_id,
                )

            training_logger.log_interaction(**log_payload)

            if done:
                current_episode = env.episode
                try:
                    env_metrics = env.relation_snapshot()
                except Exception:
                    env_metrics = {}

                # エピソード統計を取得
                stats = episode_stats.get(current_episode, {
                    "total_reward": 0.0,
                    "intervention_count": 0,
                    "unstable_no_intervention_count": 0,
                    "plan_error_count": 0,
                })

                training_logger.log_event(
                    event="episode_end",
                    episode=current_episode,
                    reward=total_reward,
                    env_metrics=env_metrics,
                    reward_breakdown=reward_details,
                    personas=list(getattr(env, "persona_pool", getattr(env, "personas", []))),
                    total_steps=env.total_steps,
                    episode_total_reward=stats["total_reward"],
                    episode_intervention_count=stats["intervention_count"],
                    episode_unstable_no_intervention_count=stats["unstable_no_intervention_count"],
                    episode_plan_error_count=stats["plan_error_count"],
                )

                # 統計をクリア
                if current_episode in episode_stats:
                    del episode_stats[current_episode]

                env.reset()
                training_logger.log_event(
                    event="episode_start",
                    episode=env.episode,
                    topic=getattr(env, "current_topic", None),
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
    
    # エントロピーボーナス付きtrain()メソッドを設定（パッチ後のget_rewardを使用）
    try:
        PPOTrainerWithEntropyBonus.train = _create_train_method_with_entropy_bonus(get_reward_fn=_env_get_reward)
        print("[TRAIN_METHOD] Custom train() with entropy bonus applied", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to apply custom train() method: {e}", flush=True)

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

    trainer = PPOTrainerWithEntropyBonus(
        args=ppo_config,
        processing_class=tokenizer,            # = tokenizer
        model=policy,
        ref_model=ref_policy,
        reward_model=dummy_reward_model,
        train_dataset=train_ds,                # IterableDataset で "query" を供給
        eval_dataset=eval_ds,
        value_model=getattr(policy, "pretrained_model", policy),
        data_collator=ppo_collator,           # ← これが重要
        entropy_coef=_entropy_coef,           # カスタムパラメータ
    )

    # 実効バッチサイズ = per_device_train_batch_size × grad_accum_steps × 2
    effective_batch_size = ppo_config.per_device_train_batch_size * _ga * 2

    # BatchSummaryCollectorを実効バッチサイズで初期化
    from app.batch_summary import BatchSummaryCollector
    batch_collector = BatchSummaryCollector(batch_size=effective_batch_size)
    print(f"[TRAINER_INIT] BatchSummaryCollector initialized with batch_size={effective_batch_size}")
    print(f"[TRAINER_INIT]   per_device_train_batch_size={ppo_config.per_device_train_batch_size}")
    print(f"[TRAINER_INIT]   grad_accum_steps={_ga}")
    
    # デバッグ: Trainer初期化後の状態確認
    print(f"[TRAINER_DEBUG] Trainer initialized", flush=True)
    print(f"[TRAINER_DEBUG] custom_entropy_coef={trainer.custom_entropy_coef}", flush=True)
    print(f"[TRAINER_DEBUG] has state: {hasattr(trainer, 'state')}", flush=True)
    print(f"[TRAINER_DEBUG] training_logger available: {training_logger is not None}", flush=True)
    print(f"[TRAINER_DEBUG] training_logger type: {type(training_logger)}", flush=True)

    # 学習率スケジューラーの確認
    if hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
        current_lr = trainer.lr_scheduler.get_last_lr()[0] if hasattr(trainer.lr_scheduler, 'get_last_lr') else _lr
        print(f"[PPO] Optimizer learning rate: {current_lr}, scheduler type: {ppo_config.lr_scheduler_type}")
    else:
        print(f"[PPO] No lr_scheduler found, using base learning rate: {_lr}")

    class KLSpikeDetectionCallback(TrainerCallback):
        """KL爆発を検知してバッチ詳細をダンプ"""
        def __init__(
            self,
            logger: TrainingRunLogger | None = None,
            approxkl_threshold: float = 0.05,
            objective_kl_threshold: float = 5.0,
        ) -> None:
            self.logger = logger
            self.approxkl_threshold = approxkl_threshold
            self.objective_kl_threshold = objective_kl_threshold

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return control

            approxkl = logs.get("policy/approxkl_avg")
            objective_kl = logs.get("objective/kl")
            step = int(getattr(state, "global_step", 0) or 0)

            # スパイク検知
            spike_detected = False
            spike_type = None
            spike_value = None

            if approxkl is not None and approxkl > self.approxkl_threshold:
                spike_detected = True
                spike_type = "approxkl"
                spike_value = approxkl
            elif objective_kl is not None and objective_kl > self.objective_kl_threshold:
                spike_detected = True
                spike_type = "objective_kl"
                spike_value = objective_kl

            if spike_detected:
                print(f"[KL SPIKE] step={step} {spike_type}={spike_value:.4f} (threshold={self.approxkl_threshold if spike_type=='approxkl' else self.objective_kl_threshold})")

                # バッチ詳細をダンプ
                spike_details = {
                    "step": step,
                    "spike_type": spike_type,
                    "spike_value": spike_value,
                    "all_metrics": logs,
                }

                # ヒストグラム情報を含める
                if "policy/approxkl_avg" in logs:
                    spike_details["approxkl_avg"] = logs["policy/approxkl_avg"]
                if "policy/clipfrac_avg" in logs:
                    spike_details["clipfrac_avg"] = logs["policy/clipfrac_avg"]
                if "objective/rlhf_reward" in logs:
                    spike_details["rlhf_reward"] = logs["objective/rlhf_reward"]

                # ロガーに記録
                if self.logger is not None:
                    try:
                        self.logger.log_event(
                            event="kl_spike_detected",
                            **spike_details
                        )
                    except Exception as e:
                        print(f"[KL SPIKE] Failed to log: {e}")

            return control

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

    # KLスパイク検知コールバック（常に有効）
    spike_detection_callback = KLSpikeDetectionCallback(
        logger=training_logger,
        approxkl_threshold=0.05,
        objective_kl_threshold=5.0,
    )
    trainer.add_callback(spike_detection_callback)
    if training_logger is not None:
        training_logger.log_event(
            event="kl_spike_detection_enabled",
            approxkl_threshold=0.05,
            objective_kl_threshold=5.0,
        )

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

    # 決定的評価コールバック
    class DeterministicEvalCallback(TrainerCallback):
        """定期的に決定的評価を実行し、ベストチェックポイントを保存"""
        def __init__(
            self,
            model,
            tokenizer,
            build_messages_fn,
            logger: TrainingRunLogger | None = None,
            eval_frequency: int = 10,  # 何ステップごとに評価するか
            output_dir: str = "../models/ppo_robot",
        ) -> None:
            self.model = model
            self.tokenizer = tokenizer
            self.build_messages_fn = build_messages_fn
            self.logger = logger
            self.eval_frequency = eval_frequency
            self.output_dir = output_dir
            self.best_score = float('-inf')
            self.best_checkpoint = None
            self._last_eval_step = -1

        def on_step_end(self, args, state, control, **kwargs):
            step = int(getattr(state, "global_step", 0) or 0)

            # 評価頻度に達していない、または既に評価済みの場合はスキップ
            if step < self.eval_frequency or step == self._last_eval_step:
                return control

            if step % self.eval_frequency != 0:
                return control

            self._last_eval_step = step

            print(f"\n[EVAL] Running deterministic evaluation at step {step}...")

            try:
                from eval_deterministic import run_deterministic_eval, print_eval_results

                # 評価実行
                results = run_deterministic_eval(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    build_messages_fn=self.build_messages_fn,
                    temperature=0.0,
                    seed=42,
                    max_steps=8,
                )

                metrics = results["metrics"]

                # 結果を表示
                print_eval_results(results)

                # スコア計算（複合指標）
                # スコア = 安定到達率 * 2.0 - 有害介入率 * 1.0 + no-op差分 * 0.5
                score = (
                    metrics.stable_reach_rate * 2.0
                    - metrics.harmful_intervention_rate * 1.0
                    + metrics.noop_baseline_diff * 0.5
                )

                print(f"[EVAL] Composite score: {score:.4f} (best: {self.best_score:.4f})")

                # ログに記録
                if self.logger is not None:
                    try:
                        self.logger.log_event(
                            event="deterministic_eval",
                            step=step,
                            stable_reach_rate=metrics.stable_reach_rate,
                            avg_turns_to_stable=metrics.avg_turns_to_stable,
                            intervention_count=metrics.intervention_count,
                            non_intervention_count=metrics.non_intervention_count,
                            noop_baseline_diff=metrics.noop_baseline_diff,
                            harmful_intervention_rate=metrics.harmful_intervention_rate,
                            total_reward=metrics.total_reward,
                            avg_reward=metrics.avg_reward,
                            composite_score=score,
                        )
                    except Exception as e:
                        print(f"[EVAL] Failed to log: {e}")

                # ベストスコア更新
                if score > self.best_score:
                    self.best_score = score
                    checkpoint_name = f"best_checkpoint_step{step}_score{score:.4f}"
                    self.best_checkpoint = checkpoint_name

                    print(f"[EVAL] ✓ New best score! Saving checkpoint: {checkpoint_name}")

                    # チェックポイント保存
                    checkpoint_path = Path(self.output_dir) / checkpoint_name
                    checkpoint_path.mkdir(parents=True, exist_ok=True)

                    try:
                        # モデルを保存
                        unwrapped_model = getattr(self.model, "pretrained_model", self.model)
                        unwrapped_model.save_pretrained(checkpoint_path)
                        self.tokenizer.save_pretrained(checkpoint_path)

                        # メトリクスも保存
                        metrics_path = checkpoint_path / "eval_metrics.json"
                        with open(metrics_path, "w") as f:
                            json.dump({
                                "step": step,
                                "composite_score": score,
                                "stable_reach_rate": metrics.stable_reach_rate,
                                "avg_turns_to_stable": metrics.avg_turns_to_stable,
                                "intervention_count": metrics.intervention_count,
                                "non_intervention_count": metrics.non_intervention_count,
                                "noop_baseline_diff": metrics.noop_baseline_diff,
                                "harmful_intervention_rate": metrics.harmful_intervention_rate,
                                "total_reward": metrics.total_reward,
                                "avg_reward": metrics.avg_reward,
                            }, f, indent=2)

                        print(f"[EVAL] Checkpoint saved to {checkpoint_path}")

                        if self.logger is not None:
                            self.logger.log_event(
                                event="best_checkpoint_saved",
                                step=step,
                                checkpoint_name=checkpoint_name,
                                score=score,
                            )
                    except Exception as e:
                        print(f"[EVAL] Failed to save checkpoint: {e}")

            except Exception as e:
                print(f"[EVAL] Evaluation failed: {e}")
                import traceback
                traceback.print_exc()

            return control

    # 決定的評価を有効化（設定で制御）
    enable_deterministic_eval = getattr(ppo_cfg, "enable_deterministic_eval", True)
    eval_frequency = getattr(ppo_cfg, "deterministic_eval_frequency", 10)

    if enable_deterministic_eval:
        deterministic_eval_callback = DeterministicEvalCallback(
            model=policy,
            tokenizer=tokenizer,
            build_messages_fn=build_robot_messages,
            logger=training_logger,
            eval_frequency=eval_frequency,
            output_dir=getattr(ppo_cfg, "output_dir", "../models/ppo_robot"),
        )
        trainer.add_callback(deterministic_eval_callback)
        print(f"[EVAL] Deterministic evaluation enabled (frequency: every {eval_frequency} steps)")
        if training_logger is not None:
            training_logger.log_event(
                event="deterministic_eval_enabled",
                frequency=eval_frequency,
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
        print(f"[TRAIN_DEBUG] Starting trainer.train()...", flush=True)
        print(f"[TRAIN_DEBUG] Trainer class: {type(trainer).__name__}", flush=True)
        print(f"[TRAIN_DEBUG] Has compute_loss: {hasattr(trainer, 'compute_loss')}", flush=True)
        print(f"[TRAIN_DEBUG] compute_loss method: {trainer.compute_loss}", flush=True)
        
        trainer.train()
        
        print(f"[TRAIN_DEBUG] trainer.train() completed", flush=True)
        print(f"[TRAIN_DEBUG] compute_loss was called {trainer._compute_loss_call_count} times", flush=True)
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
