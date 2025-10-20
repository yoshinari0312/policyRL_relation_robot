from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import List, Optional, Dict, Any
import os

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None


# ============ サブ設定 ============
@dataclass
class OllamaGenOpts:
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    num_ctx: Optional[int] = None


@dataclass
class OllamaCfg:
    base_url: str = "http://ollama:11434"
    model: str = "llama3:8b"
    model_rl: Optional[str] = None
    bases: List[str] = field(default_factory=list)
    scorer_base: Optional[str] = None
    reward_base: Optional[str] = None
    gen_options: OllamaGenOpts = field(default_factory=OllamaGenOpts)


@dataclass
class OpenAICfg:
    api_key: Optional[str] = None
    model: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: Optional[str] = None
    embedding_deployment: Optional[str] = None
    embedding_api_version: Optional[str] = None


@dataclass
class ScorerCfg:
    backend: Optional[str] = None   # "rule" | "ollama" | "openai"
    use_ema: Optional[bool] = None
    ema_alpha: Optional[float] = None
    decay_factor: Optional[float] = None


@dataclass
class EnvCfg:
    max_steps: Optional[int] = None
    include_robot: Optional[bool] = None
    max_history: Optional[int] = None
    personas: Optional[List[str]] = None
    ref_device: Optional[Any] = None
    ref_device_map: Optional[Dict[Any, Any]] = None
    debug: Optional[bool] = None
    reward_backend: Optional[str] = None


@dataclass
class PPOCfg:
    # ★ 既定を入れておく（QLoRA用の4bitモデル例）
    model_name_or_path: Optional[str] = None
    # 4bit/8bitなどで必要なら指定（例: "4bit"）。無ければ None
    revision: Optional[str] = None
    cuda_visible_devices: Optional[str] = None
    # モデルのロード時に各GPUへ割り当てる上限GiB（float | list | dict）
    max_memory_per_device_gib: Optional[Any] = None
    # 個別GPUに厳密な割当をしたい場合のマップ（キー=論理GPU index/文字列）
    max_memory_map: Optional[Dict[Any, Any]] = None
    # HF transformers の device_map 戦略（"auto"/"balanced"/"balanced_low_0" 等）
    device_map_strategy: Optional[str] = None

    # 学習ハイパラ（必要に応じて使用）
    lora_rank: Optional[int] = None
    lr: Optional[float] = None
    kl_coef: Optional[float] = None
    target_kl: Optional[float] = None
    kl_adjust_up: Optional[float] = None
    kl_adjust_down: Optional[float] = None
    cliprange: Optional[float] = None
    cliprange_value: Optional[float] = None
    decode_repetition_penalty: Optional[float] = None
    decode_no_repeat_ngram_size: Optional[int] = None
    decode_typical_p: Optional[float] = None
    decode_min_p: Optional[float] = None
    similarity_retry_threshold: Optional[float] = None
    similarity_retry_max_attempts: Optional[int] = None
    repetition_semantic_penalty: Optional[float] = None
    repetition_semantic_threshold: Optional[float] = None

    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    batch_size: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    ppo_epochs: Optional[int] = None
    grad_accum_steps: Optional[int] = None
    local_rollout_forward_batch_size: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    total_updates: Optional[int] = None
    episode_len: Optional[int] = None
    missing_eos_penalty: Optional[float] = None
    non_japanese_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    repetition_lookback: Optional[int] = None
    entropy_floor: Optional[float] = None
    entropy_patience: Optional[int] = None
    entropy_monitor_warmup: Optional[int] = None
    topic_overlap_weight: Optional[float] = None
    topic_miss_penalty: Optional[float] = None
    topic_similarity_threshold: Optional[float] = None
    topic_overlap_horizon: Optional[int] = None
    brevity_penalty: Optional[float] = None
    brevity_min_chars: Optional[int] = None

    output_dir: Optional[str] = None
    prompt_feed_debug: Optional[bool] = None


# ============ ルート構成 ============
@dataclass
class AppConfig:
    env: EnvCfg = field(default_factory=EnvCfg)
    scorer: ScorerCfg = field(default_factory=ScorerCfg)
    ollama: OllamaCfg = field(default_factory=OllamaCfg)
    openai: OpenAICfg = field(default_factory=OpenAICfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)


def _filter_kwargs(cls, dct: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """未知キーを捨ててdataclassに渡す用"""
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in (dct or {}).items() if k in allowed}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)) and len(value) == 0:
        return True
    return False


def _validate_required_fields(cfg: AppConfig) -> None:
    missing_sections: List[str] = []

    def _require(section: str, obj: Any, field_names: List[str]) -> None:
        if obj is None:
            missing_sections.append(f"{section}: <section missing>")
            return
        missing = []
        for name in field_names:
            value = getattr(obj, name, None)
            if _is_missing(value):
                missing.append(name)
        if missing:
            missing_sections.append(f"{section}: {', '.join(missing)}")

    _require("env", cfg.env, ["max_steps", "include_robot", "max_history", "personas", "debug"])
    _require("scorer", cfg.scorer, ["backend", "use_ema", "ema_alpha", "decay_factor"])
    _require("ollama", cfg.ollama, ["model"])
    _require("ollama.gen_options", getattr(cfg.ollama, "gen_options", None), ["temperature", "top_p", "num_ctx"])
    _require("openai", cfg.openai, ["api_key", "model"])
    _require(
        "openai.azure",
        cfg.openai,
        ["azure_endpoint", "azure_api_key", "azure_api_version", "embedding_deployment", "embedding_api_version"],
    )
    _require(
        "ppo",
        cfg.ppo,
        [
            "model_name_or_path",
            "lora_rank",
            "lr",
            "kl_coef",
            "max_new_tokens",
            "temperature",
            "top_p",
            "batch_size",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "ppo_epochs",
            "grad_accum_steps",
            "local_rollout_forward_batch_size",
            "gradient_checkpointing",
            "total_updates",
            "episode_len",
            "missing_eos_penalty",
            "non_japanese_penalty",
            "repetition_penalty",
            "repetition_lookback",
            "output_dir",
        ],
    )

    if missing_sections:
        joined = "\n - " + "\n - ".join(missing_sections)
        raise ValueError("Missing required configuration values:" + joined)


def load_config(yaml_path: str = "config.local.yaml") -> AppConfig:
    """config.local.yaml を（あれば）読み、既定値にマージして AppConfig を返す"""
    cfg = AppConfig()  # 既定値

    if yaml and os.path.exists(yaml_path):
        # 前段で UTF-8 化済み想定
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    else:
        y = {}

    # セクション毎に安全マージ
    env = EnvCfg(**_filter_kwargs(EnvCfg, y.get("env")))
    scorer = ScorerCfg(**_filter_kwargs(ScorerCfg, y.get("scorer")))
    ollama_dict = y.get("ollama") or {}
    ollama_gen = OllamaGenOpts(**_filter_kwargs(OllamaGenOpts, (ollama_dict.get("gen_options") or {})))
    ollama = OllamaCfg(
        **_filter_kwargs(OllamaCfg, {**ollama_dict, "gen_options": ollama_gen})
    )
    openai = OpenAICfg(**_filter_kwargs(OpenAICfg, y.get("openai")))
    ppo = PPOCfg(**_filter_kwargs(PPOCfg, y.get("ppo")))

    app_cfg = AppConfig(env=env, scorer=scorer, ollama=ollama, openai=openai, ppo=ppo)
    _validate_required_fields(app_cfg)
    return app_cfg


def get_config() -> AppConfig:
    return load_config()
