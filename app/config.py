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
class LLMCfg:
    provider: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_version: Optional[str] = None
    azure_model: Optional[str] = None
    azure_embedding_deployment: Optional[str] = None
    azure_embedding_api_version: Optional[str] = None


@dataclass
class ScorerCfg:
    backend: Optional[str] = None   # "rule" | "ollama" | "openai"
    use_ema: Optional[bool] = None
    decay_factor: Optional[float] = None


@dataclass
class TopicManagerCfg:
    enable: Optional[bool] = None
    generation_prompt: Optional[str] = None


@dataclass
class EnvCfg:
    max_steps: Optional[int] = None
    max_rounds: Optional[int] = None
    include_robot: Optional[bool] = None
    max_history: Optional[int] = None
    max_history_human: Optional[int] = None
    max_history_relation: Optional[int] = None
    personas: Optional[Any] = None  # List[str] または Dict[str, Dict[str, Any]]（triggers含む）
    ref_device: Optional[Any] = None
    ref_device_map: Optional[Dict[Any, Any]] = None
    debug: Optional[bool] = None
    reward_backend: Optional[str] = None
    start_relation_check_after_utterances: Optional[int] = None
    evaluation_horizon: Optional[int] = None
    time_penalty: Optional[float] = None
    terminal_bonus: Optional[float] = None
    intervention_cost: Optional[float] = None
    min_robot_intervention_lookback: Optional[int] = None
    terminal_bonus_duration: Optional[int] = None
    max_auto_skip: Optional[int] = None


@dataclass
class WandbCfg:
    enabled: Optional[bool] = None  # wandbを使用するかどうか
    project: Optional[str] = None  # wandbプロジェクト名
    entity: Optional[str] = None  # wandb entity (username or team)
    table_log_frequency: Optional[int] = None  # N回ごとにwandb.Tableをログ


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
    decode_typical_p: Optional[float] = None
    decode_min_p: Optional[float] = None
    kl_estimator: Optional[str] = None  # KL推定器: "k1" or "k3"
    num_mini_batches: Optional[int] = None  # ミニバッチ数（更新の粒度）

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
    entropy_floor: Optional[float] = None
    entropy_patience: Optional[int] = None
    entropy_monitor_warmup: Optional[int] = None
    filter_zero_rewards: Optional[bool] = None
    topic_overlap_weight: Optional[float] = None
    topic_miss_penalty: Optional[float] = None
    topic_similarity_threshold: Optional[float] = None
    topic_overlap_horizon: Optional[int] = None
    brevity_penalty: Optional[float] = None
    brevity_min_chars: Optional[int] = None

    output_dir: Optional[str] = None
    prompt_feed_debug: Optional[bool] = None
    enable_deterministic_eval: Optional[bool] = None
    deterministic_eval_frequency: Optional[int] = None


# ============ ルート構成 ============
@dataclass
class AppConfig:
    env: EnvCfg = field(default_factory=EnvCfg)
    scorer: ScorerCfg = field(default_factory=ScorerCfg)
    ollama: OllamaCfg = field(default_factory=OllamaCfg)
    llm: LLMCfg = field(default_factory=LLMCfg)
    ppo: PPOCfg = field(default_factory=PPOCfg)
    topic_manager: TopicManagerCfg = field(default_factory=TopicManagerCfg)
    wandb: WandbCfg = field(default_factory=WandbCfg)


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

    _require("env", cfg.env, ["max_steps", "include_robot", "max_history", "debug"])
    _require("scorer", cfg.scorer, ["backend", "use_ema", "decay_factor"])
    _require("ollama", cfg.ollama, ["model"])
    _require("ollama.gen_options", getattr(cfg.ollama, "gen_options", None), ["temperature", "top_p", "num_ctx"])
    _require("llm", cfg.llm, ["provider"])
    if (cfg.llm.provider or "").lower() == "azure":
        _require(
            "llm.azure",
            cfg.llm,
            ["azure_endpoint", "azure_api_key", "azure_api_version", "azure_model"],
        )
        _require(
            "llm.azure.embedding",
            cfg.llm,
            ["azure_embedding_deployment", "azure_embedding_api_version"],
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
            "output_dir",
        ],
    )

    if missing_sections:
        joined = "\n - " + "\n - ".join(missing_sections)
        raise ValueError("Missing required configuration values:" + joined)


def load_config(yaml_path: str = "config.local.yaml") -> AppConfig:
    """config.local.yaml を（あれば）読み、既定値にマージして AppConfig を返す"""
    cfg = AppConfig()  # 既定値

    # 相対パスの場合、このファイル(config.py)があるディレクトリを基準に探す
    if not os.path.isabs(yaml_path):
        config_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(config_dir, yaml_path)

    if yaml and os.path.exists(yaml_path):
        # 前段で UTF-8 化済み想定
        with open(yaml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
    else:
        y = {}

    # セクション毎に安全マージ
    env_dict = dict(y.get("env") or {})

    # デバッグ: env_dictの内容を確認
    if env_dict.get("debug"):
        print(f"[config.py] env_dict keys: {env_dict.keys()}")
        print(f"[config.py] 'personas' in env_dict: {'personas' in env_dict}")
        if "personas" in env_dict:
            print(f"[config.py] env_dict['personas'] type: {type(env_dict['personas'])}")
            print(f"[config.py] env_dict['personas'] keys: {env_dict['personas'].keys() if isinstance(env_dict['personas'], dict) else 'N/A'}")

    # 後方互換性: トップレベルのpersonasをenv.personasにマッピング
    legacy_personas = y.get("personas")
    if legacy_personas and "personas" not in env_dict:
        env_dict["personas"] = legacy_personas

    filtered_env_dict = _filter_kwargs(EnvCfg, env_dict)
    if env_dict.get("debug"):
        print(f"[config.py] After _filter_kwargs, 'personas' in filtered: {'personas' in filtered_env_dict}")
        if "personas" in filtered_env_dict:
            print(f"[config.py] filtered personas type: {type(filtered_env_dict['personas'])}")

    env = EnvCfg(**filtered_env_dict)
    scorer = ScorerCfg(**_filter_kwargs(ScorerCfg, y.get("scorer")))
    ollama_dict = y.get("ollama") or {}
    ollama_gen = OllamaGenOpts(**_filter_kwargs(OllamaGenOpts, (ollama_dict.get("gen_options") or {})))
    ollama = OllamaCfg(
        **_filter_kwargs(OllamaCfg, {**ollama_dict, "gen_options": ollama_gen})
    )

    llm_raw = dict(y.get("llm") or {})
    legacy_openai = y.get("openai") or {}
    legacy_planner = y.get("azure_planner") or {}

    if legacy_openai:
        llm_raw.setdefault("provider", legacy_openai.get("provider") or "azure")
        llm_raw.setdefault("azure_endpoint", legacy_openai.get("azure_endpoint"))
        llm_raw.setdefault("azure_api_key", legacy_openai.get("azure_api_key") or legacy_openai.get("api_key"))
        llm_raw.setdefault("azure_api_version", legacy_openai.get("azure_api_version") or legacy_openai.get("api_version"))
        llm_raw.setdefault("azure_model", legacy_openai.get("azure_model") or legacy_openai.get("model"))
        llm_raw.setdefault("azure_embedding_deployment", legacy_openai.get("embedding_deployment"))
        llm_raw.setdefault("azure_embedding_api_version", legacy_openai.get("embedding_api_version") or legacy_openai.get("azure_api_version"))
    if legacy_planner:
        llm_raw.setdefault("azure_endpoint", legacy_planner.get("endpoint"))

    llm = LLMCfg(**_filter_kwargs(LLMCfg, llm_raw))
    ppo = PPOCfg(**_filter_kwargs(PPOCfg, y.get("ppo")))
    topic_manager = TopicManagerCfg(**_filter_kwargs(TopicManagerCfg, y.get("topic_manager")))
    wandb_cfg = WandbCfg(**_filter_kwargs(WandbCfg, y.get("wandb")))

    app_cfg = AppConfig(env=env, scorer=scorer, ollama=ollama, llm=llm, ppo=ppo, topic_manager=topic_manager, wandb=wandb_cfg)
    _validate_required_fields(app_cfg)
    return app_cfg


def get_config() -> AppConfig:
    return load_config()
