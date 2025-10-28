from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _normalize(value: Any) -> Optional[str]:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None

_CHAT_CLIENT_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _build_chat_client(endpoint: str, api_key: str, api_version: str):
    key = (endpoint, api_key, api_version)
    cached = _CHAT_CLIENT_CACHE.get(key)
    if cached is not None:
        return cached

    try:
        from openai import AzureOpenAI  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        return None

    try:
        client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=api_key)
    except Exception:
        return None

    _CHAT_CLIENT_CACHE[key] = client
    return client


def get_azure_chat_completion_client(
    llm_cfg: Any,
    model_type: Optional[str] = None
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Azure OpenAI chat completion クライアントとモデル名を取得する。

    Args:
        llm_cfg: LLM設定オブジェクト
        model_type: モデルタイプ（"human", "relation", "robot", "topic", "intervention"）
                   Noneの場合はデフォルトモデルを使用

    Returns:
        (client, deployment) のタプル
    """
    if llm_cfg is None:
        return None, None

    endpoint = _normalize(getattr(llm_cfg, "azure_endpoint", None))
    api_key = _normalize(getattr(llm_cfg, "azure_api_key", None))
    api_version = _normalize(getattr(llm_cfg, "azure_api_version", None))

    # モデルタイプに応じてデプロイメント名を取得
    if model_type:
        model_attr = f"{model_type}_model"
        deployment = _normalize(getattr(llm_cfg, model_attr, None))
        if not deployment:
            # フォールバック: デフォルトモデルを使用
            deployment = _normalize(getattr(llm_cfg, "azure_model", None))
    else:
        deployment = _normalize(getattr(llm_cfg, "azure_model", None))

    if not deployment:
        # さらにフォールバック: embeddingデプロイメントを試す
        deployment = _normalize(getattr(llm_cfg, "azure_embedding_deployment", None))

    if not (endpoint and api_key and api_version and deployment):
        return None, None

    client = _build_chat_client(endpoint, api_key, api_version)
    if client is None:
        return None, None

    return client, deployment


def build_chat_completion_params(
    deployment: str,
    messages: list,
    llm_cfg: Any = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Azure OpenAI chat completion のパラメータを構築する。
    GPT-5 モデルの場合は reasoning パラメータを自動で追加する。

    Args:
        deployment: モデルのデプロイメント名
        messages: チャットメッセージのリスト
        llm_cfg: LLM設定オブジェクト（オプション）
        **kwargs: その他のパラメータ

    Returns:
        chat.completions.create に渡すパラメータ辞書
    """
    params = {
        "model": deployment,
        "messages": messages,
        **kwargs
    }

    # GPT-5 モデルの場合は reasoning パラメータを extra_body に追加
    # ただし、enable_reasoning_param が False の場合はスキップ
    if deployment and ("gpt-5" in deployment.lower() or "gpt5" in deployment.lower()):
        # デフォルトでは無効（Azure では現在サポートされていない）
        enable_reasoning = False
        if llm_cfg:
            enable_reasoning = getattr(llm_cfg, "enable_reasoning_param", False)

        if enable_reasoning:
            # configから設定を読み込む（デフォルトは "minimal"）
            reasoning_effort = "minimal"
            if llm_cfg:
                reasoning_effort = getattr(llm_cfg, "reasoning_effort", "minimal")

            # extra_body として渡す（将来的にサポートされる可能性に備えて）
            params["extra_body"] = {
                "reasoning": {"effort": reasoning_effort}
            }

    return params
