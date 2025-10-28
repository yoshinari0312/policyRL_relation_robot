"""LLM呼び出しの共通ユーティリティ"""

import time
from typing import Any, Callable, Dict, Optional, TypeVar
import requests


T = TypeVar('T')


def retry_with_backoff(
    func: Callable[[], T],
    max_attempts: int = 3,
    base_backoff: float = 1.0,
    exceptions: tuple = (Exception,)
) -> T:
    """
    指数バックオフを使用して関数を再試行する

    Args:
        func: 実行する関数
        max_attempts: 最大試行回数
        base_backoff: 初回のバックオフ時間（秒）
        exceptions: キャッチする例外のタプル

    Returns:
        関数の戻り値

    Raises:
        最後の試行で発生した例外
    """
    last_exception = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            if attempt < max_attempts:
                sleep_time = base_backoff * (2 ** (attempt - 1))
                time.sleep(sleep_time)

    raise last_exception


def call_ollama_generate(
    prompt: str,
    base_url: str,
    model: str,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    max_attempts: int = 3
) -> str:
    """
    Ollamaエンドポイントに対してテキスト生成をリクエストする（リトライ付き）

    Args:
        prompt: 生成用のプロンプト
        base_url: OllamaのベースURL（例: "http://ollama_a:11434"）
        model: 使用するモデル名
        options: 生成オプション（temperature, top_pなど）
        timeout: タイムアウト時間（秒）
        max_attempts: 最大再試行回数

    Returns:
        生成されたテキスト

    Raises:
        requests.RequestException: リクエストが失敗した場合
    """
    def _make_request() -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }

        if options:
            payload.update(options)

        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    return retry_with_backoff(
        _make_request,
        max_attempts=max_attempts,
        exceptions=(requests.RequestException,)
    )


def call_ollama_generate_safe(
    prompt: str,
    base_url: str,
    model: str,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
    max_attempts: int = 3,
    default: str = ""
) -> str:
    """
    Ollama呼び出し（例外をキャッチしてデフォルト値を返す安全版）

    Args:
        prompt: 生成用のプロンプト
        base_url: OllamaのベースURL
        model: 使用するモデル名
        options: 生成オプション
        timeout: タイムアウト時間（秒）
        max_attempts: 最大再試行回数
        default: エラー時のデフォルト値

    Returns:
        生成されたテキスト、またはエラー時はdefault値
    """
    try:
        return call_ollama_generate(
            prompt=prompt,
            base_url=base_url,
            model=model,
            options=options,
            timeout=timeout,
            max_attempts=max_attempts
        )
    except Exception:
        return default
