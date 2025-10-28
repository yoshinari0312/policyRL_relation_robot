#!/usr/bin/env python3
"""Azure OpenAI 基本テスト（reasoning なし）"""

import sys
import os

# appディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from azure_clients import get_azure_chat_completion_client
from config import get_config


def test_azure_basic():
    """Azure OpenAI の基本的な呼び出しをテスト（reasoning なし）"""
    print("=" * 70)
    print("Azure OpenAI 基本テスト")
    print("=" * 70)
    print()

    cfg = get_config()
    llm_cfg = getattr(cfg, "llm", None)

    if not llm_cfg:
        print("エラー: LLM設定が見つかりません")
        return

    print("【設定情報】")
    print("-" * 70)
    print(f"Provider: {getattr(llm_cfg, 'provider', 'N/A')}")
    print(f"Endpoint: {getattr(llm_cfg, 'azure_endpoint', 'N/A')}")
    print(f"API Version: {getattr(llm_cfg, 'azure_api_version', 'N/A')}")
    print(f"Model: {getattr(llm_cfg, 'azure_model', 'N/A')}")
    print()

    # クライアントを取得
    print("【クライアント接続テスト】")
    print("-" * 70)
    client, deployment = get_azure_chat_completion_client(llm_cfg)

    if not client or not deployment:
        print("❌ エラー: Azure OpenAIクライアントの取得に失敗しました")
        return

    print(f"✓ クライアント取得成功")
    print(f"✓ デプロイメント名: {deployment}")
    print()

    # API呼び出しテスト（reasoning なし）
    print("【API呼び出しテスト（reasoning なし）】")
    print("-" * 70)
    print("簡単な質問で実際にAPIを呼び出してみます...")
    print()

    try:
        test_messages = [
            {"role": "user", "content": "「こんにちは」に対して、1文で短く挨拶を返してください。"}
        ]

        print(f"リクエスト送信中...")
        print(f"  モデル: {deployment}")
        print(f"  reasoning: なし")

        # reasoning パラメータなしで呼び出し
        response = client.chat.completions.create(
            model=deployment,
            messages=test_messages
        )

        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message

            if hasattr(message, 'content'):
                content = message.content
            elif isinstance(message, dict):
                content = message.get('content', '')
            else:
                content = str(message)

            print()
            print("✓ API呼び出し成功！")
            print()
            print("【応答内容】")
            print("-" * 70)
            print(content)
            print("-" * 70)
            print()

            # 使用トークン数
            if hasattr(response, 'usage'):
                usage = response.usage
                print("【トークン使用量】")
                print(f"  Prompt tokens: {getattr(usage, 'prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {getattr(usage, 'completion_tokens', 'N/A')}")
                print(f"  Total tokens: {getattr(usage, 'total_tokens', 'N/A')}")
                print()

            print("✓✓✓ Azure API が正常に動作しています！")

        else:
            print("❌ エラー: 応答が空です")

    except Exception as e:
        print(f"❌ エラー: API呼び出しに失敗しました")
        print(f"   エラー内容: {e}")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_azure_basic()
