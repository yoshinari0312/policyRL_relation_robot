#!/usr/bin/env python3
"""Azure GPT-5 with reasoning パラメータのテストスクリプト"""

import sys
import os

# appディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from config import get_config


def test_azure_gpt5_reasoning():
    """Azure GPT-5のreasoning パラメータをテスト"""
    print("=" * 70)
    print("Azure GPT-5 Reasoning パラメータのテスト")
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
    print(f"Reasoning Effort: {getattr(llm_cfg, 'reasoning_effort', 'minimal (default)')}")
    print()

    # クライアントを取得
    print("【クライアント接続テスト】")
    print("-" * 70)
    client, deployment = get_azure_chat_completion_client(llm_cfg)

    if not client or not deployment:
        print("❌ エラー: Azure OpenAIクライアントの取得に失敗しました")
        print("   - エンドポイント、APIキー、モデル名を確認してください")
        return

    print(f"✓ クライアント取得成功")
    print(f"✓ デプロイメント名: {deployment}")
    print()

    # パラメータ構築テスト
    print("【パラメータ構築テスト】")
    print("-" * 70)

    test_messages = [
        {"role": "system", "content": "あなたは親切なアシスタントです。"},
        {"role": "user", "content": "こんにちは"}
    ]

    params = build_chat_completion_params(deployment, test_messages, llm_cfg)

    print("構築されたパラメータ:")
    for key, value in params.items():
        if key == "messages":
            print(f"  {key}: [メッセージ配列 (長さ: {len(value)})]")
        else:
            print(f"  {key}: {value}")

    # GPT-5チェック
    is_gpt5 = "gpt-5" in deployment.lower() or "gpt5" in deployment.lower()
    print()
    if is_gpt5:
        print(f"✓ GPT-5モデルを検出: reasoning パラメータが追加されました")
        if "reasoning" in params:
            print(f"  reasoning: {params['reasoning']}")
    else:
        print(f"ℹ GPT-5以外のモデル: reasoning パラメータは追加されません")

    print()

    # API呼び出しテスト
    print("【API呼び出しテスト】")
    print("-" * 70)
    print("簡単な質問で実際にAPIを呼び出してみます...")
    print()

    try:
        test_messages_simple = [
            {"role": "user", "content": "「こんにちは」に対して、1文で短く挨拶を返してください。"}
        ]

        params = build_chat_completion_params(deployment, test_messages_simple, llm_cfg)

        print(f"リクエスト送信中...")
        print(f"  モデル: {deployment}")
        if "reasoning" in params:
            print(f"  reasoning: {params['reasoning']}")

        response = client.chat.completions.create(**params)

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

            # reasoning_contentがあればそれも表示
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                print("【Reasoning Content】")
                print("-" * 70)
                print(message.reasoning_content)
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

            print("✓✓✓ すべてのテストに合格しました！")

        else:
            print("❌ エラー: 応答が空です")

    except Exception as e:
        print(f"❌ エラー: API呼び出しに失敗しました")
        print(f"   エラー内容: {e}")
        print()
        print("考えられる原因:")
        print("  1. モデル名が間違っている")
        print("  2. APIキーが無効")
        print("  3. エンドポイントが間違っている")
        print("  4. reasoning パラメータがサポートされていないモデル")
        print()
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_azure_gpt5_reasoning()
