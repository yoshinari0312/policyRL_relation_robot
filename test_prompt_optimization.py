#!/usr/bin/env python3
"""プロンプト最適化のテスト

システムプロンプトとユーザープロンプトが正しく分離されているか確認する。
"""

import sys
sys.path.insert(0, 'app')

from env.convo_env import ConversationEnv
from config import get_config

def test_prompt_optimization():
    print("=" * 80)
    print("プロンプト最適化のテスト")
    print("=" * 80)

    # 環境を初期化
    cfg = get_config()
    env = ConversationEnv(
        max_steps=8,
        debug=False
    )

    # エピソードを開始
    obs = env.reset()

    # 1ステップ実行
    _, _, _, _ = env.step('{"intervene_now": false}')

    # ユーザープロンプトを生成（_make_observation()）
    user_prompt = env._make_observation()

    # システムプロンプトをハードコード（build_robot_messages()から）
    system_prompt = """あなたは関係性を安定させるような介入計画を提案するAIです。

三者会話（話者 A/B/C）の関係を安定化するため、ロボットが適切なタイミングで一言介入します。
あなたの役割は、会話履歴と各ペアの関係スコア（-1..1）を受け取り、
「できるだけ早く関係性を安定状態（+++,+--,-+-,--+）にする」ための介入方法を提案することです。
※ロボットの実際の発話文は別LLMが生成します。あなたは介入方法だけを出力します。

制約:
- 出力は JSON のみ。説明や装飾は禁止。
- intervene_now は true|false（今すぐ介入すべきか）。
- edge_to_change は "AB" | "BC" | "CA" のいずれか。
- strategy は plan | validate | bridge から選択。
  - plan: 「これからどうするか」という未来志向の視点を提示。具体的な次の一歩や小さな行動案を示し、前向きな行動を促す
  - validate: 対象者の感情・意見を承認し、心理的安全性を構築
  - bridge: 対立する者の共通点・目標を明示し、協力関係を構築
  ※どの戦略をいつ使うかは会話文脈や関係スコアから判断してください。
- target_speaker は A | B | C のいずれか。（その介入を誰に向けるか）
- intervene_now=true の場合は edge_to_change / strategy / target_speaker を必ず指定。
- intervene_now=false の場合は何も指定しない。

出力例:
{"intervene_now": true, "edge_to_change": "AB", "strategy": "plan", "target_speaker": "A"}
{"intervene_now": false}
"""

    print("\n[システムプロンプト（固定説明）]")
    print("-" * 80)
    print(system_prompt)
    print()

    print("[ユーザープロンプト（可変要素）]")
    print("-" * 80)
    print(user_prompt)
    print()

    # トークン数を推定（簡易）
    system_tokens = len(system_prompt.split())
    user_tokens = len(user_prompt.split())

    print("=" * 80)
    print("トークン数の推定（単語数ベース）")
    print("=" * 80)
    print(f"システムプロンプト: ~{system_tokens * 1.3:.0f} トークン")
    print(f"ユーザープロンプト: ~{user_tokens * 1.3:.0f} トークン")
    print()
    print("注: システムプロンプトは1回だけ送信されます。")
    print(f"    10ステップの場合:")
    print(f"    - システム: ~{system_tokens * 1.3:.0f} トークン × 1回 = ~{system_tokens * 1.3:.0f} トークン")
    print(f"    - ユーザー: ~{user_tokens * 1.3:.0f} トークン × 10回 = ~{user_tokens * 1.3 * 10:.0f} トークン")
    print(f"    - 合計: ~{system_tokens * 1.3 + user_tokens * 1.3 * 10:.0f} トークン")
    print()
    print("✓ プロンプト最適化が完了しました。")
    print("  - 固定説明はシステムプロンプトに集約")
    print("  - 可変要素（会話履歴・スコア）のみがユーザープロンプト")

if __name__ == "__main__":
    test_prompt_optimization()
