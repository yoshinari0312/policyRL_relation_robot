#!/usr/bin/env python3
"""トピック生成と地雷フィルタリングのテストスクリプト"""

import sys
import os

# appディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from env.convo_env import ConversationEnv
from config import get_config


def test_topic_trigger():
    """トピック生成と地雷フィルタリングの動作をテスト"""
    print("=" * 80)
    print("トピック生成と地雷フィルタリングのテスト")
    print("=" * 80)
    print()

    cfg = get_config()

    print("【環境の初期化】")
    print("-" * 80)
    # 3発話生成（テスト用）
    env = ConversationEnv(max_steps=6, evaluation_horizon=3)
    print("✓ ConversationEnv初期化成功")
    print()

    # 3回リセットして、異なるトピックと地雷が選択されることを確認
    for i in range(3):
        print(f"【テストケース{i+1}】エピソードのリセット")
        print("-" * 80)

        env.reset()

        print(f"選択された地雷: {env.current_topic_trigger}")
        print(f"生成されたトピック: {env.current_topic}")
        print()

        # 各ペルソナがこの地雷を持っているかチェック
        print("各ペルソナの地雷状況:")
        for persona, triggers in sorted(env.persona_triggers.items()):
            has_trigger = env.current_topic_trigger in triggers if env.current_topic_trigger else False
            status = "✓" if has_trigger else "✗"
            print(f"  {status} {persona}: {has_trigger}")
        print()

        # 各ペルソナの人間LLMプロンプトを表示
        print("各ペルソナの人間LLMプロンプト:")
        print("-" * 80)
        from humans_ollama import build_human_prompt
        for persona in sorted(env.persona_triggers.keys()):
            print(f"\n【{persona}のプロンプト】")
            print("-" * 40)
            prompt = build_human_prompt(
                speaker=persona,
                logs=env.logs[:3] if env.logs else [],  # 最初の3発話だけ使用
                topic=env.current_topic,
                topic_trigger=env.current_topic_trigger
            )
            print(prompt)
            print("-" * 40)
        print()

        # 会話を確認（すべて表示）
        if env.logs:
            print(f"会話履歴（{len(env.logs)}発話）:")
            for idx, log in enumerate(env.logs, 1):
                print(f"  {idx:2d}. [{log['speaker']}] {log['utterance']}")
            print()

        print()

    print("=" * 80)
    print("テスト完了！")
    print("=" * 80)


if __name__ == "__main__":
    test_topic_trigger()
