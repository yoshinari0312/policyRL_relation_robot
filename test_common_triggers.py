#!/usr/bin/env python3
"""共通地雷検出機能のテストスクリプト"""

from app.utils.trigger_detection import (
    get_common_triggers_for_others,
    format_common_triggers_message,
)
from app.config import get_config


def test_common_trigger_detection():
    """共通地雷検出のロジックテスト"""
    print("=" * 60)
    print("共通地雷検出機能のテスト")
    print("=" * 60)
    print()

    # テストケース1: config.local.yamlの実際の設定を使用
    print("【テストケース1】config.local.yamlの設定を使用")
    print("-" * 60)

    cfg = get_config()
    personas_cfg = getattr(cfg.env, "personas", None)

    if personas_cfg and isinstance(personas_cfg, dict):
        all_triggers = {}
        for persona_name in personas_cfg.keys():
            persona_info = personas_cfg.get(persona_name, {})
            if isinstance(persona_info, dict):
                persona_triggers = persona_info.get("triggers", []) or []
                # 空文字列を除外
                all_triggers[persona_name] = [t for t in persona_triggers if t and t.strip()]

        print("各ペルソナの地雷:")
        for speaker, triggers in all_triggers.items():
            print(f"  {speaker}: {triggers}")
        print()

        # 各ペルソナについて、他の2人に共通する地雷を検出
        for speaker in all_triggers.keys():
            common = get_common_triggers_for_others(speaker, all_triggers)
            other_speakers = [s for s in all_triggers.keys() if s != speaker]

            print(f"ペルソナ {speaker} の場合:")
            print(f"  他の話者: {other_speakers}")
            print(f"  共通地雷: {common}")

            if common:
                msg = format_common_triggers_message(common, other_speakers)
                print(f"  メッセージ: {msg}")
            else:
                print(f"  メッセージ: （共通地雷なし）")
            print()

    print()

    # テストケース2: 手動設定でのテスト
    print("【テストケース2】手動設定でのテスト")
    print("-" * 60)

    test_triggers = {
        "A": ["お金", "勉強", "テクノロジー"],
        "B": ["お金", "勉強", "ゲーム"],
        "C": ["お金", "テクノロジー", "ゲーム"],
    }

    print("各ペルソナの地雷:")
    for speaker, triggers in test_triggers.items():
        print(f"  {speaker}: {triggers}")
    print()

    for speaker in test_triggers.keys():
        common = get_common_triggers_for_others(speaker, test_triggers)
        other_speakers = [s for s in test_triggers.keys() if s != speaker]

        print(f"ペルソナ {speaker} の場合:")
        print(f"  他の話者: {other_speakers}")
        print(f"  共通地雷: {common}")
        print(f"  期待される共通地雷:")
        if speaker == "A":
            print(f"    B∩C = {set(test_triggers['B']) & set(test_triggers['C'])} (お金、ゲーム)")
        elif speaker == "B":
            print(f"    A∩C = {set(test_triggers['A']) & set(test_triggers['C'])} (お金、テクノロジー)")
        elif speaker == "C":
            print(f"    A∩B = {set(test_triggers['A']) & set(test_triggers['B'])} (お金、勉強)")

        if common:
            msg = format_common_triggers_message(common, other_speakers)
            print(f"  メッセージ: {msg}")
        print()

    print()

    # テストケース3: プロンプト生成のテスト
    print("【テストケース3】プロンプト生成のテスト")
    print("-" * 60)

    try:
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))
        from humans_ollama import build_human_prompt

        # 空の会話ログでプロンプトを生成
        test_logs = []
        test_topic = "最近の趣味について"

        for speaker in ["A", "B", "C"]:
            print(f"\nペルソナ {speaker} のプロンプト:")
            print("=" * 60)
            prompt = build_human_prompt(speaker, test_logs, test_topic)
            print(prompt)
            print("=" * 60)
    except Exception as e:
        print(f"プロンプト生成テストをスキップ（モジュールインポートエラー: {e}）")

    print()
    print("テスト完了!")


if __name__ == "__main__":
    test_common_trigger_detection()
