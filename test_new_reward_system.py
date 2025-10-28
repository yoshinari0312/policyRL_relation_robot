#!/usr/bin/env python3
"""
新しい即座報酬システムのテストスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from env.convo_env import ConversationEnv
from config import get_config

def test_basic_step():
    """基本的なstep()のテスト"""
    print("=" * 80)
    print("新しい即座報酬システムのテスト")
    print("=" * 80)

    cfg = get_config()

    # 環境を初期化
    personas = getattr(cfg.env, "personas", {})
    env = ConversationEnv(
        max_steps=5,
        personas=personas,
        include_robot=True,
        max_history=getattr(cfg.env, "max_history", 8),
        backend=getattr(cfg.scorer, "backend", "rule"),
        decay_factor=getattr(cfg.scorer, "decay_factor", 1.5),
        debug=True,
        reward_backend=getattr(cfg.env, "reward_backend", "rule"),
        evaluation_horizon=3,
        time_penalty=0.01,
        terminal_bonus=0.25,
        intervention_cost=0.02,
    )

    print("\n環境をリセット...")
    observation = env.reset()
    print(f"初期observation: {observation[:200]}...")

    print("\n" + "=" * 80)
    print("ステップ1: 介入しない")
    print("=" * 80)

    # 介入しない行動
    action = '{"intervene_now": false}'

    try:
        observation, reward, done, info = env.step(action)

        print(f"\n報酬: {reward:.4f}")
        print(f"報酬内訳: {info.get('reward_breakdown', {})}")
        print(f"バランス: {info.get('balanced', False)}")
        print(f"介入: {info.get('intervened', False)}")
        print(f"終了: {done}")

        if info.get('replies'):
            print(f"\n人間の発話:")
            for reply in info['replies']:
                speaker = reply.get('speaker', '不明')
                utterance = reply.get('utterance', '')
                print(f"  [{speaker}] {utterance}")

    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("ステップ2: 介入する（もし不安定なら）")
    print("=" * 80)

    # 介入する行動
    action = '''
    {
        "intervene_now": true,
        "edge_to_change": "AB",
        "strategy": "bridge",
        "target_speaker": "A"
    }
    '''

    try:
        observation, reward, done, info = env.step(action)

        print(f"\n報酬: {reward:.4f}")
        print(f"報酬内訳: {info.get('reward_breakdown', {})}")
        print(f"バランス: {info.get('balanced', False)}")
        print(f"介入: {info.get('intervened', False)}")
        print(f"終了: {done}")

        if info.get('robot_utterance'):
            print(f"\nロボット発話: {info['robot_utterance']}")

        if info.get('replies'):
            print(f"\n人間の発話:")
            for reply in info['replies']:
                speaker = reply.get('speaker', '不明')
                utterance = reply.get('utterance', '')
                print(f"  [{speaker}] {utterance}")

        print(f"\nactual_u_flip: {info.get('actual_u_flip')}")
        print(f"counterfactual_u_flip: {info.get('counterfactual_u_flip')}")
        print(f"delta_u_flip: {info.get('delta_u_flip')}")

    except Exception as e:
        print(f"\nエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("テスト完了！")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_basic_step()
    sys.exit(0 if success else 1)
