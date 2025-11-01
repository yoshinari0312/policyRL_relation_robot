#!/usr/bin/env python3
"""max_auto_skipの挙動をテスト

安定状態がmax_auto_skip回連続で続いた場合、done=Trueになるかを確認する。
"""

import sys
sys.path.insert(0, 'app')

from env.convo_env import ConversationEnv
from config import get_config

def test_max_auto_skip():
    print("=" * 80)
    print("max_auto_skip テスト")
    print("=" * 80)

    # 環境を初期化（max_auto_skip=3に設定してテストを短くする）
    cfg = get_config()
    env = ConversationEnv(
        max_steps=cfg.env.max_steps,
        debug=True
    )

    # max_auto_skipの値を確認
    max_auto_skip = env.max_auto_skip
    print(f"\nmax_auto_skip: {max_auto_skip}")
    print(f"max_steps: {env.max_steps}")
    print()

    # エピソードを開始
    obs = env.reset()
    print(f"エピソード {env.episode} 開始")
    print(f"トピック: {env.current_topic}")
    print()

    # ステップを実行して、done=Trueになるのを待つ
    step_count = 0
    max_iterations = 100  # 無限ループ防止

    while step_count < max_iterations:
        step_count += 1

        print(f"\n--- ステップ {step_count} ---")
        print(f"env.t: {env.t}")
        print(f"_episode_step: {env._episode_step}")
        print(f"_unstable_step: {env._unstable_step}")

        # 介入しない選択を実行
        obs, reward, done, info = env.step('{"intervene_now": false}')

        # 関係性の状態を取得
        rel_info = env.relation_snapshot()
        is_stable = rel_info["metrics"]["unstable_triads"] == 0

        print(f"安定状態: {is_stable}")
        print(f"不安定トライアド数: {rel_info['metrics']['unstable_triads']}")
        print(f"報酬: {reward}")
        print(f"done: {done}")

        if done:
            print(f"\n✓ エピソード終了")
            print(f"  ステップ数: {step_count}")
            print(f"  _episode_step: {env._episode_step}")
            print(f"  _unstable_step: {env._unstable_step}")

            # info辞書を確認
            if "auto_skip_triggered" in info:
                print(f"  auto_skip triggered: {info['auto_skip_triggered']}")

            break

    if step_count >= max_iterations:
        print(f"\n✗ テスト失敗: {max_iterations}ステップ経過してもエピソードが終了しませんでした")
        return False

    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_max_auto_skip()
    sys.exit(0 if success else 1)
