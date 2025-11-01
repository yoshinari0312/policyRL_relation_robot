#!/usr/bin/env python3
"""filter_zero_rewards機能のテスト

安定状態のターンがバッチサイズにカウントされず、
不安定なターンのみが訓練データに含まれることを確認する。
"""

import sys
sys.path.insert(0, 'app')

from env.convo_env import ConversationEnv
from config import get_config

def test_stable_skip():
    """安定状態が正しくスキップされるかテスト"""

    print("=" * 80)
    print("安定状態スキップのテスト")
    print("=" * 80)

    # 環境を初期化
    cfg = get_config()
    env = ConversationEnv(
        max_steps=15,
        debug=True
    )

    print(f"\nfilter_zero_rewards設定: {getattr(cfg.ppo, 'filter_zero_rewards', False)}")
    print()

    # エピソードを開始
    obs = env.reset()
    print(f"エピソード開始: {env.episode}")
    print(f"トピック: {env.current_topic}")
    print()

    # 最初の状態を確認
    rel = env.relation_snapshot()
    if isinstance(rel, dict):
        metrics = rel.get("metrics", rel)
        unstable_count = metrics.get("unstable_triads", 0)
        edges = metrics.get("edges", {})
    else:
        unstable_count = 0
        edges = {}

    print(f"初期状態:")
    print(f"  不安定トライアド数: {unstable_count}")
    print(f"  エッジ: {edges}")
    print()

    # 最初の数ステップを実行
    stable_count = 0
    unstable_count_turns = 0

    for i in range(15):
        # ステップ実行前の状態
        rel_before = env.relation_snapshot()
        if isinstance(rel_before, dict):
            metrics_before = rel_before.get("metrics", rel_before)
            unstable_before = metrics_before.get("unstable_triads", 0)
        else:
            unstable_before = 0

        is_stable_before = unstable_before == 0

        print(f"--- ステップ {i + 1} ---")
        print(f"ステップ前の状態: {'安定' if is_stable_before else '不安定'} (不安定トライアド数: {unstable_before})")

        # 介入なしでステップを実行
        obs, reward, done, info = env.step('{"intervene_now": false}')

        # ステップ実行後の状態
        rel_after = info.get("rel", {})
        if isinstance(rel_after, dict):
            unstable_after = rel_after.get("unstable_triads", 0)
        else:
            unstable_after = 0

        is_stable_after = unstable_after == 0

        print(f"ステップ後の状態: {'安定' if is_stable_after else '不安定'} (不安定トライアド数: {unstable_after})")
        print(f"報酬: {reward}")
        print(f"介入: {info.get('intervened', False)}")
        print()

        if is_stable_before:
            stable_count += 1
        else:
            unstable_count_turns += 1

        if done:
            print(f"エピソード終了（ステップ {i + 1}）")
            print(f"  _episode_step={env._episode_step}")
            print(f"  _unstable_step={env._unstable_step}")
            print(f"  max_steps={env.max_steps}")
            break

    print("=" * 80)
    print("テスト結果")
    print("=" * 80)
    print(f"安定なターン数: {stable_count}")
    print(f"不安定なターン数: {unstable_count_turns}")
    print(f"_episode_step: {env._episode_step}")
    print(f"_unstable_step: {env._unstable_step}")
    print()

    if stable_count > 0:
        print("✓ 安定状態のターンが存在しました")
        print("  → FiniteOnlineDataset.__iter__()でスキップされるはずです")
    else:
        print("✗ 安定状態のターンが見つかりませんでした")
        print("  → 環境が常に不安定な状態を生成している可能性があります")

    print()
    print("注: 実際の訓練では:")
    print("  1. FiniteOnlineDataset.__iter__()が安定状態を自動的にスキップ")
    print("  2. 不安定なターンのみが訓練データとして使用される")
    print("  3. エピソードは_unstable_step >= max_stepsで終了")
    print(f"  4. max_steps={env.max_steps}なので、不安定なターンが{env.max_steps}回発生したら終了")

if __name__ == "__main__":
    test_stable_skip()
