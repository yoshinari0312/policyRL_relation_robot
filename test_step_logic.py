#!/usr/bin/env python3
"""
step()メソッドの動作テスト

問題:
- 人間1発話でスキップされてしまう
- 介入判定が実行されない

確認項目:
1. should_skip_intervention_checkのロジック
2. auto_skipループの動作
3. 関係性評価のタイミング
"""
import sys
sys.path.insert(0, '/home/yoshinari/projects/rl-convo-policy/app')

from config import get_config
from env.convo_env import ConversationEnv

def test_step_logic():
    """step()メソッドのロジックテスト"""
    
    print("=" * 80)
    print("step()メソッドのテスト")
    print("=" * 80)
    
    # 設定を読み込み
    cfg = get_config()
    
    # 環境を初期化（デバッグモード有効）
    env = ConversationEnv(
        max_steps=8,
        personas=getattr(cfg.env, "personas", {}),
        include_robot=True,
        max_history=6,
        backend=getattr(cfg.scorer, "backend", "azure"),
        decay_factor=getattr(cfg.scorer, "decay_factor", 1.5),
        debug=True,  # デバッグモード有効
        reward_backend=getattr(cfg.env, "reward_backend", "rule"),
        evaluation_horizon=getattr(cfg.env, "evaluation_horizon", 3),
        time_penalty=getattr(cfg.env, "time_penalty", 0.01),
        terminal_bonus=getattr(cfg.env, "terminal_bonus", 0.25),
        intervention_cost=getattr(cfg.env, "intervention_cost", 0.02),
    )
    
    # リセット
    print("\n1. reset()を実行")
    observation = env.reset()
    print(f"   初期会話数: {len(env.logs)}")
    print(f"   話題: {env.current_topic}")
    
    # ステップ1: 介入する（strategy=2=bridge）
    print("\n2. step('2') を実行（bridge戦略で介入）")
    observation, reward, done, info = env.step('2')
    
    print(f"\n   結果:")
    print(f"   - 介入: {info.get('intervened', False)}")
    print(f"   - edge_to_change: {info.get('plan', {}).get('edge_to_change', 'N/A')}")
    print(f"   - target_speaker: {info.get('plan', {}).get('target_speaker', 'N/A')}")
    print(f"   - 報酬: {reward:.4f}")
    print(f"   - done: {done}")
    print(f"   - ログ数: {len(env.logs)}")
    
    if not done:
        # ステップ2: 介入しない（strategy=4=no_intervention）
        print("\n3. step('4') を実行（介入しない）")
        observation, reward, done, info = env.step('4')
        
        print(f"\n   結果:")
        print(f"   - 介入: {info.get('intervened', False)}")
        print(f"   - 報酬: {reward:.4f}")
        print(f"   - done: {done}")
        print(f"   - ログ数: {len(env.logs)}")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)


if __name__ == "__main__":
    test_step_logic()
