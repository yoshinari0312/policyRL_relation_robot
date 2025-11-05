"""
新形式の介入判定テスト

LLMの出力形式を変更:
- 旧形式: JSON {"intervene_now": true, "strategy": "validate", "edge_to_change": "AB", "target_speaker": "A"}
- 新形式: 数字のみ 1, 2, 3, 4
  - 1: validate
  - 2: bridge
  - 3: plan
  - 4: no_intervention

edge_to_changeは自動決定（最も絶対値が小さい負のエッジ）
target_speakerはランダム決定
"""

from env.convo_env import ConversationEnv
from config import get_config
import sys

def test_new_format():
    print("=" * 80)
    print("新形式の介入判定テスト")
    print("=" * 80)
    print()
    
    cfg = get_config()
    
    # 環境の初期化
    env = ConversationEnv(
        max_steps=10,
        personas=None,
        include_robot=True,
        max_history=12,
        backend=None,
        decay_factor=1.5,
        debug=False,  # デバッグモード無効（簡潔な出力）
        reward_backend="rule",
        evaluation_horizon=2,
        time_penalty=0.04,
        terminal_bonus=1.0,
        intervention_cost=0.35,
        max_auto_skip=3,
    )
    
    # 環境のリセット
    obs = env.reset()
    print(f"初期観測: {obs[:200]}...")
    print()
    
    # テストケース1: validate（数字1）
    print("テストケース1: validate（数字1）")
    print("-" * 80)
    action = "1"
    print(f"アクション: {action}")
    obs, reward, done, info = env.step(action)
    print(f"報酬: {reward:.4f}")
    print(f"介入: {info.get('intervened', False)}")
    if info.get('plan'):
        plan = info['plan']
        print(f"  戦略: {plan.get('strategy')}")
        print(f"  エッジ: {plan.get('edge_to_change')}")
        print(f"  ターゲット: {plan.get('target_speaker')}")
    print(f"完了: {done}")
    print()
    
    # テストケース2: bridge（数字2）
    if not done:
        print("テストケース2: bridge（数字2）")
        print("-" * 80)
        action = "2"
        print(f"アクション: {action}")
        obs, reward, done, info = env.step(action)
        print(f"報酬: {reward:.4f}")
        print(f"介入: {info.get('intervened', False)}")
        if info.get('plan'):
            plan = info['plan']
            print(f"  戦略: {plan.get('strategy')}")
            print(f"  エッジ: {plan.get('edge_to_change')}")
            print(f"  ターゲット: {plan.get('target_speaker')}")
        print(f"完了: {done}")
        print()
    
    # テストケース3: no_intervention（数字4）
    if not done:
        print("テストケース3: no_intervention（数字4）")
        print("-" * 80)
        action = "4"
        print(f"アクション: {action}")
        obs, reward, done, info = env.step(action)
        print(f"報酬: {reward:.4f}")
        print(f"介入: {info.get('intervened', False)}")
        if info.get('plan'):
            plan = info['plan']
            print(f"  介入なし: {not plan.get('intervene_now', True)}")
        print(f"完了: {done}")
        print()
    
    # テストケース4: 旧形式のJSONも動作するか確認（後方互換性）
    if not done:
        env.reset()
        print("テストケース4: 旧形式JSON（後方互換性確認）")
        print("-" * 80)
        action = '{"intervene_now": true, "edge_to_change": "BC", "strategy": "plan", "target_speaker": "B"}'
        print(f"アクション: {action}")
        obs, reward, done, info = env.step(action)
        print(f"報酬: {reward:.4f}")
        print(f"介入: {info.get('intervened', False)}")
        if info.get('plan'):
            plan = info['plan']
            print(f"  戦略: {plan.get('strategy')}")
            print(f"  エッジ: {plan.get('edge_to_change')}")
            print(f"  ターゲット: {plan.get('target_speaker')}")
        print(f"完了: {done}")
        print()
    
    print("=" * 80)
    print("テスト完了")
    print("=" * 80)

if __name__ == "__main__":
    test_new_format()

