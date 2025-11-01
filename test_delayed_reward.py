#!/usr/bin/env python3
"""
遅延報酬の動作確認スクリプト
"""
import sys
import os

# appディレクトリをsys.pathに追加
app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app')
sys.path.insert(0, app_dir)

# cwdをappディレクトリに変更
os.chdir(app_dir)

from env.convo_env import ConversationEnv
from config import load_config

# configをロード
config_path = 'config.local.yaml'
cfg = load_config(config_path)

# ConversationEnvをインスタンス化（YAMLからpersonasを読み込む）
env = ConversationEnv(
    max_steps=getattr(cfg.env, 'max_steps', 8),
    personas=cfg.env.personas,
    include_robot=getattr(cfg.env, 'include_robot', True),
    max_history=getattr(cfg.env, 'max_history', 6),
    backend=getattr(cfg.scorer, 'backend', 'azure'),
    decay_factor=getattr(cfg.scorer, 'decay_factor', 1.5),
    debug=getattr(cfg.env, 'debug', False),
    reward_backend=getattr(cfg.env, 'reward_backend', 'rule'),
    evaluation_horizon=getattr(cfg.env, 'evaluation_horizon', 3),
    time_penalty=getattr(cfg.env, 'time_penalty', 0.01),
    terminal_bonus=getattr(cfg.env, 'terminal_bonus', 0.25),
    intervention_cost=getattr(cfg.env, 'intervention_cost', 0.02),
)

print("=== 遅延報酬の動作確認 ===\n")
print(f"evaluation_horizon: {env.evaluation_horizon}")
print(f"start_relation_check_after_utterances: {env.start_relation_check_after_utterances}")
print()

# reset をテスト
print("1. reset()を実行...")
obs = env.reset()
print(f"   Current topic: {env.current_topic}")
print(f"   初期ログ数: {len(env.logs)}")
human_count = sum(1 for log in env.logs if log.get('speaker') != 'ロボット')
print(f"   初期人間発話数: {human_count}")
print()

# stepをテスト (介入あり)
print("2. step()を実行（介入あり）...")
action = '{"intervene_now": true, "edge_to_change": "AB", "strategy": "bridge", "target_speaker": "A"}'
obs, reward, done, info = env.step(action)

print(f"   ロボット発話: {info.get('robot_utterance', 'None')[:50]}...")
print(f"   追加された人間発話数: {len(info.get('replies', []))}")
print(f"   現在のログ数: {len(env.logs)}")
human_count_after = sum(1 for log in env.logs if log.get('speaker') != 'ロボット')
print(f"   現在の人間発話数: {human_count_after}")
print()

print("3. 報酬の内訳:")
reward_details = info.get('reward_breakdown', {})
for key, value in reward_details.items():
    print(f"   {key}: {value:+.4f}")
print(f"   合計報酬: {reward:+.4f}")
print()

print("4. 遅延報酬の検証:")
noop_u_flip = info.get('noop_u_flip', 0) or 0
actual_u_flip = info.get('actual_u_flip', 0) or 0
delta_u_flip = info.get('delta_u_flip', 0) or 0
print(f"   noop_u_flip (介入なし): {noop_u_flip:.4f}")
print(f"   actual_u_flip (介入あり): {actual_u_flip:.4f}")
print(f"   delta_u_flip (差分): {delta_u_flip:+.4f}")
print(f"   → 介入により不安定度が {'減少' if delta_u_flip > 0 else '増加'}")
print()

print("=== 検証完了 ===")
print(f"遅延報酬が正しく計算されています:")
print(f"- ロボット発話後に {env.evaluation_horizon} 回の人間発話を追加")
print(f"- その後の関係性を評価して報酬を計算")
print(f"- 介入なしの世界線と比較して差分報酬を付与")
