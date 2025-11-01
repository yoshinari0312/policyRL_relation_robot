#!/usr/bin/env python3
"""
ConversationEnvの変更をテストするスクリプト
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
    max_history=getattr(cfg.env, 'max_history', 8),
    backend=getattr(cfg.scorer, 'backend', 'azure'),
    decay_factor=getattr(cfg.scorer, 'decay_factor', 1.5),
    debug=getattr(cfg.env, 'debug', True),
    reward_backend=getattr(cfg.env, 'reward_backend', 'rule'),
    evaluation_horizon=getattr(cfg.env, 'evaluation_horizon', 3),
    time_penalty=getattr(cfg.env, 'time_penalty', 0.01),
    terminal_bonus=getattr(cfg.env, 'terminal_bonus', 0.25),
    intervention_cost=getattr(cfg.env, 'intervention_cost', 0.02),
)

print("=== Testing ConversationEnv Changes ===\n")

# reset をテスト
print("1. Testing reset() with topic generation...")
obs = env.reset()
print(f"   Current topic: {env.current_topic}")
print(f"   Used topics: {env.used_topics}")
print(f"   Logs count: {len(env.logs)}")
print(f"   Human utterances: {sum(1 for log in env.logs if log.get('speaker') != 'ロボット')}")
print()

# stepをテスト
print("2. Testing step() with intervention...")
action = '{"intervene_now": true, "edge_to_change": "AB", "strategy": "bridge", "target_speaker": "A"}'
obs, reward, done, info = env.step(action)
print(f"   Done: {done}")
print(f"   Reward: {reward:.2f}")
print(f"   Balanced: {info.get('balanced', False)}")
print(f"   Robot utterance: {info.get('robot_utterance', 'None')}")
print()

# filter_logs_by_human_count をテスト
print("3. Testing _filter_logs_by_human_count()...")
print(f"   Total logs: {len(env.logs)}")
filtered_with_robot = env._filter_logs_by_human_count(env.logs, env.max_history, exclude_robot=False)
filtered_human_only = env._filter_logs_by_human_count(env.logs, env.max_history, exclude_robot=True)
print(f"   Filtered logs (with robot): {len(filtered_with_robot)}")
print(f"   Filtered logs (human only): {len(filtered_human_only)}")
robot_count = sum(1 for log in env.logs if log.get('speaker') == 'ロボット')
print(f"   Robot utterances in logs: {robot_count}")
print()

print("=== All Tests Passed ===")
