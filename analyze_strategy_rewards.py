#!/usr/bin/env python3
"""戦略ごとの報酬統計を分析"""
import json
from collections import defaultdict

strategy_rewards = defaultdict(list)
total_interventions = 0

with open('app/logs/ppo_run-20251101-073508/conversation_summary.jsonl', 'r') as f:
    for line in f:
        if not line.strip() or line.strip() == '=' * 80:
            continue
        try:
            data = json.loads(line)
            if 'intervention' in data and isinstance(data['intervention'], dict):
                if data['intervention'].get('intervened'):
                    total_interventions += 1
                    strategy = data['intervention'].get('strategy')
                    reward_data = data.get('reward', {})
                    if isinstance(reward_data, dict):
                        reward = reward_data.get('total', 0)
                    else:
                        reward = 0
                    if strategy:
                        strategy_rewards[strategy].append(reward)
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"Error: {e}")
            pass

print(f"Total interventions: {total_interventions}")
print()

for strategy in ['bridge', 'validate', 'plan']:
    rewards = strategy_rewards.get(strategy, [])
    if rewards:
        avg = sum(rewards) / len(rewards)
        positive = sum(1 for r in rewards if r > 0)
        negative = sum(1 for r in rewards if r < 0)
        print(f"{strategy:10s}: n={len(rewards):3d} ({100*len(rewards)/total_interventions:.1f}%), "
              f"avg={avg:+.3f}, min={min(rewards):+.3f}, max={max(rewards):+.3f}, "
              f"+:{positive}, -:{negative}")
    else:
        print(f"{strategy:10s}: n=  0 (never selected)")
