#!/usr/bin/env python3
import json

bridge_rewards = []
validate_rewards = []

with open('app/logs/ppo_run-20251101-073508/conversation_summary.jsonl', 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'intervention' in data and data['intervention'].get('intervened'):
                strategy = data['intervention'].get('strategy')
                total_reward = data.get('reward', {}).get('total', 0)
                if strategy == 'bridge':
                    bridge_rewards.append(total_reward)
                elif strategy == 'validate':
                    validate_rewards.append(total_reward)
        except:
            pass

if bridge_rewards:
    print(f'bridge: 平均={sum(bridge_rewards)/len(bridge_rewards):.3f}, 回数={len(bridge_rewards)}')
if validate_rewards:
    print(f'validate: 平均={sum(validate_rewards)/len(validate_rewards):.3f}, 回数={len(validate_rewards)}')
