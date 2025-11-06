#!/usr/bin/env python3
import json

strategy_rewards = {
    'validate': [],
    'bridge': [],
    'plan': [],
    'no_intervention': [],
    'output_error': []
}

with open('app/logs/ppo_run-20251101-073508/conversation_summary.jsonl', 'r') as f:
    for line in f:
        try:
            data = json.loads(line)
            if 'intervention' in data:
                intervention = data['intervention']
                # Get strategy from intervention dict (works for both intervened=true and false)
                strategy = intervention.get('strategy')
                if strategy and strategy in strategy_rewards:
                    total_reward = data.get('reward', {}).get('total', 0)
                    strategy_rewards[strategy].append(total_reward)
        except:
            pass

print("5戦略分析 (validate, bridge, plan, no_intervention, output_error)")
print("=" * 60)
for strategy in ['validate', 'bridge', 'plan', 'no_intervention', 'output_error']:
    rewards = strategy_rewards[strategy]
    if rewards:
        avg = sum(rewards) / len(rewards)
        print(f'{strategy:15s}: 平均={avg:+.3f}, 回数={len(rewards)}')
    else:
        print(f'{strategy:15s}: 回数=0 (未使用)')
