#!/usr/bin/env python3
"""簡易的なトピック地雷テスト"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from env.convo_env import ConversationEnv

env = ConversationEnv(max_steps=6, evaluation_horizon=3)
env.reset()

print(f'選択された地雷: {env.current_topic_trigger}')
print(f'生成されたトピック: {env.current_topic}')
print()

# 各ペルソナの地雷状況
print('各ペルソナの地雷状況:')
for persona in sorted(env.persona_triggers.keys()):
    has_trigger = env.current_topic_trigger in env.persona_triggers[persona] if env.current_topic_trigger else False
    status = '✓' if has_trigger else '✗'
    print(f'  {status} {persona}: {has_trigger}')
print()

# 会話を3発話だけ表示
print(f'会話の最初の3発話:')
for idx, log in enumerate(env.logs[:3], 1):
    print(f'  {idx}. [{log["speaker"]}] {log["utterance"]}')
