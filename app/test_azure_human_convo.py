"""
Azure OpenAI ベースの人間同士会話テスト（本番コード使用版）

使い方:
    python app/test_azure_human_convo.py

特徴:
- 本番ファイル（convo_env.py, humans_ollama.py, network_metrics.pyなど）の関数・パラメータを使用
- reset()の内部処理を1発話ずつ可視化
- 理想的な挙動を確認可能

理想的な挙動:
1. 1人が発話したらすぐに関係性推定を実行（3発話以上の場合）
2. 安定なら何もしない（次の人間発話へ）
3. 不安定なら介入判定
4. 介入する場合:
   - ロボット発話
   - evaluation_horizon回の人間発話を生成（1人ずつ）
   - 関係性判断
   - 安定になったら、terminal_bonus_duration回追加で人間発話を生成して安定持続を確認
   - 報酬計算
5. 介入しない場合:
   - タイムペナルティのみ
"""
from __future__ import annotations
import json
import os
from typing import List, Dict, Tuple

from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from config import get_config
from env.convo_env import ConversationEnv
from utils import filter_logs_by_human_count
from network_metrics import analyze_relations_from_scores
from humans_ollama import human_reply


def run_conversation_test(max_turns: int = 20):
    """
    会話テストを実行（本番コードの関数を使用）

    Args:
        max_turns: 最大ターン数
    """
    # 設定を読み込み
    cfg = get_config()

    # ConversationEnvを初期化
    personas = getattr(cfg.env, "personas", {})
    env = ConversationEnv(
        max_steps=max_turns,
        personas=personas,
        include_robot=True,
        max_history=getattr(cfg.env, "max_history", 12),
        backend=getattr(cfg.scorer, "backend", "azure"),
        decay_factor=getattr(cfg.scorer, "decay_factor", 1.5),
        debug=getattr(cfg.env, "debug", False),
        reward_backend=getattr(cfg.env, "reward_backend", "rule"),
        evaluation_horizon=getattr(cfg.env, "evaluation_horizon", 2),
        time_penalty=getattr(cfg.env, "time_penalty", 0.01),
        terminal_bonus=getattr(cfg.env, "terminal_bonus", 0.25),
        intervention_cost=getattr(cfg.env, "intervention_cost", 0.02),
    )

    print("\n" + "=" * 80)
    print("会話テスト開始（本番コード使用）")
    print("=" * 80)

    # ============================================================================
    # reset()の内部処理を手動で展開（1発話ずつ可視化するため）
    # ============================================================================

    # reset()の初期化部分（convo_env.py:150-161行目相当）
    env.episode += 1
    env._episode_step = 0
    env.t = 0
    env.logs = []
    env.scorer = type(env.scorer)(**env._scorer_kwargs)
    env.steps_since_last_intervention = 0
    env.stable_step_start = None
    env.terminal_bonus_given = False

    # トピックを生成（convo_env.py:163-165行目相当）
    env.current_topic = env._get_topic_suggestion()
    env.used_topics.append(env.current_topic)

    print(f"\n📌 話題: {env.current_topic}")
    if env.current_topic_trigger:
        print(f"🎯 選ばれた地雷ワード: {env.current_topic_trigger}")
    else:
        print(f"🎯 選ばれた地雷ワード: なし")
    print(f"👥 ペルソナ: {', '.join(env.persona_pool)}")

    print("\n" + "=" * 80)
    print("初期化フェーズ（不安定シード生成）")
    print("=" * 80)

    # _bootstrap_humans(evaluation_horizon)を1人ずつ展開（convo_env.py:167行目相当）
    print(f"\n💬 初期会話生成（1人ずつ{env.evaluation_horizon}回、各発話後に関係性推定）")

    persona_index = 0
    initial_turns = 0
    target_initial_turns = env.evaluation_horizon

    while initial_turns < target_initial_turns:
        current_persona = env.persona_pool[persona_index % len(env.persona_pool)]
        persona_index += 1

        print(f"\n👤 [{current_persona}] 発話生成中...")
        # 本番コードのhuman_reply()を使用（1人ずつ）
        replies = human_reply(
            env.logs,
            [current_persona],
            topic=env.current_topic,
            topic_trigger=env.current_topic_trigger
        )
        env.logs.extend(replies)

        # 発話を表示
        for reply in replies:
            speaker = reply.get("speaker", "?")
            utterance = reply.get("utterance", "")
            print(f"[{speaker}] {utterance}")

        # 人間発話数をカウント
        human_count = sum(1 for log in env.logs if log.get("speaker") != "ロボット")

        # 毎回関係性推定を試みる（3発話未満ならスキップ表示）
        if human_count >= env.start_relation_check_after_utterances:
            participants = env._participants(env.logs)
            filtered_logs = filter_logs_by_human_count(env.logs, env.max_history, exclude_robot=True)
            scores, _ = env._relation_state(filtered_logs, update_state=True)

            print(f"\n📊 関係性推定（{human_count}発話時点）:")
            for (a, b), v in sorted(scores.items()):
                print(f"  relation {a}-{b}: {v:+.2f}")

            # 安定判定
            metrics, _ = env._metrics_state(scores, participants)
            unstable_triads = metrics.get("unstable_triads", 0)
            is_stable = unstable_triads == 0

            if is_stable:
                print(f"  状態: 安定")
            else:
                print(f"  状態: 不安定（不安定トライアド数={unstable_triads}）")
        else:
            print(f"（人間発話数 {human_count} < {env.start_relation_check_after_utterances}、関係性推定スキップ）")
        print()

        initial_turns += 1

    # _ensure_unstable_seed()を展開（convo_env.py:168行目相当）
    print(f"🔄 不安定シード確保中（安定なら追加で会話生成）...")

    attempts = 0
    while env._is_balanced() and attempts < env.max_steps:
        print(f"\n  ⚠️ まだ安定状態 → 1人ずつ{env.evaluation_horizon}回追加生成（各発話後に関係性推定）")

        for turn_in_attempt in range(env.evaluation_horizon):
            current_persona = env.persona_pool[persona_index % len(env.persona_pool)]
            persona_index += 1

            print(f"\n  👤 [{current_persona}] 発話生成中...")
            replies = human_reply(
                env.logs,
                [current_persona],
                topic=env.current_topic,
                topic_trigger=env.current_topic_trigger
            )
            env.logs.extend(replies)

            # 発話を表示
            for reply in replies:
                speaker = reply.get("speaker", "?")
                utterance = reply.get("utterance", "")
                print(f"  [{speaker}] {utterance}")

            # 人間発話数をカウント
            human_count = sum(1 for log in env.logs if log.get("speaker") != "ロボット")

            # 毎回関係性推定を試みる（3発話未満ならスキップ表示）
            if human_count >= env.start_relation_check_after_utterances:
                participants = env._participants(env.logs)
                filtered_logs = filter_logs_by_human_count(env.logs, env.max_history, exclude_robot=True)
                scores, _ = env._relation_state(filtered_logs, update_state=True)

                print(f"\n  📊 関係性推定（{human_count}発話時点）:")
                for (a, b), v in sorted(scores.items()):
                    print(f"    relation {a}-{b}: {v:+.2f}")

                metrics, _ = env._metrics_state(scores, participants)
                unstable_triads = metrics.get("unstable_triads", 0)
                is_stable = unstable_triads == 0

                if is_stable:
                    print(f"    状態: 安定")
                else:
                    print(f"    状態: 不安定（不安定トライアド数={unstable_triads}）")
            else:
                print(f"  （人間発話数 {human_count} < {env.start_relation_check_after_utterances}、関係性推定スキップ）")
            print()

        attempts += 1

    print(f"✅ 不安定シード確保完了（総{len(env.logs)}発話）")

    # ============================================================================
    # ここから通常の会話フロー
    # ============================================================================

    print("\n" + "=" * 80)
    print("会話フロー開始（1人ずつ発話→関係性推定）")
    print("=" * 80)

    # 設定パラメータ
    start_relation_check_after_utterances = env.start_relation_check_after_utterances
    evaluation_horizon = env.evaluation_horizon
    terminal_bonus_duration = env.terminal_bonus_duration
    time_penalty = env.time_penalty
    intervention_cost = env.intervention_cost
    terminal_bonus = env.terminal_bonus
    max_history = env.max_history

    turn_count = 0

    while turn_count < max_turns:
        turn_count += 1
        print(f"\n{'─' * 80}")
        print(f"ターン {turn_count}")
        print(f"{'─' * 80}")

        # 現在の話者を決定
        current_persona = env.persona_pool[persona_index % len(env.persona_pool)]
        persona_index += 1

        # 1. 人間発話を1人分生成（本番コードのhuman_reply関数を使用）
        print(f"\n👤 [{current_persona}] 発話生成中...")
        replies = human_reply(
            env.logs,
            [current_persona],  # 1人だけ渡す
            topic=env.current_topic,
            topic_trigger=env.current_topic_trigger
        )

        # logsに追加
        env.logs.extend(replies)

        # 発話を表示
        for reply in replies:
            speaker = reply.get("speaker", "?")
            utterance = reply.get("utterance", "")
            print(f"[{speaker}] {utterance}")

        # 人間発話数をカウント
        human_count = sum(1 for log in env.logs if log.get("speaker") != "ロボット")

        # 2. 関係性推定（3発話以上の場合）
        if human_count < start_relation_check_after_utterances:
            print(f"  （人間発話数 {human_count} < {start_relation_check_after_utterances}、関係性推定スキップ）")
            continue

        print(f"\n📊 関係性推定中...")

        # 関係性LLM用: ロボット発話を除外（本番コードと同じ処理）
        participants = env._participants(env.logs)
        filtered_logs = filter_logs_by_human_count(env.logs, max_history, exclude_robot=True)
        scores, _ = env._relation_state(filtered_logs, update_state=True)

        # スコアを表示
        for (a, b), v in sorted(scores.items()):
            print(f"  relation {a}-{b}: {v:+.2f}")

        # メトリクスを計算
        metrics, _ = env._metrics_state(scores, participants)
        unstable_triads = metrics.get("unstable_triads", 0)
        is_stable = unstable_triads == 0

        # 3. 安定判定
        if is_stable:
            print("（関係性推定：安定）")
            print("  → 介入判定スキップ")
            continue

        print(f"（関係性推定：不安定、不安定トライアド数={unstable_triads}）")

        # 4. 介入判定（本番コードのメソッドを使用）
        print(f"\n🤖 介入判定中...")

        # 観測プロンプトを取得（本番コードと同じ）
        observation = env._make_observation()

        # Azure OpenAIで介入判定を実行（本番と同じ設定を使用）
        client, deployment = get_azure_chat_completion_client(getattr(cfg, "llm", None), model_type="intervention")
        if not client or not deployment:
            print("  ⚠️ Azure OpenAI client unavailable, fallback to no intervention")
            plan = {"intervene_now": False}
        else:
            messages = [
                {
                    "role": "system",
                    "content": "あなたは会話の関係性を安定化するためのロボット介入プランナーです。"
                               "指示に従って、JSON形式で介入計画を出力してください。"
                },
                {"role": "user", "content": observation}
            ]

            try:
                params = build_chat_completion_params(deployment, messages, cfg.llm)
                res = client.chat.completions.create(**params)
                if res and getattr(res, "choices", None):
                    choice = res.choices[0]
                    message = getattr(choice, "message", None)
                    if isinstance(message, dict):
                        action_text = message.get("content", "")
                    else:
                        action_text = getattr(message, "content", "")

                    # JSONを抽出
                    action_text = action_text.strip()
                    if "```json" in action_text:
                        start = action_text.find("```json") + 7
                        end = action_text.find("```", start)
                        if end > start:
                            action_text = action_text[start:end].strip()
                    elif "```" in action_text:
                        start = action_text.find("```") + 3
                        end = action_text.find("```", start)
                        if end > start:
                            action_text = action_text[start:end].strip()

                    # パース（本番コードと同じメソッド）
                    plan, plan_error, _ = env._parse_plan(action_text)
                    if plan_error:
                        print(f"  ⚠️ プランパースエラー: {plan_error}, fallback to no intervention")
                        plan = {"intervene_now": False}
                else:
                    plan = {"intervene_now": False}
            except Exception as e:
                print(f"  ⚠️ 介入判定LLM呼び出しエラー: {e}, fallback to no intervention")
                plan = {"intervene_now": False}

        intervene_now = plan.get("intervene_now", False)
        print(f"（介入判断：{'介入' if intervene_now else '介入しない'}）")

        if intervene_now:
            print(f"  対象エッジ: {plan.get('edge_to_change', 'N/A')}")
            print(f"  戦略: {plan.get('strategy', 'N/A')}")
            print(f"  対象話者: {plan.get('target_speaker', 'N/A')}")

        # 5. 介入実行または報酬計算
        if not intervene_now:
            # 介入しない場合: タイムペナルティのみ
            reward = -time_penalty
            print(f"（報酬確定：{reward:+.4f} = -time_penalty）")
            continue

        # 6. 介入する場合
        # ロボット発話を生成（本番コードと同じメソッド）
        print(f"\n🤖 ロボット発話生成中...")
        robot_entry = env._render_intervention(plan, simulate=False)
        robot_utterance = robot_entry.get("utterance", "")
        env.logs.append(robot_entry)
        print(f"（ロボット：{robot_utterance}）")

        # evaluation_horizon回の人間発話を生成（1人ずつ）
        print(f"\n（報酬確定まで人間発話 {evaluation_horizon} 回生成...）")
        for _ in range(evaluation_horizon):
            current_persona_eval = env.persona_pool[persona_index % len(env.persona_pool)]
            persona_index += 1

            replies_eval = human_reply(
                env.logs,
                [current_persona_eval],
                topic=env.current_topic,
                topic_trigger=env.current_topic_trigger
            )
            env.logs.extend(replies_eval)

            # 発話を表示
            for reply in replies_eval:
                speaker = reply.get("speaker", "?")
                utterance = reply.get("utterance", "")
                print(f"[{speaker}] {utterance}")

        # 関係性判断
        print(f"\n📊 関係性判断中...")
        participants_after = env._participants(env.logs)
        filtered_logs_after = filter_logs_by_human_count(env.logs, max_history, exclude_robot=True)
        scores_after, _ = env._relation_state(filtered_logs_after, update_state=True)

        for (a, b), v in sorted(scores_after.items()):
            print(f"  relation {a}-{b}: {v:+.2f}")

        metrics_after, _ = env._metrics_state(scores_after, participants_after)
        is_stable_after = metrics_after.get("unstable_triads", 0) == 0

        if is_stable_after:
            print("（関係性判断：安定）")
            print(f"（安定持続するか確認 {terminal_bonus_duration} 回...）")

            # 安定持続確認（1人ずつ）
            stability_maintained = True
            for i in range(terminal_bonus_duration):
                current_persona_check = env.persona_pool[persona_index % len(env.persona_pool)]
                persona_index += 1

                replies_check = human_reply(
                    env.logs,
                    [current_persona_check],
                    topic=env.current_topic,
                    topic_trigger=env.current_topic_trigger
                )
                env.logs.extend(replies_check)

                # 発話を表示
                for reply in replies_check:
                    speaker = reply.get("speaker", "?")
                    utterance = reply.get("utterance", "")
                    print(f"[{speaker}] {utterance}")

                # 関係性判断
                participants_check = env._participants(env.logs)
                filtered_check = filter_logs_by_human_count(env.logs, max_history, exclude_robot=True)
                scores_check, _ = env._relation_state(filtered_check, update_state=True)
                metrics_check, _ = env._metrics_state(scores_check, participants_check)

                if metrics_check.get("unstable_triads", 0) > 0:
                    print("  （関係性判断：不安定）")
                    print("  （持続なし）")
                    stability_maintained = False
                    break
                else:
                    print(f"  安定維持 {i+1}/{terminal_bonus_duration}")

            if stability_maintained:
                print("（安定持続）")
                reward = -time_penalty - intervention_cost + terminal_bonus
                print(f"（報酬確定：{reward:+.4f} = -time_penalty -{intervention_cost:.4f} +{terminal_bonus:.4f}）")
            else:
                reward = -time_penalty - intervention_cost
                print(f"（報酬確定：{reward:+.4f} = -time_penalty -{intervention_cost:.4f}）")
        else:
            print("（関係性判断：不安定）")
            reward = -time_penalty - intervention_cost
            print(f"（報酬確定：{reward:+.4f} = -time_penalty -{intervention_cost:.4f}）")

    print("\n" + "=" * 80)
    print("会話テスト終了")
    print("=" * 80)


if __name__ == "__main__":
    run_conversation_test(max_turns=20)
