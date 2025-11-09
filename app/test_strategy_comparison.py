#!/usr/bin/env python3
"""
戦略比較テストスクリプト

全く同じ初期会話に対して5つの介入戦略を試し、その効果を比較します。
学習環境と同様に、安定状態は自動スキップし、不安定になったときのみ戦略を試します。
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# .envファイルを読み込む
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    load_dotenv(dotenv_path)
except ImportError:
    pass  # dotenv がインストールされていない場合はスキップ

from azure_clients import build_chat_completion_params, get_azure_chat_completion_client
from config import get_config
from env.convo_env import select_target_edge_and_speaker
from humans_ollama import human_reply
from network_metrics import analyze_relations_from_scores
from relation_scorer import RelationScorer
from utils import filter_logs_by_human_count


def _strip_quotes(text: str) -> str:
    """先頭・末尾の引用符や括弧を削る"""
    return text.strip().strip("\"'\u300c\u300d\u300e\u300f()[]（）【】")


def _get_topic_suggestion(cfg, used_topics: List[str], persona_triggers: Dict[str, List[str]], debug: bool = False) -> Tuple[str, Optional[str]]:
    """
    トピックを生成する（ConversationEnv._get_topic_suggestion()と同様）

    Returns:
        (topic, trigger): トピック文字列と選択された地雷（なければNone）
    """
    topic_cfg = getattr(cfg, "topic_manager", None)
    if not topic_cfg or not getattr(topic_cfg, "enable", False):
        return "自由な話題", None

    system = getattr(topic_cfg, "generation_prompt", "")

    # persona_triggersから trigger_examples を生成（ランダムに1つだけ選択）
    selected_trigger = None
    if persona_triggers and "{trigger_examples}" in system:
        all_triggers = []
        for triggers in persona_triggers.values():
            all_triggers.extend(triggers)
        all_triggers = list(set(all_triggers))

        if all_triggers:
            # 使用済み地雷を除外
            available_triggers = [t for t in all_triggers if t not in [topic[1] for topic in used_topics if topic[1]]]

            if not available_triggers:
                # 全て使用済みならリセット
                available_triggers = all_triggers

            selected_trigger = random.choice(available_triggers)
            trigger_examples = selected_trigger
            system = system.replace("{trigger_examples}", trigger_examples)

            if debug:
                print(f"[topic] 選択された地雷: {selected_trigger}")
        else:
            system = system.replace("{trigger_examples}", "様々なテーマ")

    used_str = "\n".join(f"- {t[0]}" for t in used_topics) if used_topics else "(なし)"
    prompt = f"[既に提案した話題]\n{used_str}"

    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="topic")
    max_attempts = getattr(cfg.llm, "max_attempts", 5) or 5
    base_backoff = getattr(cfg.llm, "base_backoff", 0.5) or 0.5

    if client and deployment:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": prompt}]
        for attempt in range(1, max_attempts + 1):
            try:
                params = build_chat_completion_params(deployment, messages, cfg.llm, temperature=1.0)
                res = client.chat.completions.create(**params)
                if res and getattr(res, "choices", None):
                    choice = res.choices[0]
                    message = getattr(choice, "message", None)
                    if isinstance(message, dict):
                        txt = message.get("content", "")
                    else:
                        txt = getattr(message, "content", "")
                    txt = (txt or "").strip()
                    if txt:
                        return txt, selected_trigger
            except Exception as exc:
                if debug:
                    print(f"[topic] attempt {attempt} failed:", exc)
                if attempt < max_attempts:
                    time.sleep(base_backoff * (2 ** (attempt - 1)))

    return "最近の出来事について", selected_trigger


def _generate_robot_utterance_free(
    logs: List[Dict[str, Any]],
    target_edge: str,
    target_speaker: str,
    robot_max_history: int,
    cfg,
    debug: bool = False
) -> str:
    """
    戦略指定なしでロボット発話を生成（free戦略用）

    既存の_render_intervention()と同様だが、戦略指示を省略する
    """
    # 話者名を取得
    persona_pool = sorted(set(log["speaker"] for log in logs if log.get("speaker") and log.get("speaker") != "ロボット"))

    target_char = target_speaker
    edge_to_change = target_edge
    partner_char = next((ch for ch in edge_to_change if ch != target_char), "B")

    try:
        target_idx = "ABC".index(target_char)
        target_name = persona_pool[target_idx] if target_idx < len(persona_pool) else target_char
    except (ValueError, IndexError):
        target_name = target_char

    try:
        partner_idx = "ABC".index(partner_char)
        partner_name = persona_pool[partner_idx] if partner_idx < len(persona_pool) else partner_char
    except (ValueError, IndexError):
        partner_name = partner_char

    # robot_max_history個の人間発話 + その間のロボット発話を取得
    filtered_logs = filter_logs_by_human_count(logs, robot_max_history)
    history_lines = [
        f"[{entry.get('speaker', '?')}] {entry.get('utterance', '').strip()}" for entry in filtered_logs
    ]
    history_text = "\n".join(history_lines) if history_lines else "(履歴なし)"

    # 戦略指示なしのプロンプト
    user_payload = (
        f"あなたは{target_name}さんと{partner_name}さんの関係性を良くするためにロボットの発言を生成します。出力は日本語で一文のみ。話者ラベルや括弧は使わない。\n"
        "会話履歴を参考にしながら、関係性を改善する発話を生成してください。\n\n"
        f"会話履歴:\n{history_text}\n\n"
        "必須条件:\n"
        f"- {target_name} さんと{partner_name} さんの名前を一度だけ入れる。\n"
        "- 直前の話題を自然に引き継ぐ。\n"
        "- 一文のみ (60字前後)。\n"
    )

    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="robot")
    max_attempts = getattr(cfg.llm, "max_attempts", 5) or 5
    base_backoff = getattr(cfg.llm, "base_backoff", 0.5) or 0.5

    fallback_text = f"{target_name}さん、{partner_name}さん、お二人の意見は両方とも大切ですね。"

    if client and deployment:
        messages = [
            {"role": "system", "content": "あなたは関係性を良くするためのロボットの発言生成器として、適切な発話内容を作成します。常に日本語の一文だけを返します。"},
            {"role": "user", "content": user_payload},
        ]
        for attempt in range(1, max_attempts + 1):
            try:
                params = build_chat_completion_params(deployment, messages, cfg.llm)
                res = client.chat.completions.create(**params)
                if res and getattr(res, "choices", None):
                    choice = res.choices[0]
                    message = getattr(choice, "message", None)
                    if isinstance(message, dict):
                        txt = message.get("content", "")
                    else:
                        txt = getattr(message, "content", "")
                    txt = (txt or "").strip()
                    cleaned = _strip_quotes(txt or "")
                    if cleaned:
                        return cleaned
            except Exception as exc:
                if "content_filter" in str(exc) or "ResponsibleAIPolicyViolation" in str(exc):
                    if debug:
                        print(f"[robot_utterance_free] Azure content filter triggered, using fallback immediately")
                    break

                if debug:
                    print(f"[robot_utterance_free] attempt {attempt} failed:", exc)
                if attempt < max_attempts:
                    time.sleep(base_backoff * (2 ** (attempt - 1)))

    return fallback_text


def _generate_robot_utterance_with_strategy(
    logs: List[Dict[str, Any]],
    strategy: str,
    target_edge: str,
    target_speaker: str,
    robot_max_history: int,
    cfg,
    debug: bool = False
) -> str:
    """
    指定された戦略でロボット発話を生成（validate/bridge/plan用）

    ConversationEnv._render_intervention()と同様のロジック
    """
    persona_pool = sorted(set(log["speaker"] for log in logs if log.get("speaker") and log.get("speaker") != "ロボット"))

    target_char = target_speaker
    edge_to_change = target_edge
    partner_char = next((ch for ch in edge_to_change if ch != target_char), "B")

    try:
        target_idx = "ABC".index(target_char)
        target_name = persona_pool[target_idx] if target_idx < len(persona_pool) else target_char
    except (ValueError, IndexError):
        target_name = target_char

    try:
        partner_idx = "ABC".index(partner_char)
        partner_name = persona_pool[partner_idx] if partner_idx < len(persona_pool) else partner_char
    except (ValueError, IndexError):
        partner_name = partner_char

    filtered_logs = filter_logs_by_human_count(logs, robot_max_history)
    history_lines = [
        f"[{entry.get('speaker', '?')}] {entry.get('utterance', '').strip()}" for entry in filtered_logs
    ]
    history_text = "\n".join(history_lines) if history_lines else "(履歴なし)"

    directive_map = {
        "plan": f"{target_name}さんに対して、{partner_name}さんとの関係を改善するために「これからどうするか」という未来志向の視点を提示してください。具体的な次の一歩や小さな行動案を一文で示し、前向きな行動を促してください。",
        "validate": f"{target_name}さんの感情や意見を明確に承認・共感し、その価値を認めてください。{partner_name}さんとの関係改善に向けて、{target_name}さんが理解され、尊重されていると感じられるような一文を述べてください。",
        "bridge": f"{target_name}さんと{partner_name}さんの間に共通点・共通の目標・相互依存性を見出し、両者をつなぐ役割を果たす一文を述べてください。対立や誤解を和らげ、協力的な関係構築を促進してください。",
    }
    directive = directive_map.get(strategy, directive_map["plan"])

    user_payload = (
        f"あなたは{target_name}さんと{partner_name}さんの関係性を良くするためにロボットの発言を生成します。出力は日本語で一文のみ。話者ラベルや括弧は使わない。\n"
        "会話履歴を参考にしながら、以下の指示に従って発話を生成してください。\n\n"
        f"指示: {directive}\n"
        f"会話履歴:\n{history_text}\n\n"
        "必須条件:\n"
        f"- {target_name} さんと{partner_name} さんの名前を一度だけ入れる。\n"
        "- 直前の話題を自然に引き継ぐ。\n"
        "- 一文のみ (60字前後)。\n"
    )

    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="robot")
    max_attempts = getattr(cfg.llm, "max_attempts", 5) or 5
    base_backoff = getattr(cfg.llm, "base_backoff", 0.5) or 0.5

    templates = {
        "plan": f"{target_name}さん、次の一歩として、まず小さなことから始めてみませんか？",
        "validate": f"{target_name}さん、あなたの意見はとても大切です。もっと聞かせてください。",
        "bridge": f"{target_name}さん、{partner_name}さん、お二人には共通の目標があると思います。",
    }
    fallback_text = templates.get(strategy, templates["plan"])

    if client and deployment:
        messages = [
            {"role": "system", "content": "あなたは関係性を良くするためのロボットの発言生成器として、適切な発話内容を作成します。常に日本語の一文だけを返します。"},
            {"role": "user", "content": user_payload},
        ]
        for attempt in range(1, max_attempts + 1):
            try:
                params = build_chat_completion_params(deployment, messages, cfg.llm)
                res = client.chat.completions.create(**params)
                if res and getattr(res, "choices", None):
                    choice = res.choices[0]
                    message = getattr(choice, "message", None)
                    if isinstance(message, dict):
                        txt = message.get("content", "")
                    else:
                        txt = getattr(message, "content", "")
                    txt = (txt or "").strip()
                    cleaned = _strip_quotes(txt or "")
                    if cleaned:
                        return cleaned
            except Exception as exc:
                if "content_filter" in str(exc) or "ResponsibleAIPolicyViolation" in str(exc):
                    if debug:
                        print(f"[robot_utterance_{strategy}] Azure content filter triggered, using fallback immediately")
                    break

                if debug:
                    print(f"[robot_utterance_{strategy}] attempt {attempt} failed:", exc)
                if attempt < max_attempts:
                    time.sleep(base_backoff * (2 ** (attempt - 1)))

    return fallback_text


def test_single_strategy(
    strategy_name: str,
    initial_logs: List[Dict[str, Any]],
    initial_scores: Dict[Tuple[str, str], float],
    target_edge: str,
    target_speaker: str,
    persona_pool: List[str],
    topic: str,
    topic_trigger: Optional[str],
    speaker_aggressiveness: Dict[str, bool],
    cfg,
    debug: bool = False
) -> Dict[str, Any]:
    """
    単一戦略のテストを実行

    Args:
        strategy_name: 戦略名 ("validate", "bridge", "plan", "no_intervention", "free")
        initial_logs: 初期会話ログ（不安定になった時点）
        initial_scores: 初期関係性スコア
        target_edge: 対象エッジ
        target_speaker: 対象話者
        persona_pool: 参加者リスト
        topic: 話題
        topic_trigger: 話題の元になった地雷
        speaker_aggressiveness: 話者の過激度
        cfg: 設定
        debug: デバッグ出力

    Returns:
        テスト結果の辞書
    """
    # 会話ログをコピー
    logs = copy.deepcopy(initial_logs)

    # 関係性スコアラーを新規作成（EMA状態をリセット）
    scorer_kwargs = {
        "backend": cfg.scorer.backend,
        "use_ema": getattr(cfg.scorer, "use_ema", False),
        "decay_factor": float(cfg.scorer.decay_factor),
        "verbose": False,
    }
    scorer = RelationScorer(**scorer_kwargs)

    # 初期関係性を記録
    participants = sorted(set(log["speaker"] for log in logs if log.get("speaker") and log.get("speaker") != "ロボット"))

    # 初期スコアをスコアラーに設定（EMA状態の初期化）
    # これにより、初期状態からの変化を正確に追跡できる
    for edge, score in initial_scores.items():
        scorer.scores[edge] = score

    result = {
        "strategy": strategy_name,
        "target_edge": target_edge,
        "target_speaker": target_speaker,
        "robot_utterance": None,
        "before_intervention": {
            "scores": {f"{a}-{b}": score for (a, b), score in initial_scores.items()},
            "target_edge_score": initial_scores.get(tuple(sorted(target_edge)), 0.0),
        },
        "after_horizon": None,
        "after_bonus": None,
    }

    # ロボット介入
    robot_utterance = None
    if strategy_name != "no_intervention":
        robot_max_history = int(getattr(cfg.env, "robot_max_history", 6))

        if strategy_name == "free":
            robot_utterance = _generate_robot_utterance_free(
                logs, target_edge, target_speaker, robot_max_history, cfg, debug
            )
        else:
            robot_utterance = _generate_robot_utterance_with_strategy(
                logs, strategy_name, target_edge, target_speaker, robot_max_history, cfg, debug
            )

        logs.append({"speaker": "ロボット", "utterance": robot_utterance})
        result["robot_utterance"] = robot_utterance

        if debug:
            print(f"  [{strategy_name}] ロボット発話: {robot_utterance}")

    # evaluation_horizon発話を生成
    evaluation_horizon = int(getattr(cfg.env, "evaluation_horizon", 3))
    start_relation_check_after = 6
    max_history_relation = int(getattr(cfg.env, "max_history_relation", 3))

    # 介入後の人間発話を記録
    human_utterances_after_intervention = []

    for _ in range(evaluation_horizon):
        replies = human_reply(logs, persona_pool, topic=topic, topic_trigger=topic_trigger, num_speakers=1, speaker_aggressiveness=speaker_aggressiveness)
        if replies:
            logs.extend(replies)
            # 人間発話のみを記録
            for reply in replies:
                if reply.get("speaker") != "ロボット":
                    human_utterances_after_intervention.append({
                        "speaker": reply["speaker"],
                        "utterance": reply["utterance"]
                    })
            # 関係性スコアラーを更新
            human_count = sum(1 for log in logs if log.get('speaker') != 'ロボット')
            if human_count >= start_relation_check_after:
                filtered_logs = filter_logs_by_human_count(logs, max_history_relation, exclude_robot=True)
                scorer.get_scores(filtered_logs, participants, return_trace=False, update_state=True)

    # evaluation_horizon後の関係性を評価
    human_count_after = sum(1 for log in logs if log.get('speaker') != 'ロボット')
    if human_count_after >= start_relation_check_after:
        filtered_logs_after = filter_logs_by_human_count(logs, max_history_relation, exclude_robot=True)
        scores_after = scorer.get_scores(filtered_logs_after, participants, return_trace=False, update_state=False)
        metrics_after, _ = analyze_relations_from_scores(scores_after, include_nodes=participants, verbose=False, return_trace=True)

        is_stable_after = metrics_after.get("unstable_triads", 0) == 0
        target_edge_score_after = scores_after.get(tuple(sorted(target_edge)), 0.0)

        result["after_horizon"] = {
            "scores": {f"{a}-{b}": score for (a, b), score in scores_after.items()},
            "unstable_triads": metrics_after.get("unstable_triads", 0),
            "is_stable": is_stable_after,
            "target_edge_score": target_edge_score_after,
            "human_utterances": human_utterances_after_intervention,
        }

        if debug:
            print(f"  [{strategy_name}] evaluation_horizon後: 不安定トライアド={metrics_after.get('unstable_triads', 0)}, 安定={is_stable_after}")

        # 安定になった場合、terminal_bonus_duration発話を追加
        if is_stable_after:
            terminal_bonus_duration = int(getattr(cfg.env, "terminal_bonus_duration", 2))

            # terminal_bonus_duration中の人間発話を記録
            human_utterances_during_bonus = []

            for _ in range(terminal_bonus_duration):
                replies = human_reply(logs, persona_pool, topic=topic, topic_trigger=topic_trigger, num_speakers=1, speaker_aggressiveness=speaker_aggressiveness)
                if replies:
                    logs.extend(replies)
                    # 人間発話のみを記録
                    for reply in replies:
                        if reply.get("speaker") != "ロボット":
                            human_utterances_during_bonus.append({
                                "speaker": reply["speaker"],
                                "utterance": reply["utterance"]
                            })
                    human_count = sum(1 for log in logs if log.get('speaker') != 'ロボット')
                    if human_count >= start_relation_check_after:
                        filtered_logs = filter_logs_by_human_count(logs, max_history_relation, exclude_robot=True)
                        scorer.get_scores(filtered_logs, participants, return_trace=False, update_state=True)

            # terminal_bonus_duration後の関係性を評価
            human_count_bonus = sum(1 for log in logs if log.get('speaker') != 'ロボット')
            if human_count_bonus >= start_relation_check_after:
                filtered_logs_bonus = filter_logs_by_human_count(logs, max_history_relation, exclude_robot=True)
                scores_bonus = scorer.get_scores(filtered_logs_bonus, participants, return_trace=False, update_state=False)
                metrics_bonus, _ = analyze_relations_from_scores(scores_bonus, include_nodes=participants, verbose=False, return_trace=True)

                is_stable_bonus = metrics_bonus.get("unstable_triads", 0) == 0
                target_edge_score_bonus = scores_bonus.get(tuple(sorted(target_edge)), 0.0)

                result["after_bonus"] = {
                    "scores": {f"{a}-{b}": score for (a, b), score in scores_bonus.items()},
                    "unstable_triads": metrics_bonus.get("unstable_triads", 0),
                    "is_stable": is_stable_bonus,
                    "target_edge_score": target_edge_score_bonus,
                    "human_utterances": human_utterances_during_bonus,
                }

                if debug:
                    print(f"  [{strategy_name}] terminal_bonus後: 不安定トライアド={metrics_bonus.get('unstable_triads', 0)}, 安定={is_stable_bonus}")

    return result


def test_single_session(
    session_id: int,
    cfg,
    persona_pool: List[str],
    persona_triggers: Dict[str, List[str]],
    used_topics: List[Tuple[str, Optional[str]]],
    max_auto_skip: int = 10,
    repeats: int = 5,
    debug: bool = False
) -> Optional[Dict[str, Any]]:
    """
    単一セッションのテストを実行

    Args:
        session_id: セッションID
        cfg: 設定
        persona_pool: 参加者リスト
        persona_triggers: 地雷辞書
        used_topics: 使用済みトピックリスト
        max_auto_skip: 安定時の最大スキップ発話数
        repeats: 各戦略を何回ずつ実行するか
        debug: デバッグ出力

    Returns:
        テスト結果の辞書（スキップされた場合はNone）
    """
    # トピック生成
    topic, topic_trigger = _get_topic_suggestion(cfg, used_topics, persona_triggers, debug)
    used_topics.append((topic, topic_trigger))

    if debug:
        print(f"\n=== セッション {session_id} ===")
        print(f"話題: {topic}")
        if topic_trigger:
            print(f"地雷: {topic_trigger}")

    # 話者の過激度を設定
    speaker_aggressiveness = {}
    for speaker in persona_pool:
        speaker_aggressiveness[speaker] = random.random() < 0.5

    # 初期3発話を生成
    logs = []
    for _ in range(3):
        replies = human_reply(logs, persona_pool, topic=topic, topic_trigger=topic_trigger, num_speakers=1, speaker_aggressiveness=speaker_aggressiveness)
        if replies:
            logs.extend(replies)

    if debug:
        print("\n【初期会話】")
        for log in logs:
            print(f"{log['speaker']}: {log['utterance']}")

    # 初期関係性を評価
    scorer_kwargs = {
        "backend": cfg.scorer.backend,
        "use_ema": getattr(cfg.scorer, "use_ema", False),
        "decay_factor": float(cfg.scorer.decay_factor),
        "verbose": False,
    }
    scorer = RelationScorer(**scorer_kwargs)

    participants = sorted(set(log["speaker"] for log in logs if log.get("speaker") and log.get("speaker") != "ロボット"))
    start_relation_check_after = int(getattr(cfg.env, "start_relation_check_after_utterances", 3))
    max_history_relation = int(getattr(cfg.env, "max_history_relation", 3))

    human_count = sum(1 for log in logs if log.get('speaker') != 'ロボット')

    # 安定状態の自動スキップ
    auto_skip_count = 0

    while auto_skip_count < max_auto_skip:
        # 関係性を評価
        if human_count >= start_relation_check_after:
            filtered_logs = filter_logs_by_human_count(logs, max_history_relation, exclude_robot=True)
            scores = scorer.get_scores(filtered_logs, participants, return_trace=False, update_state=True)
            metrics, _ = analyze_relations_from_scores(scores, include_nodes=participants, verbose=False, return_trace=True)
            unstable_triads = metrics.get("unstable_triads", 0)

            if debug:
                print(f"\n関係性評価（{auto_skip_count+1}回目）: 不安定トライアド={unstable_triads}")

            # 不安定なら抜ける
            if unstable_triads > 0:
                if debug:
                    print("→ 不安定状態を検出、戦略テストへ進む")
                break

        # 安定なので人間1発話を追加
        auto_skip_count += 1
        if debug:
            print(f"→ 安定状態、人間発話を追加生成（{auto_skip_count}/{max_auto_skip}）")

        replies = human_reply(logs, persona_pool, topic=topic, topic_trigger=topic_trigger, num_speakers=1, speaker_aggressiveness=speaker_aggressiveness)
        if replies:
            logs.extend(replies)
            human_count += 1
        else:
            if debug:
                print("→ 発話生成失敗、スキップ終了")
            break

    # max_auto_skip回試しても不安定にならなかった場合はスキップ
    if auto_skip_count >= max_auto_skip:
        if debug:
            print(f"→ {max_auto_skip}回試しても不安定にならず、この話題をスキップ")
        return None

    # 不安定状態になったので、ここから5戦略をテスト
    # この時点の会話ログと関係性スコアを保存
    unstable_logs = copy.deepcopy(logs)

    # 最終的な初期関係性スコアを取得
    filtered_logs_initial = filter_logs_by_human_count(unstable_logs, max_history_relation, exclude_robot=True)
    initial_scores = scorer.get_scores(filtered_logs_initial, participants, return_trace=False, update_state=False)
    initial_metrics, _ = analyze_relations_from_scores(initial_scores, include_nodes=participants, verbose=False, return_trace=True)

    # 対象エッジと対象話者を選択
    edges = initial_metrics.get("edges", {})
    if edges:
        target_edge, target_speaker = select_target_edge_and_speaker(edges, debug=debug)
    else:
        target_edge = "AB"
        target_speaker = "A"

    if debug:
        print(f"\n【初期関係性（不安定状態）】")
        print(f"関係性スコア: {', '.join(f'{a}-{b}={score:+.2f}' for (a,b), score in initial_scores.items())}")
        print(f"不安定トライアド数: {initial_metrics.get('unstable_triads', 0)}")
        print(f"対象エッジ: {target_edge}")
        print(f"対象話者: {target_speaker}")

    # 5戦略を各repeats回ずつテスト
    strategies = ["validate", "bridge", "plan", "no_intervention", "free"]
    strategy_results = {}

    for strategy in strategies:
        if debug:
            print(f"\n--- 戦略: {strategy} ---")

        # 各戦略をrepeats回実行
        results_for_strategy = []
        for repeat in range(repeats):
            if debug:
                print(f"  試行 {repeat+1}/{repeats}")

            result = test_single_strategy(
                strategy_name=strategy,
                initial_logs=unstable_logs,
                initial_scores=initial_scores,
                target_edge=target_edge,
                target_speaker=target_speaker,
                persona_pool=persona_pool,
                topic=topic,
                topic_trigger=topic_trigger,
                speaker_aggressiveness=speaker_aggressiveness,
                cfg=cfg,
                debug=debug
            )

            results_for_strategy.append(result)

        strategy_results[strategy] = results_for_strategy

    # セッション結果を構築
    session_result = {
        "session_id": session_id,
        "topic": topic,
        "topic_trigger": topic_trigger,
        "initial_conversation": [{"speaker": log["speaker"], "utterance": log["utterance"]} for log in unstable_logs],
        "auto_skip_count": auto_skip_count,
        "initial_relation": {
            "scores": {f"{a}-{b}": score for (a, b), score in initial_scores.items()},
            "unstable_triads": initial_metrics.get("unstable_triads", 0),
            "is_stable": False,
            "target_edge": target_edge,
            "target_speaker": target_speaker,
        },
        "strategies": strategy_results,
    }

    return session_result


def calculate_statistics(session_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    全セッションの統計を計算

    Args:
        session_results: セッション結果のリスト

    Returns:
        統計データの辞書
    """
    strategies = ["validate", "bridge", "plan", "no_intervention", "free"]
    strategy_stats = {strategy: defaultdict(list) for strategy in strategies}

    for session in session_results:
        for strategy, results_list in session["strategies"].items():
            # 各戦略の複数回の試行を展開
            for result in results_list:
                # 対象エッジの変化量を計算
                before_score = result["before_intervention"]["target_edge_score"]

                if result["after_horizon"]:
                    after_score = result["after_horizon"]["target_edge_score"]
                    edge_change = after_score - before_score

                    strategy_stats[strategy]["edge_changes"].append(edge_change)
                    strategy_stats[strategy]["stable_after_horizon"].append(result["after_horizon"]["is_stable"])

                    # 全エッジの平均変化量を計算
                    before_scores = result["before_intervention"]["scores"]
                    after_scores = result["after_horizon"]["scores"]
                    all_edge_changes = []
                    for edge_key in before_scores.keys():
                        if edge_key in after_scores:
                            all_edge_changes.append(after_scores[edge_key] - before_scores[edge_key])
                    if all_edge_changes:
                        strategy_stats[strategy]["all_edges_changes"].append(sum(all_edge_changes) / len(all_edge_changes))

                    # 不安定トライアド減少量
                    before_unstable = session["initial_relation"]["unstable_triads"]
                    after_unstable = result["after_horizon"]["unstable_triads"]
                    strategy_stats[strategy]["unstable_triads_reduction"].append(before_unstable - after_unstable)

                    # terminal_bonusの成功
                    if result["after_bonus"]:
                        strategy_stats[strategy]["stable_after_bonus"].append(result["after_bonus"]["is_stable"])

    # 統計を計算
    stats = {}
    for strategy in strategies:
        data = strategy_stats[strategy]

        total_tests = len(data["edge_changes"])
        stable_count = sum(data["stable_after_horizon"]) if data["stable_after_horizon"] else 0
        stable_rate = stable_count / total_tests if total_tests > 0 else 0.0

        terminal_bonus_count = sum(data["stable_after_bonus"]) if data["stable_after_bonus"] else 0
        terminal_bonus_rate = terminal_bonus_count / stable_count if stable_count > 0 else 0.0

        edge_changes = data["edge_changes"]
        edge_change_mean = sum(edge_changes) / len(edge_changes) if edge_changes else 0.0
        edge_change_std = 0.0
        if len(edge_changes) > 1:
            mean = edge_change_mean
            variance = sum((x - mean) ** 2 for x in edge_changes) / len(edge_changes)
            edge_change_std = variance ** 0.5

        edge_improvement_count = sum(1 for x in edge_changes if x > 0)
        edge_improvement_rate = edge_improvement_count / len(edge_changes) if edge_changes else 0.0

        all_edges_changes = data["all_edges_changes"]
        all_edges_change_mean = sum(all_edges_changes) / len(all_edges_changes) if all_edges_changes else 0.0

        unstable_reductions = data["unstable_triads_reduction"]
        unstable_reduction_mean = sum(unstable_reductions) / len(unstable_reductions) if unstable_reductions else 0.0

        stats[strategy] = {
            "total_tests": total_tests,
            "stable_count": stable_count,
            "stable_rate": stable_rate,
            "terminal_bonus_count": terminal_bonus_count,
            "terminal_bonus_rate": terminal_bonus_rate,
            "target_edge_change_mean": edge_change_mean,
            "target_edge_change_std": edge_change_std,
            "target_edge_improvement_rate": edge_improvement_rate,
            "all_edges_change_mean": all_edges_change_mean,
            "unstable_triads_reduction_mean": unstable_reduction_mean,
        }

    # 比較データ
    best_stable_rate = max((s, stats[s]["stable_rate"]) for s in strategies)
    best_edge_improvement = max((s, stats[s]["target_edge_change_mean"]) for s in strategies)
    best_terminal_bonus_rate = max((s, stats[s]["terminal_bonus_rate"]) for s in strategies if stats[s]["stable_count"] > 0)

    return {
        "strategy_stats": stats,
        "comparison": {
            "best_stable_rate": f"{best_stable_rate[0]} ({best_stable_rate[1]:.3f})",
            "best_edge_improvement": f"{best_edge_improvement[0]} ({best_edge_improvement[1]:+.3f})",
            "best_terminal_bonus_rate": f"{best_terminal_bonus_rate[0]} ({best_terminal_bonus_rate[1]:.3f})",
        }
    }


def save_logs(
    session_results: List[Dict[str, Any]],
    statistics: Dict[str, Any],
    total_sessions_attempted: int,
    sessions_skipped: int,
    output_dir: Path,
    timestamp: str,
    repeats: int = 5
):
    """
    ログを保存（JSONL, テキスト, 統計JSON）

    Args:
        session_results: セッション結果のリスト
        statistics: 統計データ
        total_sessions_attempted: 試行セッション数
        sessions_skipped: スキップされたセッション数
        output_dir: 出力ディレクトリ
        timestamp: タイムスタンプ
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSONL出力
    jsonl_path = output_dir / f"strategy_comparison_{timestamp}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for session in session_results:
            f.write(json.dumps(session, ensure_ascii=False) + "\n")
    print(f"セッションデータを保存: {jsonl_path}")

    # 統計JSON出力
    stats_data = {
        "summary": {
            "total_sessions_attempted": total_sessions_attempted,
            "sessions_with_instability": len(session_results),
            "sessions_skipped_stable": sessions_skipped,
            "strategies_tested_per_session": 5,
            "repeats_per_strategy": repeats,
        },
        **statistics,
    }
    stats_path = output_dir / f"strategy_comparison_{timestamp}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"統計データを保存: {stats_path}")

    # テキスト出力
    txt_path = output_dir / f"strategy_comparison_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        # セッション詳細
        for session in session_results:
            f.write(f"{'='*60}\n")
            f.write(f"セッション {session['session_id']}\n")
            f.write(f"{'='*60}\n")
            f.write(f"話題: {session['topic']}\n")
            if session.get('topic_trigger'):
                f.write(f"地雷: {session['topic_trigger']}\n")
            f.write(f"\n【初期会話】\n")
            for log in session["initial_conversation"]:
                f.write(f"{log['speaker']}: {log['utterance']}\n")

            f.write(f"\n【初期関係性（不安定状態）】\n")
            rel = session["initial_relation"]
            f.write(f"関係性スコア: {', '.join(f'{k}={v:+.2f}' for k, v in rel['scores'].items())}\n")
            f.write(f"不安定トライアド数: {rel['unstable_triads']}\n")
            f.write(f"対象エッジ: {rel['target_edge']}\n")
            f.write(f"対象話者: {rel['target_speaker']}\n")

            # 戦略ごとの結果（複数回の試行の平均）
            for strategy, results_list in session["strategies"].items():
                f.write(f"\n{'-'*40}\n")
                f.write(f"戦略: {strategy} ({len(results_list)}回試行の平均)\n")
                f.write(f"{'-'*40}\n")

                # 各試行の結果を集計
                edge_changes = []
                stable_count = 0
                terminal_bonus_count = 0

                for result in results_list:
                    if result["after_horizon"]:
                        edge_change = result["after_horizon"]["target_edge_score"] - result["before_intervention"]["target_edge_score"]
                        edge_changes.append(edge_change)
                        if result["after_horizon"]["is_stable"]:
                            stable_count += 1
                            if result["after_bonus"] and result["after_bonus"]["is_stable"]:
                                terminal_bonus_count += 1

                avg_edge_change = sum(edge_changes) / len(edge_changes) if edge_changes else 0.0
                stable_rate = stable_count / len(results_list) if results_list else 0.0
                terminal_bonus_rate = terminal_bonus_count / stable_count if stable_count > 0 else 0.0

                f.write(f"平均対象エッジ変化量: {avg_edge_change:+.3f}\n")
                f.write(f"安定成功率: {stable_count}/{len(results_list)} ({stable_rate*100:.1f}%)\n")
                if stable_count > 0:
                    f.write(f"terminal_bonus成功率: {terminal_bonus_count}/{stable_count} ({terminal_bonus_rate*100:.1f}%)\n")
                else:
                    f.write(f"terminal_bonus成功率: N/A (安定成功なし)\n")

                # 人間発話の例（最初の試行のみ）
                if results_list and results_list[0]["after_horizon"] and results_list[0]["after_horizon"].get("human_utterances"):
                    f.write(f"\n人間発話例（1回目の試行）:\n")
                    for utterance in results_list[0]["after_horizon"]["human_utterances"]:
                        f.write(f"  {utterance['speaker']}: {utterance['utterance']}\n")

            f.write(f"\n")

        # 統計サマリー
        f.write(f"\n{'='*60}\n")
        f.write(f"戦略比較統計\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"テスト概要:\n")
        f.write(f"- 不安定になったセッション数: {total_sessions_attempted}\n")
        f.write(f"- スキップされたセッション数: {sessions_skipped}\n")
        f.write(f"- 戦略ごとの試行回数: {repeats}\n")
        f.write(f"- 総試行数: {total_sessions_attempted * 5 * repeats} （{total_sessions_attempted}セッション × 5戦略 × {repeats}回）\n")
        f.write(f"\n")

        for strategy, stats in statistics["strategy_stats"].items():
            f.write(f"{'-'*40}\n")
            f.write(f"戦略: {strategy}\n")
            f.write(f"{'-'*40}\n")
            f.write(f"テスト回数: {stats['total_tests']}\n")
            f.write(f"安定成功率: {stats['stable_count']}/{stats['total_tests']} ({stats['stable_rate']*100:.1f}%)\n")
            if stats['stable_count'] > 0:
                f.write(f"terminal_bonus成功率: {stats['terminal_bonus_count']}/{stats['stable_count']} ({stats['terminal_bonus_rate']*100:.1f}%)\n")
            else:
                f.write(f"terminal_bonus成功率: N/A (安定成功なし)\n")
            f.write(f"対象エッジ変化量: 平均={stats['target_edge_change_mean']:+.3f}, 標準偏差={stats['target_edge_change_std']:.3f}\n")
            f.write(f"対象エッジ改善率: {stats['target_edge_improvement_rate']*100:.1f}%\n")
            f.write(f"全エッジ平均変化: {stats['all_edges_change_mean']:+.3f}\n")
            f.write(f"不安定トライアド減少: 平均={stats['unstable_triads_reduction_mean']:.2f}\n")
            f.write(f"\n")

        f.write(f"【比較】\n")
        f.write(f"最高安定成功率: {statistics['comparison']['best_stable_rate']}\n")
        f.write(f"最高エッジ改善: {statistics['comparison']['best_edge_improvement']}\n")
        f.write(f"最高terminal_bonus成功率: {statistics['comparison']['best_terminal_bonus_rate']}\n")

    print(f"テキストレポートを保存: {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="戦略比較テストスクリプト")
    parser.add_argument("--num-sessions", type=int, default=10, help="テストセッション数")
    parser.add_argument("--max-auto-skip", type=int, default=10, help="安定時の最大スキップ発話数")
    parser.add_argument("--repeats", type=int, default=5, help="各戦略を何回ずつ実行するか")
    parser.add_argument("--evaluation-horizon", type=int, default=None, help="評価タイミング（人間発話数）")
    parser.add_argument("--terminal-bonus-duration", type=int, default=None, help="安定確認発話数")
    parser.add_argument("--quiet", action="store_true", help="詳細出力を抑制（統計のみ表示）")
    parser.add_argument("--output-dir", type=str, default=None, help="出力ディレクトリ")
    args = parser.parse_args()

    cfg = get_config()

    # 設定のオーバーライド
    if args.evaluation_horizon is not None:
        cfg.env.evaluation_horizon = args.evaluation_horizon
    if args.terminal_bonus_duration is not None:
        cfg.env.terminal_bonus_duration = args.terminal_bonus_duration

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "logs"

    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ペルソナ設定
    personas_cfg = getattr(cfg.env, "personas", None)
    if personas_cfg and isinstance(personas_cfg, dict):
        persona_pool = sorted(personas_cfg.keys())
        persona_triggers = {
            name: info.get("triggers", []) if isinstance(info, dict) else []
            for name, info in personas_cfg.items()
        }
    else:
        raise ValueError("config.env.personas が Dict 形式で定義されていません")

    print(f"戦略比較テスト開始")
    print(f"セッション数: {args.num_sessions}")
    print(f"max_auto_skip: {args.max_auto_skip}")
    print(f"戦略ごとの試行回数: {args.repeats}")
    print(f"evaluation_horizon: {cfg.env.evaluation_horizon}")
    print(f"terminal_bonus_duration: {cfg.env.terminal_bonus_duration}")
    print(f"参加者: {', '.join(persona_pool)}")
    print()

    # テスト実行
    session_results = []
    used_topics = []
    sessions_skipped = 0
    debug = not args.quiet

    for i in range(1, args.num_sessions + 1):
        result = test_single_session(
            session_id=i,
            cfg=cfg,
            persona_pool=persona_pool,
            persona_triggers=persona_triggers,
            used_topics=used_topics,
            max_auto_skip=args.max_auto_skip,
            repeats=args.repeats,
            debug=debug
        )

        if result is None:
            sessions_skipped += 1
            if debug:
                print(f"セッション {i}: スキップ（ずっと安定）")
        else:
            session_results.append(result)
            if not debug:
                print(f"セッション {i}: 完了")

    print(f"\n全セッション完了")
    print(f"- 不安定になったセッション: {len(session_results)}")
    print(f"- スキップされたセッション: {sessions_skipped}")

    if not session_results:
        print("\n警告: テストデータがありません（全セッションがスキップされました）")
        return

    # 統計計算
    print(f"\n統計を計算中...")
    statistics = calculate_statistics(session_results)

    # ログ保存
    print(f"\nログを保存中...")
    save_logs(
        session_results=session_results,
        statistics=statistics,
        total_sessions_attempted=len(session_results),
        sessions_skipped=sessions_skipped,
        output_dir=output_dir,
        timestamp=timestamp,
        repeats=args.repeats
    )

    # 統計表示
    print(f"\n{'='*60}")
    print(f"戦略比較統計")
    print(f"{'='*60}\n")

    for strategy, stats in statistics["strategy_stats"].items():
        print(f"{'-'*40}")
        print(f"戦略: {strategy}")
        print(f"{'-'*40}")
        print(f"テスト回数: {stats['total_tests']}")
        print(f"安定成功率: {stats['stable_count']}/{stats['total_tests']} ({stats['stable_rate']*100:.1f}%)")
        if stats['stable_count'] > 0:
            print(f"terminal_bonus成功率: {stats['terminal_bonus_count']}/{stats['stable_count']} ({stats['terminal_bonus_rate']*100:.1f}%)")
        else:
            print(f"terminal_bonus成功率: N/A (安定成功なし)")
        print(f"対象エッジ変化量: 平均={stats['target_edge_change_mean']:+.3f}, 標準偏差={stats['target_edge_change_std']:.3f}")
        print(f"対象エッジ改善率: {stats['target_edge_improvement_rate']*100:.1f}%")
        print(f"全エッジ平均変化: {stats['all_edges_change_mean']:+.3f}")
        print(f"不安定トライアド減少: 平均={stats['unstable_triads_reduction_mean']:.2f}")
        print()

    print(f"【比較】")
    print(f"最高安定成功率: {statistics['comparison']['best_stable_rate']}")
    print(f"最高エッジ改善: {statistics['comparison']['best_edge_improvement']}")
    print(f"最高terminal_bonus成功率: {statistics['comparison']['best_terminal_bonus_rate']}")


if __name__ == "__main__":
    main()
