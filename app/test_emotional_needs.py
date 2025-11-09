#!/usr/bin/env python3
"""
GPT-5による戦略推定テスト

各ペルソナに「隠れた感情ニーズ」を設定し、GPT-5が会話から適切な介入戦略を推定できるかをテストします。
（ニーズ推定は行わず、会話から直接戦略を推定します）
"""

from __future__ import annotations

import argparse
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
    pass

from azure_clients import build_chat_completion_params, get_azure_chat_completion_client
from config import get_config

# 感情ニーズの定義
EMOTIONAL_NEEDS = ["recognition", "mediation", "solution", "independence"]

# ニーズと戦略のマッピング
NEED_TO_STRATEGY = {
    "recognition": "validate",
    "mediation": "bridge",
    "solution": "plan",
    "independence": "no_intervention"
}

# 戦略から数字へのマッピング
STRATEGY_TO_NUM = {
    "validate": 1,
    "bridge": 2,
    "plan": 3,
    "no_intervention": 4
}

# ニーズの説明
NEED_DESCRIPTIONS = {
    "recognition": "自分の意見や感情を認めてほしい",
    "mediation": "仲裁をしてほしい、誰かに仲を取り持ってほしい",
    "solution": "具体的な解決策がほしい",
    "independence": "放っておいてほしい、自分で解決したい"
}


def _get_persona_triggers(cfg) -> Dict[str, List[str]]:
    """configから各ペルソナの地雷リストを取得"""
    personas_cfg = getattr(cfg.env, "personas", None)
    persona_triggers = {}

    if personas_cfg and isinstance(personas_cfg, dict):
        for persona_name in ["A", "B", "C"]:
            persona_info = personas_cfg.get(persona_name, {})
            if isinstance(persona_info, dict):
                triggers = persona_info.get("triggers", []) or []
                persona_triggers[persona_name] = [t for t in triggers if t and t.strip()]

    return persona_triggers


def _select_trigger_and_generate_topic(
    persona_triggers: Dict[str, List[str]],
    used_triggers: List[str],
    cfg,
    debug: bool = False
) -> Tuple[str, Optional[str]]:
    """
    地雷を1つ選び、それをもとに話題を生成

    Returns:
        (topic, selected_trigger)
    """
    # 全ペルソナの地雷を平坦化
    all_triggers = []
    for triggers in persona_triggers.values():
        all_triggers.extend(triggers)
    all_triggers = list(set(all_triggers))

    if not all_triggers:
        return "自由な話題", None

    # 未使用の地雷を取得
    available_triggers = [t for t in all_triggers if t not in used_triggers]

    # 全部使い切ったらリセット
    if not available_triggers:
        used_triggers.clear()
        available_triggers = all_triggers

    # ランダムに1つ選択
    selected_trigger = random.choice(available_triggers)
    used_triggers.append(selected_trigger)

    if debug:
        print(f"  [topic] 選択された地雷: {selected_trigger}")
        print(f"  [topic] 地雷を持つペルソナ: {[p for p, ts in persona_triggers.items() if selected_trigger in ts]}")

    # 話題を生成（Azure OpenAIで）
    topic = _generate_topic_from_trigger(selected_trigger, cfg, debug)

    return topic, selected_trigger


def _generate_topic_from_trigger(trigger: str, cfg, debug: bool = False) -> str:
    """地雷をもとに話題を生成"""
    system_prompt = f"""あなたは会話のテーマを提案する専門家です。
提案されたキーワード「{trigger}」に関連する、短い会話テーマを1つ提案してください。

【出力形式】
- 一文で簡潔に（10-20文字程度）
- 会話のきっかけになるような話題
- 例：「最近の健康管理について」「週末の旅行計画」"""

    user_prompt = f"キーワード「{trigger}」に関連する会話テーマを1つ提案してください。"

    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="topic")

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
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
                    return txt
        except Exception as exc:
            if debug:
                print(f"[generate_topic] attempt {attempt+1} failed: {exc}")
            if attempt < max_attempts - 1:
                time.sleep(0.5)

    # フォールバック
    return f"{trigger}について"


def _generate_human_utterance_with_need(
    speaker: str,
    emotional_need: str,
    conversation_history: List[Dict[str, str]],
    topic: str,
    topic_trigger: Optional[str],
    persona_triggers: Dict[str, List[str]],
    cfg,
    debug: bool = False
) -> str:
    """
    感情ニーズと地雷システムを持つ人間の発話を生成（学習環境の仕様に準拠）

    Args:
        speaker: 話者名（A, B, C）
        emotional_need: 隠れた感情ニーズ
        conversation_history: 会話履歴
        topic: 話題
        topic_trigger: トピックの元になった地雷
        persona_triggers: 各ペルソナの地雷リスト
        cfg: 設定
        debug: デバッグ出力
    """
    # 会話履歴をフォーマット
    history_text = "\n".join([f"[{log['speaker']}] {log['utterance']}" for log in conversation_history]) if conversation_history else "(まだ発話なし)"

    # 基本プロンプト
    lines = [
        f"あなたは会話参加者の一人「{speaker}」です。実在の人のように、状況に応じて態度や言い方が微妙に変わります。\n",
        f"会話は A/B/C で進みます。あなたは {speaker} として、会話履歴に続けて、1–2文で自然に応答してください。\n",
        f"分析・箇条書き・メタ説明・ラベル出力はしません。{speaker} の発話だけを出力します。\n",
    ]

    if topic:
        lines.append(f"会話のテーマ：{topic}\n")

    lines.append("会話の開始時点では、他の参加者がまだ発言していない場合もあります。その場合は話題に沿って発言してください\n")

    # 隠れた感情ニーズ（表には出さない）
    need_descriptions_internal = {
        "recognition": "あなたは内心、自分の意見や感情を認めてほしいと思っています。相手が共感・承認してくれると嬉しく感じます。",
        "mediation": "あなたは内心、仲裁を求めており、誰かに仲を取り持ってほしいと思っています。",
        "solution": "あなたは内心、具体的な解決策がほしいと思っています。抽象的な話よりも、実践的なアプローチを求めています。",
        "independence": "あなたは内心、他人に介入されず、放っておいてほしいと思っています。自分のペースで、自分の力で問題を解決したいと考えています。"
    }
    lines.append(f"\n{need_descriptions_internal[emotional_need]}\n")

    # 地雷システム（学習環境のbuild_human_promptと同じロジック）
    triggers = persona_triggers.get(speaker, [])

    if triggers and topic_trigger:
        # 過激度をランダムに決定
        is_aggressive = random.random() < 0.5

        # 他の参加者を取得
        other_speakers = [s for s in persona_triggers.keys() if s != speaker]

        # topic_triggerが自分の地雷に含まれているか
        if topic_trigger in triggers:
            # 誰が同じ地雷を持っているか
            who_has_same_trigger = [s for s in persona_triggers.keys()
                                     if topic_trigger in persona_triggers.get(s, [])]

            if len(who_has_same_trigger) == 3:  # 3人全員が持っている
                other_angry = [s for s in who_has_same_trigger if s != speaker]
                angry_names = "、".join(other_angry)
                if is_aggressive:
                    lines.append(
                        f"最初はあなた（{speaker}）も{angry_names}も全員が不機嫌な状態です。"
                        f"全員がイライラしているため、お互いに攻撃的な言葉遣いで八つ当たりし、対立します。\n"
                    )
                else:
                    lines.append(
                        f"最初はあなた（{speaker}）も{angry_names}も全員が少し不機嫌な状態です。"
                        f"イライラしていますが、話し合いの余地はあります。少し不満を含んだ言い方になることがあります。\n"
                    )
            elif len(who_has_same_trigger) == 2:  # 2人だけが持っている
                other_angry = [s for s in who_has_same_trigger if s != speaker]
                angry_name = other_angry[0]

                calm_person = [s for s in persona_triggers.keys() if s not in who_has_same_trigger]
                calm_name = calm_person[0] if calm_person else "他の参加者"

                if is_aggressive:
                    lines.append(
                        f"最初はあなた（{speaker}）も{angry_name}も不機嫌な状態です。"
                        f"{angry_name}に対して攻撃的な言葉遣いで八つ当たりし、対立します。{calm_name}には穏やかに接します。\n"
                    )
                else:
                    lines.append(
                        f"最初はあなた（{speaker}）も{angry_name}も少し不機嫌な状態です。"
                        f"{angry_name}に対して少し不満を含んだ言い方になることがありますが、話し合いはできます。{calm_name}には穏やかに接します。\n"
                    )
            else:
                # 自分だけが持っている地雷の場合
                other_names = "、".join(other_speakers)
                if is_aggressive:
                    lines.append(
                        f"最初はかなり不機嫌で、{other_names}など他の参加者に対して攻撃的な言葉遣いで文句を言ったり八つ当たりし、対立します。\n"
                    )
                else:
                    lines.append(
                        f"最初は少し不機嫌で、{other_names}など他の参加者に対して少し不満を含んだ言い方になることがあります。\n"
                    )

    system_prompt = "".join(lines)
    user_prompt = f"""以下の会話を踏まえて、{speaker}として次の発話を生成してください。

会話履歴:
{history_text}

{speaker}の発話:"""

    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="human")
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            params = build_chat_completion_params(deployment, messages, cfg.llm, temperature=0.8)
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
                    return txt
        except Exception as exc:
            if debug:
                print(f"[generate_utterance] attempt {attempt+1} failed: {exc}")
            if attempt < max_attempts - 1:
                time.sleep(0.5)

    return f"そうですね、{topic}について考えさせられますね。"


def _calculate_relation_scores(conversation: List[Dict[str, str]], cfg, debug: bool = False) -> Dict[str, float]:
    """
    relation_scorerを使って関係性スコアを計算

    Returns:
        {"AB": score, "BC": score, "CA": score}
    """
    from relation_scorer import RelationScorer

    # config.scorer.backendを取得
    scorer_backend = getattr(cfg.scorer, "backend", "azure") if hasattr(cfg, "scorer") else "azure"
    use_ema = getattr(cfg.scorer, "use_ema", False) if hasattr(cfg, "scorer") else False

    scorer = RelationScorer(backend=scorer_backend, use_ema=use_ema)
    speakers = ["A", "B", "C"]

    # get_scoresは全ペアのスコアを返す (タプルキー)
    pair_scores = scorer.get_scores(conversation, speakers, update_state=False)

    # タプルキーを文字列キーに変換
    scores = {}
    for (s1, s2), score in pair_scores.items():
        edge = f"{s1}{s2}"
        scores[edge] = score

        if debug:
            print(f"  [relation] {edge}: {score:+.2f}")

    return scores


def _select_target_edge_and_speaker(scores: Dict[str, float], debug: bool = False) -> Tuple[str, str]:
    """
    最も関係性が悪いエッジと、そのエッジに含まれるランダムな話者を選択

    Returns:
        (target_edge, target_speaker)
    """
    # 最も関係性が悪いエッジを選択（負の値で最も小さいもの）
    target_edge = min(scores.items(), key=lambda kv: kv[1])[0]

    # そのエッジに含まれる話者をランダムに選択
    speakers_in_edge = list(target_edge)
    target_speaker = random.choice(speakers_in_edge)

    if debug:
        print(f"  [target] edge: {target_edge}, speaker: {target_speaker}")

    return target_edge, target_speaker


def select_strategy_from_conversation(
    conversation: List[Dict[str, str]],
    cfg,
    debug: bool = False
) -> Tuple[str, str, str, Dict[str, float]]:
    """
    学習環境と同じプロンプトで、GPT-5に介入戦略を選択させる

    Returns:
        (strategy, target_edge, target_speaker, scores)
    """
    # 関係性スコアを計算
    scores = _calculate_relation_scores(conversation, cfg, debug)

    # 対象エッジと話者を選択
    target_edge, target_speaker = _select_target_edge_and_speaker(scores, debug)
    target_score = scores[target_edge]

    # 会話履歴をフォーマット
    history_lines = [f"[{log['speaker']}] {log['utterance']}" for log in conversation]

    # 学習環境と同じシステムプロンプト（ppo_train.pyのbuild_robot_messagesと同じ）
    system_content = """
あなたは、会話が不安定になっているときに、ロボットがどのように介入すれば関係を安定化できるかを判断し、数字1桁を出力するアシスタントです。
会話は三者（A, B, C）の間で行われており、一時的な対立や不調和が生じています。

【目的】
ターゲットとなる人物を中心に、その人物が含まれる関係（エッジ）を安定（+方向）に導くために、
最も効果的な介入戦略を1つ選択してください。

---

【入力情報】
- 会話履歴（A, B, Cの発話）
- 各ペアの関係スコア（-1〜1）
- 介入対象者（ターゲット）
- 改善すべき関係ペア（エッジ）

---

【選択肢】
1. validate — 感情や意見を承認し、心理的安全性を高める。
2. bridge — 対立している相手との共通点や協力軸を見つけ、調和を促す。
3. plan — 具体的な行動方針を提案し、関係修復を前進させる。
4. no_intervention — 軽度な不調和で自然回復が見込まれる場合、または介入が逆効果となる場合。

---

【判断基準】
会話内容から、ターゲットが今求めている心理的ニーズを推定し、関係を安定化させるために最適な戦略を選択してください。

- 承認や共感を求めている場合 → 1 (validate)
- 仲裁や協調を求めている場合 → 2 (bridge)
- 具体的な行動や方向性を求めている場合 → 3 (plan)
- 自立的に進めたい、干渉を避けたい場合 → 4 (no_intervention)

---

【出力形式】
- 数字1桁（1〜4）のみを出力してください。
- 理由・説明・補足は出力しない。
"""

    # 学習環境のプロンプト + 性格判断のヒント
    prompt_lines = [
        "=== 会話履歴 ===",
        *history_lines,
        "",
        "=== 現在の関係スコア（-1〜1） ===",
        ", ".join(f"w_{edge}={value:+.2f}" for edge, value in scores.items()),
        "",
        f"=== 改善対象エッジ === {target_edge}（現在: {target_score:+.2f}）",
        f"=== 介入対象（ターゲット） === {target_speaker}",
    ]
    user_content = "\n".join(prompt_lines)

    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="intervention")
    max_attempts = 3

    strategy_map = {
        1: "validate",
        2: "bridge",
        3: "plan",
        4: "no_intervention"
    }

    for attempt in range(max_attempts):
        try:
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
            params = build_chat_completion_params(deployment, messages, cfg.llm, temperature=0.3)
            res = client.chat.completions.create(**params)

            if res and getattr(res, "choices", None):
                choice = res.choices[0]
                message = getattr(choice, "message", None)
                if isinstance(message, dict):
                    txt = message.get("content", "")
                else:
                    txt = getattr(message, "content", "")

                txt = (txt or "").strip()

                # 数字を抽出（学習環境と同じロジック）
                strategy_num = None
                for char in txt:
                    if char in '1234':
                        strategy_num = int(char)
                        break

                if strategy_num:
                    strategy = strategy_map[strategy_num]
                    return strategy, target_edge, target_speaker, scores
        except Exception as exc:
            if debug:
                print(f"[select_strategy] attempt {attempt+1} failed: {exc}")
            if attempt < max_attempts - 1:
                time.sleep(0.5)

    # フォールバック：ランダムに返す
    if debug:
        print("[select_strategy] All attempts failed, returning random strategy")
    strategy_num = random.randint(1, 4)
    return strategy_map[strategy_num], target_edge, target_speaker, scores


def test_single_session(
    session_id: int,
    cfg,
    speakers: List[str],
    persona_triggers: Dict[str, List[str]],
    used_triggers: List[str],
    debug: bool = False
) -> Dict[str, Any]:
    """
    単一セッションのテストを実行

    Returns:
        テスト結果の辞書
    """
    # 地雷を選んで話題を生成（学習環境と同じロジック）
    topic, topic_trigger = _select_trigger_and_generate_topic(persona_triggers, used_triggers, cfg, debug)

    print(f"\n{'='*60}")
    print(f"セッション {session_id}")
    print(f"{'='*60}")
    print(f"話題: 「{topic}」")
    if topic_trigger:
        who_has_trigger = [p for p, ts in persona_triggers.items() if topic_trigger in ts]
        print(f"地雷: 「{topic_trigger}」（{', '.join(who_has_trigger)}が不機嫌）")
    print()

    # 各話者に感情ニーズをランダムに割り当て（内部的に保持）
    true_needs = {speaker: random.choice(EMOTIONAL_NEEDS) for speaker in speakers}

    if debug:
        print(f"真の感情ニーズ: {true_needs}")

    # 3発話を生成（各話者1回ずつ）
    conversation = []
    print("【初期会話】")
    for speaker in speakers:
        utterance = _generate_human_utterance_with_need(
            speaker=speaker,
            emotional_need=true_needs[speaker],
            conversation_history=conversation,
            topic=topic,
            topic_trigger=topic_trigger,
            persona_triggers=persona_triggers,
            cfg=cfg,
            debug=debug
        )
        conversation.append({"speaker": speaker, "utterance": utterance})
        print(f"{speaker}: {utterance}")

    print()

    # GPT-5で戦略を推定（学習環境と同じロジック）
    gpt5_strategy, target_edge, target_speaker, scores = select_strategy_from_conversation(conversation, cfg, debug)

    # 対象話者の真のニーズと正解戦略
    true_need = true_needs[target_speaker]
    correct_strategy = NEED_TO_STRATEGY[true_need]

    strategy_correct = (gpt5_strategy == correct_strategy)

    print(f"【関係性スコア】")
    for edge, score in scores.items():
        print(f"  {edge}: {score:+.2f}")
    print()

    print(f"【戦略推定】")
    print(f"対象エッジ: {target_edge} (スコア: {scores[target_edge]:+.2f})")
    print(f"対象話者: {target_speaker}")
    print(f"真のニーズ: {true_need} ({NEED_DESCRIPTIONS[true_need]}) → 期待戦略: {correct_strategy}")
    print(f"GPT-5の選択: {gpt5_strategy} (数字: {STRATEGY_TO_NUM[gpt5_strategy]})")
    print(f"結果: {'✓ 正解' if strategy_correct else '✗ 不正解'}")

    return {
        "session_id": session_id,
        "topic": topic,
        "topic_trigger": topic_trigger,
        "true_needs": true_needs,
        "conversation": conversation,
        "scores": scores,
        "target_edge": target_edge,
        "target_speaker": target_speaker,
        "true_need": true_need,
        "correct_strategy": correct_strategy,
        "gpt5_strategy": gpt5_strategy,
        "strategy_correct": strategy_correct
    }


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """統計を計算"""
    # 戦略推定の統計
    total_sessions = len(results)
    strategy_correct_count = sum(1 for r in results if r["strategy_correct"])
    strategy_accuracy = strategy_correct_count / total_sessions if total_sessions > 0 else 0.0

    # ニーズタイプ別の戦略推定精度
    need_stats = {need: {"correct": 0, "total": 0} for need in EMOTIONAL_NEEDS}

    for result in results:
        true_need = result["true_need"]
        need_stats[true_need]["total"] += 1
        if result["strategy_correct"]:
            need_stats[true_need]["correct"] += 1

    need_accuracies = {
        need: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        for need, stats in need_stats.items()
    }

    # 戦略ごとの使用頻度
    strategy_counts = defaultdict(int)
    for result in results:
        strategy_counts[result["gpt5_strategy"]] += 1

    return {
        "strategy_estimation": {
            "overall_accuracy": strategy_accuracy,
            "total_correct": strategy_correct_count,
            "total_sessions": total_sessions,
            "by_need": need_accuracies,
            "strategy_distribution": dict(strategy_counts)
        }
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-5による戦略推定テスト")
    parser.add_argument("--num-sessions", type=int, default=20, help="テストセッション数")
    parser.add_argument("--debug", action="store_true", help="デバッグ出力を有効化")
    parser.add_argument("--output-dir", type=str, default=None, help="出力ディレクトリ")
    args = parser.parse_args()

    cfg = get_config()

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)

    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    speakers = ["A", "B", "C"]

    # 地雷システムの初期化（学習環境と同じ）
    persona_triggers = _get_persona_triggers(cfg)
    used_triggers: List[str] = []

    print(f"{'='*60}")
    print(f"GPT-5 戦略推定テスト")
    print(f"{'='*60}")
    print(f"セッション数: {args.num_sessions}")
    print(f"地雷システム: 有効")
    for persona, triggers in persona_triggers.items():
        print(f"  {persona}: {len(triggers)}個の地雷")
    print(f"{'='*60}")

    # テスト実行
    results = []
    for i in range(1, args.num_sessions + 1):
        result = test_single_session(
            session_id=i,
            cfg=cfg,
            speakers=speakers,
            persona_triggers=persona_triggers,
            used_triggers=used_triggers,
            debug=args.debug
        )
        results.append(result)

    # 統計計算
    print(f"\n統計を計算中...")
    stats = calculate_statistics(results)

    # JSONL出力
    jsonl_path = output_dir / f"emotional_needs_test_{timestamp}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"詳細ログを保存: {jsonl_path}")

    # 統計JSON出力
    stats_path = output_dir / f"emotional_needs_test_{timestamp}_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"統計データを保存: {stats_path}")

    # 統計表示
    print(f"\n{'='*60}")
    print(f"GPT-5 戦略推定テスト結果")
    print(f"{'='*60}\n")

    print(f"テストセッション数: {args.num_sessions}\n")

    print(f"【戦略推定精度】")
    print(f"全体: {stats['strategy_estimation']['total_correct']}/{stats['strategy_estimation']['total_sessions']} ({stats['strategy_estimation']['overall_accuracy']*100:.1f}%)")

    print(f"\n【ニーズタイプ別の精度】")
    for need, accuracy in stats['strategy_estimation']['by_need'].items():
        count = sum(1 for r in results if r['true_need'] == need)
        correct = sum(1 for r in results if r['true_need'] == need and r['strategy_correct'])
        correct_strategy = NEED_TO_STRATEGY[need]
        print(f"- {need} (→ {correct_strategy}): {correct}/{count} ({accuracy*100:.1f}%)")

    print(f"\n【戦略の使用頻度】")
    for strategy, count in stats['strategy_estimation']['strategy_distribution'].items():
        print(f"- {strategy}: {count}回 ({count/args.num_sessions*100:.1f}%)")

    print(f"\n【考察】")
    accuracy = stats['strategy_estimation']['overall_accuracy']
    if accuracy > 0.7:
        print(f"- 精度が高すぎます（{accuracy*100:.1f}%）")
        print("  → 会話から戦略を推定するのは比較的容易")
        print("  → 学習課題としては簡単すぎる可能性")
    elif accuracy < 0.4:
        print(f"- 精度が低すぎます（{accuracy*100:.1f}%）")
        print("  → 会話から戦略を推定するのは非常に困難")
        print("  → ニーズの表現が不明瞭すぎる可能性")
    else:
        print(f"- 適切な難易度です（{accuracy*100:.1f}%、理想: 40-70%）")
        print("  → 学習環境への導入を検討できます")


if __name__ == "__main__":
    main()
