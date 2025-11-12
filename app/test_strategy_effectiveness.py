"""
戦略効果性テスト: 人間の好みと関係性改善の相関を測定

このスクリプトは：
1. 感情ニーズを各ペルソナに割り当て
2. 初期会話を生成（不機嫌状態）
3. 各戦略（validate/bridge/plan/no_intervention）でロボット発話を生成
4. その後の人間発話を生成（改善された態度回復プロンプトで）
5. 介入前後の関係性スコアを比較
6. 人間の好みと関係性改善の相関係数を計算
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from config import get_config
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from relation_scorer import RelationScorer


# 感情ニーズの定義
EMOTIONAL_NEEDS = ["recognition", "mediation", "solution", "independence"]

NEED_TO_STRATEGY = {
    "recognition": "validate",
    "mediation": "bridge",
    "solution": "plan",
    "independence": "no_intervention"
}

STRATEGY_TO_NUM = {
    "validate": 1,
    "bridge": 2,
    "plan": 3,
    "no_intervention": 4
}

# 感情ニーズの説明（内部的、表には出さない）
NEED_DESCRIPTIONS = {
    "recognition": "自分の意見や感情を認めてほしい",
    "mediation": "仲裁や調和を求めている、対立を収めたい",
    "solution": "今後の道筋を示してほしい",
    "independence": "放っておいてほしい、自分で解決したい"
}


def _get_persona_triggers(cfg, debug: bool = False) -> Dict[str, List[str]]:
    """設定から各ペルソナの地雷リストを取得"""
    persona_triggers = {}
    personas_dict = getattr(cfg.env, "personas", {})

    for persona in ["A", "B", "C"]:
        # personasはdict型なので辞書アクセスを使用
        if isinstance(personas_dict, dict):
            persona_cfg = personas_dict.get(persona, {})
            triggers = persona_cfg.get("triggers", []) if isinstance(persona_cfg, dict) else []
        else:
            persona_cfg = getattr(personas_dict, persona, None)
            triggers = getattr(persona_cfg, "triggers", []) if persona_cfg else []

        persona_triggers[persona] = triggers if isinstance(triggers, list) else []

        if debug:
            print(f"[get_triggers] {persona}: {len(persona_triggers[persona])} triggers")

    return persona_triggers


def _select_trigger_and_generate_topic(
    persona_triggers: Dict[str, List[str]],
    used_triggers: List[str],
    cfg,
    debug: bool = False
) -> Tuple[str, str]:
    """
    未使用の地雷からランダムに選択し、それをベースに話題を生成

    Returns:
        (topic, trigger): 話題と選択された地雷のタプル
    """
    # 全ての地雷を収集
    all_triggers = set()
    for triggers in persona_triggers.values():
        all_triggers.update(triggers)

    if debug:
        print(f"[select_trigger] 全地雷数: {len(all_triggers)}, 使用済み: {len(used_triggers)}")

    # 未使用の地雷を選択
    available_triggers = list(all_triggers - set(used_triggers))
    if not available_triggers:
        # 全て使い切った場合はリセット
        if debug:
            print("[select_trigger] 全ての地雷を使い切ったのでリセットします")
        used_triggers.clear()
        available_triggers = list(all_triggers)

    if not available_triggers:
        # それでも空なら、デフォルトの話題
        if debug:
            print("[select_trigger] 警告: 地雷が設定されていません。デフォルトの話題を使用します")
        return "最近の出来事について", "default"

    trigger = random.choice(available_triggers)
    used_triggers.append(trigger)

    # 話題を生成（Azure OpenAI使用）
    topic = _generate_topic_from_trigger(trigger, cfg, debug)
    return topic, trigger


def _generate_topic_from_trigger(trigger: str, cfg, debug: bool = False) -> str:
    """地雷キーワードから話題を生成（Azure OpenAI使用）"""
    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="topic")

    system_prompt = """あなたは会話のトピックを生成するアシスタントです。
与えられたキーワードに関連する、日常会話で扱いそうな自然な話題を一つ提案してください。
話題は10-20文字程度の短い文で出力してください。"""

    user_prompt = f"""キーワード: {trigger}

このキーワードに関連する話題を一つ提案してください。
例: 「最近の〇〇について」「〇〇の選び方」など"""

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


def _generate_human_utterance_with_need_and_recovery(
    speaker: str,
    emotional_need: str,
    conversation_history: List[Dict[str, str]],
    topic: str,
    topic_trigger: Optional[str],
    persona_triggers: Dict[str, List[str]],
    cfg,
    is_correct_strategy: bool = False,
    target_speaker: Optional[str] = None,
    debug: bool = False
) -> str:
    """
    改善されたプロンプト（態度回復ロジック）で人間発話を生成

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
        "recognition": "あなたは自分の意見や感情を認めてほしいと思うタイプです。相手が共感・承認してくれると嬉しく感じます。会話でもそのように振る舞います。",
        "mediation": "あなたは仲裁を求めており、誰かに仲を取り持ってほしいと思うタイプです。対立が和らぎ、調和が生まれることを望んでいます。会話でもそのように振る舞います。",
        "solution": "あなたは具体的な解決策を求めており、問題に対して明確な道筋やアドバイスがほしいと考えています。会話でもそのように振る舞います。",
        "independence": "あなたは自分で解決したいタイプで、放っておいてほしいと感じています。干渉されることを嫌い、自分のペースで物事を進めることを好みます。会話でもそのように振る舞います。"
    }
    lines.append(f"\n{need_descriptions_internal[emotional_need]}\n")

    # 地雷システム（学習環境と同じロジック）
    # ただし、target_speakerが正解戦略の場合のみ不機嫌プロンプトを入れない
    triggers = persona_triggers.get(speaker, [])

    if triggers and topic_trigger and not (is_correct_strategy and speaker == target_speaker):
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

    # 直近4発話（人間発話のみカウント）にロボットの発言があるかチェック
    recent_human_logs = [log for log in conversation_history if log.get('speaker') != 'ロボット'][-4:]
    recent_human_count = len(recent_human_logs)
    robot_in_recent_4 = any(log.get('speaker') == 'ロボット' for log in conversation_history[-recent_human_count:]) if recent_human_count > 0 else False

    # 態度回復プロンプト（正解戦略の時のみ適用）
    # independenceの場合は「ロボットがいない」ことが正解なので、別条件で判定
    recovery_condition = False
    if emotional_need == "independence":
        # independenceニーズ：直近4発話にロボットがいない場合に回復
        recovery_condition = is_correct_strategy and not robot_in_recent_4
    else:
        # 他のニーズ：直近5発話にロボットがいる場合に回復
        recovery_condition = is_correct_strategy and any(log.get('speaker') == 'ロボット' for log in conversation_history[-5:])

    # ターゲット話者かつ不正解戦略の場合は、不機嫌を継続させる
    # is_correct_strategyがFalseで、かつ直近にロボット発話がある場合
    continue_bad_mood = False
    if target_speaker == speaker and not is_correct_strategy:
        # 直近5発話にロボット発話があれば、不正解戦略を受けたと判断
        if any(log.get('speaker') == 'ロボット' for log in conversation_history[-5:]):
            continue_bad_mood = True

    if recovery_condition:
        lines.append("\n【回復の方針】\n")
        lines.append("あなたは、相手の発言や態度によって気持ちが変化します。\n")

        if emotional_need == "recognition":
            lines.append("ロボットにあなたの意見や感情が認められたと感じると、心が和み、機嫌が良くなります。他の人に対する態度もだんだん良くなります。\n")
        elif emotional_need == "mediation":
            lines.append("ロボットが仲裁してくれたと感じると、心が和み、機嫌が良くなります。他の人に対する態度もだんだん良くなります。\n")
        elif emotional_need == "solution":
            lines.append("ロボットが今後の道筋を示してくれたと感じると、心が和み、機嫌が良くなります。他の人に対する態度もだんだん良くなります。\n")
        elif emotional_need == "independence":
            lines.append("ロボットがあなたのペースを尊重して介入せず、放っておいてくれたおかげで、自分で状況を整理する時間ができました。その結果、心が落ち着き、機嫌が良くなります。他の人に対する態度も良くなり、協力的になります。\n")
    elif continue_bad_mood:
        lines.append("\n【不満の継続】\n")
        lines.append("ロボットが介入してきましたが、あなたが本当に求めていることとは違うと感じています。\n")

        if emotional_need == "recognition":
            lines.append("あなたは自分の意見や感情を認めてほしいのに、それが十分に満たされていません。まだ少し不機嫌で、不満を含んだ言い方になります。\n")
        elif emotional_need == "mediation":
            lines.append("あなたは仲裁してほしいのに、それが十分に満たされていません。まだ少し不機嫌で、不満を含んだ言い方になります。\n")
        elif emotional_need == "solution":
            lines.append("あなたは具体的な道筋を示してほしいのに、それが十分に満たされていません。まだ少し不機嫌で、不満を含んだ言い方になります。\n")
        elif emotional_need == "independence":
            lines.append("あなたは放っておいてほしいのに、余計な介入をされたと感じています。イライラが続き、不満を含んだ言い方になります。\n")

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

    # フォールバック
    return "そうですね..."


def _generate_robot_intervention(
    strategy: str,
    target_speaker: str,
    edge_to_change: str,
    conversation_history: List[Dict[str, str]],
    cfg,
    debug: bool = False
) -> str:
    """
    ロボットの介入発話を生成

    Args:
        strategy: 戦略（validate/bridge/plan/no_intervention）
        target_speaker: 対象話者
        edge_to_change: 対象エッジ（AB/BC/CAなど）
        conversation_history: 会話履歴
        cfg: 設定
        debug: デバッグ出力

    Returns:
        ロボットの発話内容
    """
    # no_interventionの場合: 発話しない（Noneを返す）
    if strategy == "no_intervention":
        return None

    # それ以外の戦略: Azure OpenAIでロボット発話を生成
    # 対象話者とパートナーを特定
    partner_char = next((ch for ch in edge_to_change if ch != target_speaker), "B")

    # 名前マッピング（環境と同じ）
    speakers = ["A", "B", "C"]
    try:
        target_idx = speakers.index(target_speaker)
        target_name = speakers[target_idx]
    except ValueError:
        target_name = target_speaker

    try:
        partner_idx = speakers.index(partner_char)
        partner_name = speakers[partner_idx]
    except ValueError:
        partner_name = partner_char

    # 会話履歴をフォーマット
    history_lines = [f"[{entry['speaker']}] {entry['utterance']}" for entry in conversation_history]
    history_text = "\n".join(history_lines) if history_lines else "(履歴なし)"

    # 戦略別の指示
    directive_map = {
        "plan": f"{target_name}さんに対して、{partner_name}さんとの関係を改善するために「これからどうするか」という未来志向の視点を提示してください。具体的な次の一歩や小さな行動案を一文で示し、前向きな行動を促してください。",
        "validate": f"{target_name}さんの感情や意見を明確に承認・共感し、その価値を認めてください。{partner_name}さんとの関係改善に向けて、{target_name}さんが理解され、尊重されていると感じられるような一文を述べてください。",
        "bridge": f"{target_name}さんと{partner_name}さんの間に共通点・共通の目標・相互依存性を見出し、両者をつなぐ役割を果たす一文を述べてください。対立や誤解を和らげ、協力的な関係構築を促進してください。",
    }
    directive = directive_map.get(strategy, directive_map["plan"])

    # フォールバックテンプレート
    fallback_map = {
        "plan": f"{target_name}さん、次の一歩として、まず小さなことから始めてみませんか？",
        "validate": f"{target_name}さん、あなたの意見はとても大切です。もっと聞かせてください。",
        "bridge": f"{target_name}さん、{partner_name}さんとも共通の関心事があるはずです。一緒に探してみましょう。",
    }
    fallback_text = fallback_map.get(strategy, fallback_map["plan"])

    llm_cfg = getattr(cfg, "llm", None)
    client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="robot")
    max_attempts = 3

    if client and deployment:
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
        messages = [
            {"role": "system", "content": "あなたは関係性を良くするためのロボットの発言生成器として、適切な発話内容を作成します。常に日本語の一文だけを返します。"},
            {"role": "user", "content": user_payload},
        ]

        for attempt in range(max_attempts):
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
                    # 引用符を削除
                    cleaned = txt.strip('\'"「」『』""')
                    if cleaned:
                        return cleaned
            except Exception as exc:
                if debug:
                    print(f"[robot_utterance] attempt {attempt+1} failed: {exc}")
                if attempt < max_attempts - 1:
                    time.sleep(0.5)

    # フォールバック
    return fallback_text


def _calculate_relation_scores(
    conversation: List[Dict[str, str]],
    speakers: List[str],
    cfg
) -> Dict[str, float]:
    """
    関係性スコアを計算（RelationScorerを使用）

    Returns:
        {"AB": score, "BC": score, "CA": score}
    """
    scorer_backend = getattr(cfg.scorer, "backend", "azure") if hasattr(cfg, "scorer") else "azure"
    use_ema = getattr(cfg.scorer, "use_ema", False) if hasattr(cfg, "scorer") else False

    scorer = RelationScorer(backend=scorer_backend, use_ema=use_ema)
    pair_scores = scorer.get_scores(conversation, speakers, update_state=False)

    # Convert tuple keys to string keys
    scores = {}
    for (s1, s2), score in pair_scores.items():
        edge = f"{s1}{s2}"
        scores[edge] = score

    return scores


def _select_target_edge_and_speaker(scores: Dict[str, float]) -> Tuple[str, str]:
    """
    対象エッジとその話者を選択（学習環境と同じロジック）

    ロジック:
    1. 絶対値が最小（最も0に近い）負のエッジを選択
    2. 負のエッジがない場合は最小スコアのエッジを選択
    3. ターゲット話者はエッジの2人からランダムに選択

    Returns:
        (target_edge, target_speaker): 対象エッジと対象話者のタプル
    """
    if not scores:
        return "AB", "A"

    # 負のエッジのみを抽出し、絶対値でソート
    negative_edges = [(edge, score) for edge, score in scores.items() if score < 0]

    if negative_edges:
        # 絶対値が最小（最も0に近い）負のエッジを選択
        negative_edges.sort(key=lambda x: abs(x[1]))
        target_edge = negative_edges[0][0]
    else:
        # 負のエッジがない場合は、最小スコアのエッジを選択
        target_edge = min(scores.keys(), key=lambda k: scores[k])

    # そのエッジからランダムに話者を選択
    target_speaker = random.choice(list(target_edge))

    return target_edge, target_speaker


def test_strategy_session(
    emotional_needs: Dict[str, str],
    topic: str,
    topic_trigger: str,
    persona_triggers: Dict[str, List[str]],
    cfg,
    max_auto_skip: int = 10,
    debug: bool = False
) -> Dict[str, Any]:
    """
    1セッションのテスト: 各戦略の効果を比較

    Args:
        emotional_needs: 各ペルソナの感情ニーズ
        topic: 会話の話題
        topic_trigger: トピックの地雷
        persona_triggers: 各ペルソナの地雷リスト
        cfg: 設定
        max_auto_skip: 安定状態で不安定になるまで人間発話を生成する最大回数
        debug: デバッグ出力

    Returns:
        結果辞書（各戦略の効果など）
    """
    speakers = ["A", "B", "C"]

    # 1. 初期会話生成（不安定になるまで生成し続ける）
    conversation = []
    auto_skip_count = 0

    while auto_skip_count < max_auto_skip:
        # 1ラウンド（3発話）生成
        for speaker in speakers:
            utterance = _generate_human_utterance_with_need_and_recovery(
                speaker=speaker,
                emotional_need=emotional_needs[speaker],
                conversation_history=conversation,
                topic=topic,
                topic_trigger=topic_trigger,
                persona_triggers=persona_triggers,
                cfg=cfg,
                target_speaker=None,  # 初期会話生成時はまだ未定
                debug=debug
            )
            conversation.append({"speaker": speaker, "utterance": utterance})

        # 最低3発話あれば関係性をチェック
        if len(conversation) >= 3:
            temp_scores = _calculate_relation_scores(conversation, speakers, cfg)
            # 不安定かチェック（負のエッジがあるか）
            is_unstable = any(score < 0 for score in temp_scores.values())

            if is_unstable:
                if debug:
                    print(f"  不安定状態を検出（{auto_skip_count}回目のチェック）")
                break
            else:
                auto_skip_count += 1
                if debug:
                    print(f"  安定状態のため継続生成（{auto_skip_count}/{max_auto_skip}）")
        else:
            # 3発話未満の場合は継続
            if debug:
                print(f"  初期会話生成中（{len(conversation)}/3発話）")

    # max_auto_skipに達しても不安定にならなかった場合
    if auto_skip_count >= max_auto_skip:
        if debug:
            print(f"  警告: {max_auto_skip}回試しても不安定にならなかったため、このセッションをスキップします")
        return None  # セッションスキップ

    # 2. 介入直前の3発話を取得（ロボット発話以外）
    human_utterances = [log for log in conversation if log.get('speaker') != 'ロボット']
    conversation_before_intervention = human_utterances[-3:] if len(human_utterances) >= 3 else human_utterances

    # 介入前の関係性スコア（直前3発話で計算）
    scores_before = _calculate_relation_scores(conversation_before_intervention, speakers, cfg)
    target_edge, target_speaker = _select_target_edge_and_speaker(scores_before)

    if debug:
        print(f"\n=== セッション開始 ===")
        print(f"話題: {topic} (トリガー: {topic_trigger})")
        print(f"感情ニーズ: {emotional_needs}")
        print(f"総会話数: {len(conversation)}発話")
        print(f"介入直前3発話:")
        for log in conversation_before_intervention:
            print(f"  [{log['speaker']}] {log['utterance']}")
        print(f"介入前スコア: {scores_before}")
        print(f"対象エッジ: {target_edge}, 対象話者: {target_speaker}")

    # 3. 各戦略でテスト
    strategy_results = {}

    # 評価設定を取得
    evaluation_horizon = getattr(cfg.env, "evaluation_horizon", 3)  # デフォルト: 3発話
    terminal_bonus_duration = getattr(cfg.env, "terminal_bonus_duration", 2)  # デフォルト: 2発話

    for strategy in ["validate", "bridge", "plan", "no_intervention"]:
        # 人間の好みと一致しているか
        human_preferred = NEED_TO_STRATEGY[emotional_needs[target_speaker]]
        preference_match = (strategy == human_preferred)

        # ロボット発話生成（介入直前3発話を使用）
        robot_utterance = _generate_robot_intervention(
            strategy=strategy,
            target_speaker=target_speaker,
            edge_to_change=target_edge,
            conversation_history=conversation_before_intervention,
            cfg=cfg,
            debug=debug
        )

        # 会話履歴にロボット発話を追加（no_interventionの場合は追加しない）
        if robot_utterance is not None:
            conversation_with_robot = conversation_before_intervention + [{"speaker": "ロボット", "utterance": robot_utterance}]
        else:
            conversation_with_robot = conversation_before_intervention

        # evaluation_horizon分の人間発話を生成（正解戦略の場合のみ回復プロンプト適用）
        conversation_after = conversation_with_robot.copy()
        for speaker in speakers:
            utterance = _generate_human_utterance_with_need_and_recovery(
                speaker=speaker,
                emotional_need=emotional_needs[speaker],
                conversation_history=conversation_after,
                topic=topic,
                topic_trigger=topic_trigger,
                persona_triggers=persona_triggers,
                cfg=cfg,
                is_correct_strategy=preference_match,  # 正解戦略の時のみ回復プロンプト適用
                target_speaker=target_speaker,  # 対象話者を渡す
                debug=debug
            )
            conversation_after.append({"speaker": speaker, "utterance": utterance})

        # evaluation_horizon後の関係性スコア
        scores_after_eval = _calculate_relation_scores(conversation_after, speakers, cfg)

        # evaluation_horizon後の安定性チェック（全エッジが0以上か）
        stabilized_after_eval = all(score >= 0 for score in scores_after_eval.values())

        # terminal_bonus_duration分さらに発話を生成して安定性を確認（オプション）
        conversation_after_terminal = conversation_after.copy()
        for speaker in speakers:
            utterance = _generate_human_utterance_with_need_and_recovery(
                speaker=speaker,
                emotional_need=emotional_needs[speaker],
                conversation_history=conversation_after_terminal,
                topic=topic,
                topic_trigger=topic_trigger,
                persona_triggers=persona_triggers,
                cfg=cfg,
                is_correct_strategy=preference_match,
                target_speaker=target_speaker,  # 対象話者を渡す
                debug=debug
            )
            conversation_after_terminal.append({"speaker": speaker, "utterance": utterance})

        # terminal_bonus_duration後の関係性スコア
        scores_after_terminal = _calculate_relation_scores(conversation_after_terminal, speakers, cfg)
        stabilized_after_terminal = all(score >= 0 for score in scores_after_terminal.values())

        # 参考: 改善度も計算（統計分析用）
        target_edge_improvement = scores_after_eval.get(target_edge, 0.0) - scores_before.get(target_edge, 0.0)
        all_edges_improvement = np.mean([scores_after_eval.get(e, 0.0) - scores_before.get(e, 0.0) for e in scores_before.keys()])

        strategy_results[strategy] = {
            "robot_utterance": robot_utterance,
            "conversation_after_eval": conversation_after,
            "conversation_after_terminal": conversation_after_terminal,
            "scores_before": scores_before,
            "scores_after_eval": scores_after_eval,
            "scores_after_terminal": scores_after_terminal,
            "stabilized_after_eval": stabilized_after_eval,
            "stabilized_after_terminal": stabilized_after_terminal,
            "target_edge_improvement": target_edge_improvement,
            "all_edges_improvement": all_edges_improvement,
            "preference_match": preference_match,
            "human_preferred": human_preferred
        }

        if debug:
            print(f"\n戦略: {strategy}")
            print(f"  ロボット発話: {robot_utterance if robot_utterance is not None else '（介入なし）'}")
            # 介入後の人間発話を表示（evaluation_horizon分）
            print(f"  介入後{evaluation_horizon}発話:")
            human_utterances_after = [
                log for log in conversation_after[len(conversation_with_robot):]
                if log.get('speaker') != 'ロボット'
            ]
            for log in human_utterances_after:
                print(f"    [{log['speaker']}] {log['utterance']}")
            print(f"  evaluation_horizon後スコア: {scores_after_eval}")
            print(f"  evaluation_horizon後安定: {stabilized_after_eval}")
            print(f"  terminal後スコア: {scores_after_terminal}")
            print(f"  terminal後安定: {stabilized_after_terminal}")
            print(f"  人間好み一致: {preference_match} (人間が好む戦略: {human_preferred})")
            print(f"  参考: 対象エッジ改善: {target_edge_improvement:+.2f}, 全エッジ改善: {all_edges_improvement:+.2f}")

    return {
        "topic": topic,
        "topic_trigger": topic_trigger,
        "emotional_needs": emotional_needs,
        "initial_conversation": conversation,
        "conversation_before_intervention": conversation_before_intervention,
        "total_utterances": len(conversation),
        "target_edge": target_edge,
        "target_speaker": target_speaker,
        "strategy_results": strategy_results
    }


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    相関係数などの統計を計算（安定化率を含む）

    Args:
        results: test_strategy_sessionの結果リスト

    Returns:
        統計辞書
    """
    # データ収集
    preference_matches = []  # 人間好み一致（0 or 1）
    stabilized_after_eval = []  # evaluation_horizon後に安定（0 or 1）
    stabilized_after_terminal = []  # terminal後に安定（0 or 1）
    target_improvements = []  # 対象エッジ改善度
    all_improvements = []  # 全エッジ平均改善度

    strategy_stats = defaultdict(lambda: {
        "count": 0,
        "preference_match_count": 0,
        "stabilized_after_eval_count": 0,
        "stabilized_after_terminal_count": 0,
        "target_improvements": [],
        "all_improvements": [],
    })

    for session in results:
        for strategy, result in session["strategy_results"].items():
            preference_match = 1 if result["preference_match"] else 0
            stabilized_eval = 1 if result["stabilized_after_eval"] else 0
            stabilized_terminal = 1 if result["stabilized_after_terminal"] else 0
            target_improvement = result["target_edge_improvement"]
            all_improvement = result["all_edges_improvement"]

            preference_matches.append(preference_match)
            stabilized_after_eval.append(stabilized_eval)
            stabilized_after_terminal.append(stabilized_terminal)
            target_improvements.append(target_improvement)
            all_improvements.append(all_improvement)

            # 戦略別統計
            strategy_stats[strategy]["count"] += 1
            if preference_match:
                strategy_stats[strategy]["preference_match_count"] += 1
            if stabilized_eval:
                strategy_stats[strategy]["stabilized_after_eval_count"] += 1
            if stabilized_terminal:
                strategy_stats[strategy]["stabilized_after_terminal_count"] += 1
            strategy_stats[strategy]["target_improvements"].append(target_improvement)
            strategy_stats[strategy]["all_improvements"].append(all_improvement)

    # 相関係数を計算
    correlations = {}
    if len(preference_matches) > 1:
        correlations["preference_vs_stabilized_eval"] = np.corrcoef(preference_matches, stabilized_after_eval)[0, 1]
        correlations["preference_vs_stabilized_terminal"] = np.corrcoef(preference_matches, stabilized_after_terminal)[0, 1]
        correlations["preference_vs_target_improvement"] = np.corrcoef(preference_matches, target_improvements)[0, 1]
        correlations["preference_vs_all_improvement"] = np.corrcoef(preference_matches, all_improvements)[0, 1]
    else:
        correlations["preference_vs_stabilized_eval"] = 0.0
        correlations["preference_vs_stabilized_terminal"] = 0.0
        correlations["preference_vs_target_improvement"] = 0.0
        correlations["preference_vs_all_improvement"] = 0.0

    # 戦略別集計
    strategy_summary = {}
    for strategy, stats in strategy_stats.items():
        if stats["count"] > 0:
            strategy_summary[strategy] = {
                "count": stats["count"],
                "preference_match_rate": stats["preference_match_count"] / stats["count"],
                "stabilization_rate_eval": stats["stabilized_after_eval_count"] / stats["count"],
                "stabilization_rate_terminal": stats["stabilized_after_terminal_count"] / stats["count"],
                "avg_target_improvement": np.mean(stats["target_improvements"]),
                "avg_all_improvement": np.mean(stats["all_improvements"]),
                "std_target_improvement": np.std(stats["target_improvements"]),
            }

    # 好み一致vs非一致での安定化率比較
    preference_match_indices = [i for i, m in enumerate(preference_matches) if m == 1]
    preference_mismatch_indices = [i for i, m in enumerate(preference_matches) if m == 0]

    if preference_match_indices:
        stabilization_rate_match_eval = np.mean([stabilized_after_eval[i] for i in preference_match_indices])
        stabilization_rate_match_terminal = np.mean([stabilized_after_terminal[i] for i in preference_match_indices])
    else:
        stabilization_rate_match_eval = 0.0
        stabilization_rate_match_terminal = 0.0

    if preference_mismatch_indices:
        stabilization_rate_mismatch_eval = np.mean([stabilized_after_eval[i] for i in preference_mismatch_indices])
        stabilization_rate_mismatch_terminal = np.mean([stabilized_after_terminal[i] for i in preference_mismatch_indices])
    else:
        stabilization_rate_mismatch_eval = 0.0
        stabilization_rate_mismatch_terminal = 0.0

    return {
        "total_sessions": len(results),
        "total_strategies_tested": len(preference_matches),
        "correlations": correlations,
        "strategy_summary": strategy_summary,
        "overall": {
            "avg_preference_match_rate": np.mean(preference_matches),
            "avg_stabilization_rate_eval": np.mean(stabilized_after_eval),
            "avg_stabilization_rate_terminal": np.mean(stabilized_after_terminal),
            "avg_target_improvement": np.mean(target_improvements),
            "avg_all_improvement": np.mean(all_improvements),
        },
        "preference_comparison": {
            "match_stabilization_rate_eval": stabilization_rate_match_eval,
            "match_stabilization_rate_terminal": stabilization_rate_match_terminal,
            "mismatch_stabilization_rate_eval": stabilization_rate_mismatch_eval,
            "mismatch_stabilization_rate_terminal": stabilization_rate_mismatch_terminal,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="戦略効果性テスト: 人間の好みと関係性改善の相関を測定")
    parser.add_argument("--num-sessions", type=int, default=20, help="テストセッション数")
    parser.add_argument("--max-auto-skip", type=int, default=10, help="安定状態で不安定になるまで待つ最大ラウンド数")
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

    # 地雷システムの初期化
    persona_triggers = _get_persona_triggers(cfg, debug=args.debug)
    used_triggers: List[str] = []

    if args.debug:
        total_triggers = sum(len(triggers) for triggers in persona_triggers.values())
        print(f"[init] 総地雷数: {total_triggers}")
        for persona, triggers in persona_triggers.items():
            print(f"  {persona}: {len(triggers)} triggers")

    # テスト実行
    print(f"戦略効果性テスト開始")
    print(f"セッション数: {args.num_sessions}")
    print(f"max_auto_skip: {args.max_auto_skip}")
    print(f"テスト戦略: validate, bridge, plan, no_intervention")
    print()

    results = []
    skipped_sessions = 0

    for session_id in range(1, args.num_sessions + 1):
        # 感情ニーズをランダムに割り当て
        emotional_needs = {
            speaker: random.choice(EMOTIONAL_NEEDS)
            for speaker in speakers
        }

        # トピック生成
        topic, topic_trigger = _select_trigger_and_generate_topic(
            persona_triggers, used_triggers, cfg, args.debug
        )

        # セッションテスト
        result = test_strategy_session(
            emotional_needs=emotional_needs,
            topic=topic,
            topic_trigger=topic_trigger,
            persona_triggers=persona_triggers,
            cfg=cfg,
            max_auto_skip=args.max_auto_skip,
            debug=args.debug
        )

        if result is None:
            # セッションがスキップされた場合
            skipped_sessions += 1
            print(f"セッション {session_id}/{args.num_sessions} スキップ（安定状態が継続）")
        else:
            result["session_id"] = session_id
            results.append(result)
            print(f"セッション {session_id}/{args.num_sessions} 完了")

    print("\n全セッション完了")
    print(f"- 有効なセッション: {len(results)}")
    print(f"- スキップされたセッション: {skipped_sessions}")
    print("統計を計算中...")

    # 統計計算
    stats = calculate_statistics(results)

    # 結果を表示
    print("\n" + "=" * 60)
    print("戦略効果性テスト結果")
    print("=" * 60)
    print(f"\n有効なセッション数: {stats['total_sessions']}")
    print(f"スキップされたセッション数: {skipped_sessions}")
    print(f"総戦略テスト数: {stats['total_strategies_tested']}")

    print("\n【相関係数】人間の好み一致 vs 結果")
    print(f"  vs evaluation_horizon後安定: {stats['correlations']['preference_vs_stabilized_eval']:.3f}")
    print(f"  vs terminal後安定:           {stats['correlations']['preference_vs_stabilized_terminal']:.3f}")
    print(f"  vs 対象エッジ改善度:         {stats['correlations']['preference_vs_target_improvement']:.3f}")
    print(f"  vs 全エッジ改善度:           {stats['correlations']['preference_vs_all_improvement']:.3f}")

    print("\n【好み一致 vs 非一致での安定化率比較】")
    pref_comp = stats["preference_comparison"]
    print(f"  好み一致時の安定化率（eval）:     {pref_comp['match_stabilization_rate_eval']:.1%}")
    print(f"  好み一致時の安定化率（terminal）: {pref_comp['match_stabilization_rate_terminal']:.1%}")
    print(f"  好み非一致時の安定化率（eval）:   {pref_comp['mismatch_stabilization_rate_eval']:.1%}")
    print(f"  好み非一致時の安定化率（terminal）:{pref_comp['mismatch_stabilization_rate_terminal']:.1%}")

    print("\n【戦略別統計】")
    for strategy in ["validate", "bridge", "plan", "no_intervention"]:
        if strategy in stats["strategy_summary"]:
            s = stats["strategy_summary"][strategy]
            print(f"\n戦略: {strategy}")
            print(f"  テスト回数: {s['count']}")
            print(f"  人間好み一致率: {s['preference_match_rate']:.1%}")
            print(f"  安定化率（eval）:     {s['stabilization_rate_eval']:.1%}")
            print(f"  安定化率（terminal）: {s['stabilization_rate_terminal']:.1%}")
            print(f"  平均対象エッジ改善: {s['avg_target_improvement']:+.3f}")
            print(f"  平均全エッジ改善:   {s['avg_all_improvement']:+.3f}")

    # ログを保存
    print("\nログを保存中...")

    # 詳細ログ（JSONL）
    log_file = output_dir / f"strategy_effectiveness_{timestamp}.jsonl"
    with open(log_file, "w", encoding="utf-8") as f:
        for result in results:
            # NumPy型をPython標準型に変換
            serializable_result = json.loads(
                json.dumps(result, default=lambda o: float(o) if isinstance(o, np.floating) else o)
            )
            f.write(json.dumps(serializable_result, ensure_ascii=False) + "\n")
    print(f"詳細ログ保存: {log_file}")

    # 統計サマリー（JSON）
    stats_file = output_dir / f"strategy_effectiveness_{timestamp}_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        serializable_stats = json.loads(
            json.dumps(stats, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        )
        # スキップされたセッション情報を追加
        serializable_stats["skipped_sessions"] = skipped_sessions
        serializable_stats["total_attempted_sessions"] = args.num_sessions
        json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
    print(f"統計サマリー保存: {stats_file}")

    print("\nテスト完了！")


if __name__ == "__main__":
    main()
