from __future__ import annotations

import asyncio
import copy
import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from config import get_config
from context_reward import score_context_alignment
from humans_ollama import human_reply
from network_metrics import BALANCED, UNBALANCED, analyze_relations_from_scores
from relation_scorer import RelationScorer
from utils import filter_logs_by_human_count

_PLANNER_STRATEGIES = ("reframe", "validate", "bridge")

# プラン検証用の定数
_TARGET_EDGES = {"AB", "BC", "CA"}  # 変更対象となるエッジ（ペア関係）
_ALLOWED_STRATEGIES = _PLANNER_STRATEGIES  # 許可される介入戦略
_TARGET_SPEAKERS = {"A", "B", "C"}  # 介入対象となる話者

_STABLE_SIGN_PATTERNS: Dict[str, Dict[str, int]] = {
    "+++": {"AB": +1, "BC": +1, "CA": +1},
    "+--": {"AB": +1, "BC": -1, "CA": -1},
    "-+-": {"AB": -1, "BC": +1, "CA": -1},
    "--+": {"AB": -1, "BC": -1, "CA": +1},
}


def _strip_quotes(text: str) -> str:
    # 先頭・末尾の引用符や括弧を削る
    return text.strip().strip("\"'\u300c\u300d\u300e\u300f()[]（）【】")


def _coerce_bool(value: Any) -> Optional[bool]:
    # Boolean への変換を試みる
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _coerce_int(value: Any) -> Optional[int]:
    # Integer への変換を試みる
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ConversationEnv:
    """ 三者会話環境 """

    def __init__(
        self,
        *,
        max_steps: int,
        personas: Optional[Iterable[str]] = None,
        include_robot: bool = True,
        max_history: int = 12,
        backend: Optional[str] = None,
        decay_factor: float = 1.5,
        debug: bool = False,
        reward_backend: str = "rule",
        evaluation_horizon: int = 2,
        time_penalty: Optional[float] = None,
        terminal_bonus: Optional[float] = None,
        intervention_cost: Optional[float] = None,
    ) -> None:
        cfg = get_config()
        use_ema_cfg = getattr(cfg.scorer, "use_ema", True)
        if use_ema_cfg is None:
            use_ema_cfg = False

        self.max_steps = max(1, int(max_steps))  # 後方互換性のため残す

        # max_rounds が指定されている場合は、それを使用して目標人間発話数を計算
        # max_rounds が指定されていない場合は、max_steps をフォールバックとして使用
        max_rounds_cfg = getattr(cfg.env, "max_rounds", None)
        if max_rounds_cfg is not None:
            self.max_rounds = max(1, int(max_rounds_cfg))
            # 1ラウンド = len(persona_pool) 人間発話（通常3発話）
            # persona_pool はまだ初期化されていないので、デフォルト3を使用
            self.target_human_utterances = self.max_rounds * 3
        else:
            # max_rounds が指定されていない場合は、max_steps から推定
            self.max_rounds = self.max_steps
            self.target_human_utterances = self.max_steps * 3

        self.include_robot = bool(include_robot)
        self.max_history = max(1, int(max_history))  # 介入判定LLM用
        self.max_history_human = max(1, int(getattr(cfg.env, "max_history_human", 12)))  # 人間LLM用
        self.max_history_relation = max(1, int(getattr(cfg.env, "max_history_relation", 3)))  # 関係性LLM用
        self.debug = bool(debug)
        self.reward_backend = (reward_backend or "rule").lower()
        self.evaluation_horizon = max(1, int(evaluation_horizon))
        self.start_relation_check_after_utterances = getattr(cfg.env, "start_relation_check_after_utterances", 3)

        # 報酬パラメータをYAMLから読み込み（引数で指定されていない場合）
        self.time_penalty = float(time_penalty) if time_penalty is not None else float(getattr(cfg.env, "time_penalty", 0.01))
        self.terminal_bonus = float(terminal_bonus) if terminal_bonus is not None else float(getattr(cfg.env, "terminal_bonus", 0.25))
        self.intervention_cost = float(intervention_cost) if intervention_cost is not None else float(getattr(cfg.env, "intervention_cost", 0.02))

        # 新しいパラメータをYAMLから読み込み
        self.min_robot_intervention_lookback = int(getattr(cfg.env, "min_robot_intervention_lookback", 6))
        self.terminal_bonus_duration = int(getattr(cfg.env, "terminal_bonus_duration", 2))

        # ロボット介入履歴のトラッキング
        self.steps_since_last_intervention: int = 0  # 最後の介入からのステップ数
        self.stable_step_start: Optional[int] = None  # 安定状態が開始されたステップ（Noneなら未安定）
        self.terminal_bonus_given: bool = False  # terminal_bonusが既に与えられたかどうか

        # personasの処理（List形式とDict形式の両方をサポート）
        if personas is None:
            # personasが指定されていない場合は、設定から読み込む
            personas_cfg = getattr(cfg.env, "personas", None)
            if personas_cfg and isinstance(personas_cfg, dict):
                self.persona_pool: List[str] = sorted(personas_cfg.keys())
                self.persona_triggers: Dict[str, List[str]] = {
                    name: info.get("triggers", []) if isinstance(info, dict) else []
                    for name, info in personas_cfg.items()
                }
            else:
                self.persona_pool: List[str] = []
                self.persona_triggers: Dict[str, List[str]] = {}
        elif isinstance(personas, dict):
            # Dict形式: {"A": {"triggers": [...]}, "B": {...}, ...}
            self.persona_pool = sorted(personas.keys())
            self.persona_triggers = {
                name: info.get("triggers", []) if isinstance(info, dict) else []
                for name, info in personas.items()
            }
        else:
            # List形式（後方互換性）: ["A", "B", "C"]
            persona_list = [p for p in personas if p]
            self.persona_pool = list(persona_list)
            self.persona_triggers = {}

        # persona_pool が初期化されたので、正確な target_human_utterances を計算
        num_personas = len(self.persona_pool) if self.persona_pool else 3
        self.target_human_utterances = self.max_rounds * num_personas

        self._scorer_kwargs = {
            "backend": backend,
            "use_ema": use_ema_cfg,
            "decay_factor": float(decay_factor),
            "verbose": bool(debug),
        }
        self.scorer = RelationScorer(**self._scorer_kwargs)

        self.logs: List[Dict[str, Any]] = []
        self.episode = 0
        self._episode_step = 0
        self.t = 0
        self.total_steps = 0
        self.used_topics: List[str] = []  # 既に使用したトピックのリスト
        self.current_topic: Optional[str] = None  # 現在のトピック
        self.current_topic_trigger: Optional[str] = None  # 現在のトピックの元になった地雷

        self._last_observation: Optional[str] = None
        self.reset()

    def reset(self) -> str:
        # 環境をリセットして初期状態を返す
        self.episode += 1
        self._episode_step = 0
        self.t = 0
        self.logs = []
        self.scorer = RelationScorer(**self._scorer_kwargs)

        # ロボット介入履歴のリセット
        self.steps_since_last_intervention = 0
        self.stable_step_start = None
        self.terminal_bonus_given = False

        # トピックを生成
        self.current_topic = self._get_topic_suggestion()
        self.used_topics.append(self.current_topic)

        # 初期会話を生成（evaluation_horizon回のターン = 通常1ラウンド）
        self._bootstrap_humans(self.evaluation_horizon)

        # _ensure_unstable_seed()をコメントアウト
        # 初期状態が安定の場合でも追加発話を生成しないようにする
        # 訓練時に不安定な初期状態が必要な場合は、この行を有効化してください
        # self._ensure_unstable_seed()

        observation = self._make_observation()
        self._last_observation = observation
        return observation

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        環境に1ステップ作用を与え、(次状態, 報酬, 終了フラグ, 情報辞書) を返す

        即座報酬の設計（IMMEDIATE_REWARD_DESIGN.mdを参照）:
        1. 人間発話を1回生成
        2. 関係性を評価
        3. 安定なら報酬なしで終了
        4. 不安定なら介入判定
        5. 介入する場合:
           - evaluation_horizon回の人間発話を内部生成（並行処理）
           - 反実仮想シミュレーションも並行実行
           - 安定になったらterminal_bonus_duration回追加
        6. 報酬を計算して即座に返す

        この設計により、PPOが学習可能な即座報酬を実現
        """
        if self.debug:
            print(f"\n🔧 [DEBUG] step() 開始")
            print(f"  現在のログ数: {len(self.logs)}")
            print(f"  _episode_step: {self._episode_step}")
            print(f"  steps_since_last_intervention: {self.steps_since_last_intervention}")

        # スナップショットを保存（人間発話生成前）
        snapshot_logs = [dict(entry) for entry in self.logs]

        # 人間発話を1回生成（条件付き）
        # - ステップ1（_episode_step == 0）: スキップ（reset()で既に生成済み）
        # - 前ステップで介入しなかった（steps_since_last_intervention > 0）: 1発話生成
        # - 前ステップで介入した（steps_since_last_intervention == 0）: スキップ（evaluation_horizon分が既に生成済み）
        should_generate_human = self._episode_step > 0 and self.steps_since_last_intervention > 0

        if should_generate_human:
            if self.debug:
                print(f"  👤 人間発話を1回生成中（会話を進める）...")

            # 1人間発話だけを生成
            replies = human_reply(self.logs, self.persona_pool, topic=self.current_topic, topic_trigger=self.current_topic_trigger, num_speakers=1)
            if replies:
                # 生成された1発話を追加
                self.logs.extend(replies)

                # 関係性を更新
                participants = self._participants(self.logs)
                if participants:
                    human_utterance_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')
                    if human_utterance_count >= self.start_relation_check_after_utterances:
                        filtered_logs = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                        self.scorer.get_scores(filtered_logs, participants, return_trace=False, update_state=True)

                if self.debug:
                    print(f"  生成後のログ数: {len(self.logs)}")
                    last_entry = self.logs[-1]
                    print(f"  最新発話: [{last_entry.get('speaker')}] {last_entry.get('utterance', '')[:50]}...")
        else:
            if self.debug:
                reason = "ステップ1（reset()で既に生成済み）" if self._episode_step == 0 else "前ステップで介入済み（evaluation_horizon分が既に生成済み）"
                print(f"  ⏭️  人間発話生成をスキップ（{reason}）")

        # 関係性を評価（人間発話生成後のログを使用）
        participants = self._participants(self.logs)
        human_utterance_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')
        if human_utterance_count >= self.start_relation_check_after_utterances:
            filtered_logs = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
            rel, trace_scores = self._relation_state(filtered_logs, update_state=True)
        else:
            rel = {}
            trace_scores = []

        metrics, trace_metrics = self._metrics_state(rel, participants)
        is_stable = metrics.get("unstable_triads", 0) == 0 and bool(self.logs)

        if self.debug:
            print(f"  📊 関係性評価:")
            print(f"    不安定トライアド数: {metrics.get('unstable_triads', 0)}")
            print(f"    安定状態: {is_stable}")

        # 安定な場合は報酬なしで終了
        if is_stable:
            if self.debug:
                print(f"  ✅ 安定状態 → 早期リターン（報酬なし）")

            # steps_since_last_interventionを更新（介入しなかったのでインクリメント）
            self.steps_since_last_intervention += 1

            self._episode_step += 1
            self.t += 1
            self.total_steps += 1

            # 終了条件: 人間発話数が目標に達したか確認
            human_utterance_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')
            done = human_utterance_count >= self.target_human_utterances

            next_observation = self._make_observation()
            self._last_observation = next_observation

            info = {
                "plan": None,
                "plan_error": None,
                "intervened": False,
                "balanced": True,
                "robot_utterance": None,
                "replies": [entry for entry in self.logs[len(snapshot_logs):]],
                "rel": metrics,
                "reward_breakdown": {},
                "personas": list(self.persona_pool),
                "next_observation": next_observation,
            }

            return next_observation, 0.0, done, info

        # 不安定な場合は介入判定
        if self.debug:
            print(f"  ❌ 不安定状態 → 介入判定を実行")

        plan, plan_error, direct_utterance = self._parse_plan(action)

        if self.debug:
            print(f"  📋 アクション解析:")
            if plan:
                print(f"    intervene_now: {plan.get('intervene_now', False)}")
                if plan.get('intervene_now'):
                    print(f"    edge_to_change: {plan.get('edge_to_change')}")
                    print(f"    strategy: {plan.get('strategy')}")
                    print(f"    target_speaker: {plan.get('target_speaker')}")
            if plan_error:
                print(f"    ⚠️ プランエラー: {plan_error}")

        # 介入判定
        robot_entry: Optional[Dict[str, Any]] = None
        intervened = False
        if plan and plan.get("intervene_now"):
            if self.debug:
                print(f"  🤖 ロボット介入を実行")
            robot_entry = self._render_intervention(plan, simulate=False)
            intervened = True
            if self.debug:
                print(f"    発話内容: {robot_entry.get('utterance', '')[:80]}...")
        elif direct_utterance:
            robot_entry = {"speaker": "ロボット", "utterance": _strip_quotes(direct_utterance)}
            intervened = True
        else:
            if self.debug:
                print(f"  ⏭️  介入しない選択")

        # 報酬の初期化
        reward = 0.0
        reward_breakdown: Dict[str, float] = {}
        actual_u_flip = None
        counterfactual_u_flip = None
        delta = None

        if not intervened:
            # Case A: 介入しない場合
            # タイムペナルティのみ即座に付与
            if self.debug:
                print(f"  💸 タイムペナルティを付与: {-self.time_penalty:.4f}")
            reward = -self.time_penalty
            reward_breakdown["time_penalty"] = -self.time_penalty

            self.steps_since_last_intervention += 1

        else:
            # Case B: 介入する場合
            # ロボット発話を追加
            self.logs.append(robot_entry)

            if self.debug:
                print(f"  🔄 並行シミュレーション開始")
                print(f"    - 実世界: evaluation_horizon={self.evaluation_horizon}回の人間発話")
                print(f"    - 反実仮想: 介入しなかった場合をシミュレーション")

            # 並行処理: 実世界と反実仮想シミュレーションを同時実行
            async def run_parallel_simulation():
                # スナップショット（ロボット発話前）
                snapshot = [dict(entry) for entry in snapshot_logs]

                # 実世界: ロボット発話後、evaluation_horizon回の人間発話
                # （self.logsを直接使用せず、新しいスコアラーで管理）
                actual_scorer = RelationScorer(**self._scorer_kwargs)
                actual_logs_task = self._bootstrap_humans_async(
                    self.logs.copy(),
                    self.evaluation_horizon,
                    actual_scorer
                )

                # 反実仮想: 介入しなかった場合（スナップショットから）
                counterfactual_task = self._simulate_forward_async(
                    snapshot,
                    self.evaluation_horizon,
                    plan=None
                )

                # 両方の完了を待つ
                actual_logs, (cf_u_flip, _) = await asyncio.gather(
                    actual_logs_task,
                    counterfactual_task
                )

                return actual_logs, cf_u_flip

            # asyncio.run()で同期的に実行
            actual_logs, counterfactual_u_flip = asyncio.run(run_parallel_simulation())

            if self.debug:
                print(f"  ✅ 並行シミュレーション完了")
                print(f"    反実仮想不安定度: {counterfactual_u_flip:.4f}")

            # 実世界のログを環境に反映
            self.logs = actual_logs

            if self.debug:
                print(f"  📝 実世界の会話ログを更新（{len(self.logs)}発話）")

            # evaluation_horizon後の関係性を評価
            participants_after = self._participants(self.logs)
            human_utterance_count_after = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

            if human_utterance_count_after >= self.start_relation_check_after_utterances:
                filtered_logs_after = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                rel_after, _ = self._relation_state(filtered_logs_after, update_state=True)
                actual_u_flip = self._compute_u_flip_from_scores(rel_after)
            else:
                rel_after = {}
                actual_u_flip = 1.0

            metrics_after, _ = self._metrics_state(rel_after, participants_after)
            is_stable_after = metrics_after.get("unstable_triads", 0) == 0

            if self.debug:
                print(f"  📊 evaluation_horizon後の関係性:")
                print(f"    実際の不安定度: {actual_u_flip:.4f}")
                print(f"    安定状態: {is_stable_after}")

            # 報酬計算
            delta = counterfactual_u_flip - actual_u_flip
            reward = delta - self.intervention_cost - self.time_penalty
            reward_breakdown["delta_u_flip"] = delta
            reward_breakdown["counterfactual_u_flip"] = counterfactual_u_flip
            reward_breakdown["actual_u_flip"] = actual_u_flip
            reward_breakdown["intervention_cost"] = -self.intervention_cost
            reward_breakdown["time_penalty"] = -self.time_penalty

            if self.debug:
                print(f"  💰 報酬計算:")
                print(f"    関係性改善効果: {delta:.4f}")
                print(f"    介入コスト: {-self.intervention_cost:.4f}")
                print(f"    タイムペナルティ: {-self.time_penalty:.4f}")
                print(f"    小計: {reward:.4f}")

            # 安定になった場合、terminal_bonus_durationチェック
            if is_stable_after:
                if self.debug:
                    print(f"  🎯 安定達成 → terminal_bonusチェック開始")
                    print(f"    追加で{self.terminal_bonus_duration}人間発話分の安定性を確認")

                # terminal_bonus_duration人間発話を生成（実際には1ラウンド = 3人間発話が最小単位）
                self._bootstrap_humans(self.terminal_bonus_duration)

                # 最後の関係性を再評価
                participants_check = self._participants(self.logs)
                human_count_check = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

                stability_maintained = True
                if human_count_check >= self.start_relation_check_after_utterances:
                    filtered_check = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                    rel_check, _ = self._relation_state(filtered_check, update_state=True)
                    metrics_check, _ = self._metrics_state(rel_check, participants_check)

                    if metrics_check.get("unstable_triads", 0) > 0:
                        # 不安定に戻った
                        stability_maintained = False
                        if self.debug:
                            print(f"    ❌ 不安定に戻った")
                    else:
                        if self.debug:
                            print(f"    ✅ 安定維持")

                # terminal_bonus_duration人間発話後も安定が続いた場合
                if stability_maintained:
                    if self.debug:
                        print(f"  🎁 安定が持続 → terminal_bonus付与: +{self.terminal_bonus:.4f}")
                    reward += self.terminal_bonus
                    reward_breakdown["terminal_bonus"] = self.terminal_bonus
                elif self.debug:
                    print(f"  ⚠️  安定が持続せず → terminal_bonusなし")

            self.steps_since_last_intervention = 0

        # ステップカウンタを更新
        self._episode_step += 1
        self.t += 1
        self.total_steps += 1

        # 最終的な関係性とobservation
        final_participants = self._participants(self.logs)
        final_human_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

        # 終了条件: 人間発話数が目標に達したか確認
        done = final_human_count >= self.target_human_utterances

        if final_human_count >= self.start_relation_check_after_utterances:
            final_filtered = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
            final_rel, _ = self._relation_state(final_filtered, update_state=True)
        else:
            final_rel = {}

        final_metrics, _ = self._metrics_state(final_rel, final_participants)
        final_balanced = final_metrics.get("unstable_triads", 0) == 0

        next_observation = self._make_observation()
        self._last_observation = next_observation

        if self.debug:
            print(f"  🏁 step()完了")
            print(f"    最終報酬: {reward:.4f}")
            print(f"    done: {done}")
            print(f"    総ログ数: {len(self.logs)}")

        # info辞書
        info: Dict[str, Any] = {
            "plan": plan,
            "plan_error": plan_error,
            "intervened": intervened,
            "balanced": final_balanced,
            "robot_utterance": robot_entry["utterance"] if robot_entry else None,
            "replies": [entry for entry in self.logs[len(snapshot_logs):]],
            "rel": final_metrics,
            "reward_breakdown": reward_breakdown,
            "counterfactual_u_flip": counterfactual_u_flip,
            "actual_u_flip": actual_u_flip,
            "delta_u_flip": delta,
            "personas": list(self.persona_pool),
            "next_observation": next_observation,
        }

        # 介入した場合、evaluation_horizon後の関係性を追加
        if intervened and 'metrics_after' in locals():
            info["rel_after_horizon"] = metrics_after
            info["stable_after_horizon"] = is_stable_after

        return next_observation, reward, done, info

    def _get_topic_suggestion(self) -> str:
        """LLMで話題を生成し、短いトピック文字列を返す。"""
        import random
        cfg = get_config()
        topic_cfg = getattr(cfg, "topic_manager", None)
        if not topic_cfg or not getattr(topic_cfg, "enable", False):
            # トピック機能が無効の場合はデフォルトトピックを返す
            self.current_topic_trigger = None
            return "自由な話題"

        system = getattr(topic_cfg, "generation_prompt", "")

        # persona_triggersから trigger_examples を生成
        # ランダムに1つだけ選択
        selected_trigger = None
        if self.persona_triggers and "{trigger_examples}" in system:
            # 全personaのtriggersを平坦化（重複を除く）
            all_triggers = []
            for triggers in self.persona_triggers.values():
                all_triggers.extend(triggers)
            # 重複を除去
            all_triggers = list(set(all_triggers))

            if all_triggers:
                # ランダムに1つだけ選択
                selected_trigger = random.choice(all_triggers)
                trigger_examples = selected_trigger
                system = system.replace("{trigger_examples}", trigger_examples)
            else:
                # triggersが空の場合はプレースホルダーを削除
                system = system.replace("{trigger_examples}", "様々なテーマ")

        # 選択された地雷を保存（人間LLMで使用するため）
        self.current_topic_trigger = selected_trigger

        used_str = "\n".join(f"- {t}" for t in self.used_topics) if self.used_topics else "(なし)"
        prompt = f"[既に提案した話題]\n{used_str}"

        llm_cfg = getattr(cfg, "llm", None)
        client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="topic")
        max_attempts = getattr(cfg.llm, "max_attempts", 5) or 5
        base_backoff = getattr(cfg.llm, "base_backoff", 0.5) or 0.5
        
        if client and deployment:
            messages = [{"role":"system","content":system},
                        {"role":"user","content":prompt}]
            for attempt in range(1, max_attempts + 1):
                try:
                    # GPT-5の場合はreasoningパラメータを追加
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
                    if self.debug:
                        print(f"[topic] attempt {attempt} failed:", exc)
                    if attempt < max_attempts:
                        import time
                        time.sleep(base_backoff * (2 ** (attempt - 1)))
        
        # フォールバック
        return "最近の出来事について"

    def planning_context(self) -> Dict[str, Any]:
        # 関係安定化のための介入計画を立案する
        weights = self._edge_weights(self.logs)
        u_flip, stable_sign, distances = self._compute_u_flip(weights)
        target_edge = "AB"
        if distances:
            target_edge = max(distances.items(), key=lambda kv: kv[1])[0]
        return {
            "scores": weights,
            "u_flip": u_flip,
            "stable_sign": stable_sign,
            "distances": distances,
            "target_edge": target_edge,
            "evaluation_horizon": self.evaluation_horizon,
            "time_penalty": self.time_penalty,
            "intervention_cost": self.intervention_cost,
            "balanced": self._is_balanced(),
        }

    def _make_observation(self) -> str:
        context = self.planning_context()
        # max_history個の人間発話 + その間のロボット発話を取得
        filtered_logs = filter_logs_by_human_count(self.logs, self.max_history)
        history_lines = [f"[{item.get('speaker', '?')}] {item.get('utterance', '').strip()}" for item in filtered_logs]
        if not history_lines:
            history_lines = ["(履歴なし)"]

        scores = context.get("scores", {}) or {"AB": 0.0, "BC": 0.0, "CA": 0.0}
        stable_sign = context.get("stable_sign", "???")
        target_edge = context.get("target_edge", "AB")
        eval_horizon = context.get("evaluation_horizon", self.evaluation_horizon)
        time_penalty = context.get("time_penalty", self.time_penalty)
        intervention_cost = context.get("intervention_cost", self.intervention_cost)

        prompt_lines = [
            "三者会話（話者 A/B/C）の関係を安定化するため、ロボットが適切なタイミングで一言介入します。",
            "あなたの役割は、現在の会話履歴と各ペアの関係スコア（-1..1）を受け取り、"
            "「できるだけ早く関係性を安定状態（+++,+--,-+-,--+）にする」ための介入方法（今介入すべきか・対象リンク・発話先・介入タイプ）を提案することです。",
            # "介入しなくても自動的に安定状態になりそうな場合には介入しない選択をすることも有効です。"
            "※ロボットの実際の発話文は別LLMが生成します。あなたは介入方法だけを出力します。",
            "",
            "制約:",
            "- 出力は JSON のみ。説明や装飾は禁止。",
            "- intervene_now は true|false（今すぐ介入すべきか）。",
            "- edge_to_change は \"AB\" | \"BC\" | \"CA\" のいずれか。",
            "- strategy は " + " | ".join(_PLANNER_STRATEGIES) + " から選択。",
            "  - reframe: 否定的状況を肯定的視点から再解釈、認知を転換",
            "  - validate: 対象者の感情・意見を承認し、心理的安全性を構築",
            "  - bridge: 対立する者の共通点・目標を明示し、協力関係を構築",
            "  ※どの戦略をいつ使うかは会話文脈や関係スコアから判断してください。",
            "- target_speaker は A | B | C のいずれか。（その介入を誰に向けるか）",
            "- intervene_now=true の場合は edge_to_change / strategy / target_speaker を必ず指定。",
            "- intervene_now=false の場合は何も指定しない。",
            "",
            "履歴:",
            *history_lines,
            "",
            "現在の関係スコア（-1..1）: " + ", ".join(f"w_{edge}={value:+.2f}" for edge, value in scores.items()),
            # f"推定安定パターン: {stable_sign}",
            # f"推奨リンク候補: {target_edge}",
            # f"評価窓 H_eval: {eval_horizon}",
            # f"時間ペナルティ α: {time_penalty:.3f}",
            # f"介入コスト c_act: {intervention_cost:.3f}",
            "",
            "出力例（JSONのみ）:",
            '{"intervene_now": true, "edge_to_change": "AB", "strategy": "reframe", "target_speaker": "A"}',
            '{"intervene_now": false}',
        ]
        return "\n".join(prompt_lines)

    def relation_snapshot(self) -> Dict[str, Any]:
        # 現在の関係スコアとメトリクスのスナップショットを返す
        participants = self._participants(self.logs)
        scores = self.scorer.get_scores(self.logs, participants, return_trace=False, update_state=False)
        metrics = analyze_relations_from_scores(scores, include_nodes=participants)
        u_flip, stable_sign, distances = self._compute_u_flip(self._edge_weights_from_scores(scores))
        return {
            "scores": scores,
            "metrics": metrics,
            "u_flip": u_flip,
            "stable_sign": stable_sign,
            "distances": distances,
        }

    def _parse_plan(self, action: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        # 先頭・末尾の引用符や括弧を削る
        if action is None:
            return None, "empty_action", None
        text = action.strip()
        if not text:
            return None, "empty_action", None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None, "not_json", text

        if not isinstance(payload, dict):
            return None, "plan_not_object", text

        plan: Dict[str, Any] = {}
        intervene_now = _coerce_bool(payload.get("intervene_now"))
        if intervene_now is None:
            intervene_now = bool(payload.get("intervene_now"))
        plan["intervene_now"] = bool(intervene_now)

        edge_raw = str(payload.get("edge_to_change", "AB")).upper().strip()
        if edge_raw not in {"AB", "BC", "CA"}:
            edge_raw = "AB"
        plan["edge_to_change"] = edge_raw

        strategy_raw = str(payload.get("strategy", "reframe")).strip()
        if strategy_raw not in _PLANNER_STRATEGIES:
            strategy_raw = "reframe"
        plan["strategy"] = strategy_raw

        target_raw = str(payload.get("target_speaker", "A")).upper().strip()
        if target_raw not in {"A", "B", "C"}:
            target_raw = "A"
        plan["target_speaker"] = target_raw

        recheck_raw = _coerce_int(payload.get("recheck_after_turns", self.evaluation_horizon))
        if recheck_raw is None or recheck_raw <= 0:
            recheck_raw = self.evaluation_horizon
        plan["recheck_after_turns"] = max(self.evaluation_horizon, recheck_raw)

        return plan, None, None

    def _render_intervention(self, plan: Dict[str, Any], *, simulate: bool) -> Dict[str, Any]:
        # 介入発話を生成する
        labels = self._human_labels()
        target_char = plan.get("target_speaker", "A")
        edge_to_change = plan.get("edge_to_change", "AB")
        partner_char = next((ch for ch in edge_to_change if ch != target_char), "B")

        try:
            target_idx = "ABC".index(target_char)
            target_name = labels[target_idx]
        except ValueError:
            target_name = target_char
        try:
            partner_idx = "ABC".index(partner_char)
            partner_name = labels[partner_idx]
        except ValueError:
            partner_name = partner_char

        strategy = plan.get("strategy")

        # max_history個の人間発話 + その間のロボット発話を取得
        filtered_logs = filter_logs_by_human_count(self.logs, self.max_history)
        history_lines = [
            f"[{entry.get('speaker', '?')}] {entry.get('utterance', '').strip()}" for entry in filtered_logs
        ]
        history_text = "\n".join(history_lines) if history_lines else "(履歴なし)"

        directive_map = {
            "reframe": f"直前の{target_name}さんや他の参加者の発言から、否定的な側面ではなく肯定的な意図や建設的な視点を見出し、それを一文で伝えてください。対立を『視点の違い』として価値あるものに再定義してください。",
            "validate": f"{target_name}さんの感情や意見を明確に承認・共感し、その価値を認めてください。心理的安全性を作り、建設的な対話を可能にする一文を述べてください。",
            "bridge": f"{target_name}さんと{partner_name}さんの間に共通点・共通の目標・相互依存性を見出してください。対立ではなく協力関係として捉え直せる視点を一文で提示してください。",
        }
        directive = directive_map.get(strategy, directive_map["reframe"])

        fallback_text = self._fallback_intervention_text(target_name, partner_name, strategy)
        if simulate:
            return {"speaker": "ロボット", "utterance": fallback_text}

        cfg = get_config()
        llm_cfg = getattr(cfg, "llm", None)
        client, deployment = get_azure_chat_completion_client(llm_cfg, model_type="robot")
        max_attempts = getattr(cfg.llm, "max_attempts", 5) or 5
        base_backoff = getattr(cfg.llm, "base_backoff", 0.5) or 0.5
        if client and deployment:
            user_payload = (
                "あなたは関係性を安定させるロボットの発話生成器です。出力は日本語で一文のみ。話者ラベルや括弧は使わない。\n"
                "会話履歴を参考にしながら、以下の指示に従って発話を生成してください。\n\n"
                f"指示: {directive}\n"
                f"会話履歴:\n{history_text}\n\n"
                "必須条件:\n"
                f"- {target_name} さんの名前を一度だけ入れる。\n"
                "- 直前の話題を自然に引き継ぐ。\n"
                "- 一文のみ (60字前後)。\n"
            )
            messages = [
                {"role": "system", "content": "あなたは戦略を元に、介入ロボットの発話内容を作成します。常に日本語の一文だけを返します。"},
                {"role": "user", "content": user_payload},
            ]
            for attempt in range(1, max_attempts + 1):
                try:
                    # GPT-5の場合はreasoningパラメータを追加
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
                            return {"speaker": "ロボット", "utterance": cleaned}
                except Exception as exc:
                    if getattr(_CFG.env, "debug", False):
                        print(f"[robot_utterance] attempt {attempt} failed:", exc)
                    if attempt < max_attempts:
                        time.sleep(base_backoff * (2 ** (attempt - 1)))
                    else:
                        if getattr(_CFG.env, "debug", False):
                            print("[robot_utterance] all attempts failed, falling back to local heuristic")
        return {"speaker": "ロボット", "utterance": fallback_text}

    def _fallback_intervention_text(self, target_name: str, partner_name: str, strategy: str) -> str:
        # フォールバック用の介入発話テンプレート
        templates = {
            "reframe": f"{target_name}さん、その視点には建設的な意図があると思います。",
            "validate": f"{target_name}さん、あなたの意見はとても大切です。もっと聞かせてください。",
            "bridge": f"{target_name}さん、{partner_name}さん、お二人には共通の目標があると思います。",
        }
        return templates.get(strategy, templates["reframe"])

    def _bootstrap_humans(self, target_turns: int) -> None:
        # 人間参加者の発話を追加して会話を進める
        turns_added = 0
        safety_limit = max(1, self.max_steps * max(1, len(self.persona_pool)) * 3)
        while turns_added < target_turns and turns_added < safety_limit:
            # 1人間発話ずつ生成
            replies = human_reply(self.logs, self.persona_pool, topic=self.current_topic, topic_trigger=self.current_topic_trigger, num_speakers=1)
            if not replies:
                break
            self.logs.extend(replies)
            participants = self._participants(self.logs)
            if participants:
                # 3発話後から関係性推定を更新
                human_utterance_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')
                if human_utterance_count >= self.start_relation_check_after_utterances:
                    # 関係性LLMにはロボット発話を除外して渡す
                    filtered_logs = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                    self.scorer.get_scores(filtered_logs, participants, return_trace=False, update_state=True)
            turns_added += 1

    async def _bootstrap_humans_async(
        self,
        logs: List[Dict[str, Any]],
        target_turns: int,
        scorer: RelationScorer,
    ) -> List[Dict[str, Any]]:
        """
        人間参加者の発話を非同期で追加（シミュレーション用）

        Args:
            logs: 会話ログ（変更されない、コピーして使用）
            target_turns: 生成する人間発話の回数
            scorer: 関係性スコアラー

        Returns:
            更新された会話ログ
        """
        logs_copy = copy.deepcopy(logs)
        turns_added = 0
        safety_limit = max(1, self.max_steps * max(1, len(self.persona_pool)))

        while turns_added < target_turns and turns_added < safety_limit:
            # human_replyは同期関数なので、asyncio.to_thread()でラップ
            # 1人間発話ずつ生成
            replies = await asyncio.to_thread(
                human_reply,
                logs_copy,
                self.persona_pool,
                topic=self.current_topic,
                topic_trigger=self.current_topic_trigger,
                num_speakers=1
            )
            if not replies:
                break
            logs_copy.extend(replies)
            participants = self._participants(logs_copy)
            if participants:
                human_utterance_count = sum(1 for log in logs_copy if log.get('speaker') != 'ロボット')
                if human_utterance_count >= self.start_relation_check_after_utterances:
                    filtered_logs = filter_logs_by_human_count(logs_copy, self.max_history_relation, exclude_robot=True)
                    # get_scoresも同期関数なので、asyncio.to_thread()でラップ
                    await asyncio.to_thread(
                        scorer.get_scores,
                        filtered_logs,
                        participants,
                        return_trace=False,
                        update_state=True
                    )
            turns_added += len([r for r in replies if r.get("speaker") != "ロボット"])

        return logs_copy

    def _ensure_unstable_seed(self) -> None:
        # 初期状態が不安定になるまで人間発話を追加する
        attempts = 0
        while self._is_balanced() and attempts < self.max_steps:
            self._bootstrap_humans(self.evaluation_horizon)
            attempts += 1

    def _participants(self, logs: List[Dict[str, Any]]) -> List[str]:
        # 発話者リストを取得する（ロボット除外オプション付き）
        names = {entry.get("speaker") for entry in logs if entry.get("speaker") and entry.get("speaker") != "ロボット"}
        return sorted(names)

    def _human_labels(self) -> List[str]:
        # 人間参加者のラベルリストを取得する
        labels = [name for name in self.persona_pool if name]
        while len(labels) < 3:
            labels.append(chr(ord("A") + len(labels)))
        return labels[:3]

    def _edge_weights(self, logs: List[Dict[str, Any]]) -> Dict[str, float]:
        # エッジの重みを計算する
        participants = self._participants(logs)
        scores = self.scorer.get_scores(logs, participants, return_trace=False, update_state=False)
        return self._edge_weights_from_scores(scores)

    def _edge_weights_from_scores(self, scores: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        # スコアからエッジの重みを計算する
        labels = self._human_labels()
        key_map = {
            "AB": tuple(sorted((labels[0], labels[1]))),
            "BC": tuple(sorted((labels[1], labels[2]))),
            "CA": tuple(sorted((labels[2], labels[0]))),
        }
        return {edge: float(scores.get(pair, 0.0)) for edge, pair in key_map.items()}

    def _relation_state(self, logs: List[Dict[str, Any]], *, update_state: bool) -> Tuple[Dict[Tuple[str, str], float], List[str]]:
        # 関係スコアの状態を取得する
        participants = self._participants(logs)
        scores, trace = self.scorer.get_scores(logs, participants, return_trace=True, update_state=update_state)
        return scores, trace

    def _metrics_state(self, scores: Dict[Tuple[str, str], float], participants: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        # 関係メトリクスの状態を取得する
        metrics, trace = analyze_relations_from_scores(scores, include_nodes=participants, verbose=True, return_trace=True)
        return metrics, trace

    def _compute_u_flip(self, weights: Dict[str, float]) -> Tuple[float, str, Dict[str, float]]:
        # 不安定度 u_flip と最適安定パターンを計算する
        best_total = math.inf
        best_sign = "+++"
        best_distances: Dict[str, float] = {edge: 0.0 for edge in weights}
        for sign, mapping in _STABLE_SIGN_PATTERNS.items():
            total = 0.0
            distances: Dict[str, float] = {}
            for edge, direction in mapping.items():
                weight = weights.get(edge, 0.0)
                distance = max(0.0, -direction * weight)
                distances[edge] = distance
                total += distance
            if total < best_total:
                best_total = total
                best_sign = sign
                best_distances = distances
        if not math.isfinite(best_total):
            best_total = 0.0
        return best_total, best_sign, best_distances

    def _compute_u_flip_from_scores(self, scores: Dict[Tuple[str, str], float]) -> float:
        weights = self._edge_weights_from_scores(scores)
        u_flip, _, _ = self._compute_u_flip(weights)
        return u_flip

    def _simulate_forward(
        self,
        snapshot: List[Dict[str, Any]],
        human_turns: int,
        plan: Optional[Dict[str, Any]],
    ) -> float:
        """
        将来の人間発話をシミュレートして不安定度 u_flip を計算する
        
        遅延報酬の設計:
        - ロボットが介入する場合: ロボット発話 + human_turns回の人間発話後の関係性
        - ロボットが介入しない場合: human_turns回の人間発話後の関係性
        
        Args:
            snapshot: 現在の会話ログのスナップショット
            human_turns: シミュレートする人間発話の回数（evaluation_horizon）
            plan: ロボットの介入プラン（Noneの場合は介入なし）
        
        Returns:
            u_flip: シミュレート後の不安定度
        """
        logs = copy.deepcopy(snapshot)
        scorer = RelationScorer(**self._scorer_kwargs)

        # ロボットが介入する場合、ロボット発話を追加
        if plan and plan.get("intervene_now"):
            logs.append(self._render_intervention(plan, simulate=True))

        # human_turns回の人間発話を追加
        turns_added = 0
        safety_limit = max(1, self.max_steps * max(1, len(self.persona_pool)))
        while turns_added < human_turns and turns_added < safety_limit:
            # 1人間発話ずつ生成
            replies = human_reply(logs, self.persona_pool, topic=self.current_topic, topic_trigger=self.current_topic_trigger, num_speakers=1)
            if not replies:
                break
            logs.extend(replies)
            participants = self._participants(logs)
            if participants:
                # 関係性推定を3発話後から行う
                human_utterance_count = sum(1 for log in logs if log.get('speaker') != 'ロボット')
                if human_utterance_count >= self.start_relation_check_after_utterances:
                    # 関係性LLMにはロボット発話を除外して渡す
                    filtered_logs = filter_logs_by_human_count(logs, self.max_history_relation, exclude_robot=True)
                    scorer.get_scores(filtered_logs, participants, return_trace=False, update_state=True)
            turns_added += len([r for r in replies if r.get("speaker") != "ロボット"])

        # human_turns回後の関係性を計算
        participants = self._participants(logs)
        human_utterance_count = sum(1 for log in logs if log.get('speaker') != 'ロボット')
        if human_utterance_count >= self.start_relation_check_after_utterances and participants:
            # 関係性LLMにはロボット発話を除外して渡す
            filtered_logs = filter_logs_by_human_count(logs, self.max_history_relation, exclude_robot=True)
            scores = scorer.get_scores(filtered_logs, participants, return_trace=False, update_state=False)
            weights = self._edge_weights_from_scores(scores)
            u_flip, _, _ = self._compute_u_flip(weights)
        else:
            # 3発話未満の場合は不安定度を最大値（1.0）に設定
            u_flip = 1.0

        return u_flip

    async def _simulate_forward_async(
        self,
        snapshot: List[Dict[str, Any]],
        human_turns: int,
        plan: Optional[Dict[str, Any]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        将来の人間発話を非同期でシミュレートして不安定度 u_flip を計算する

        Args:
            snapshot: 現在の会話ログのスナップショット
            human_turns: シミュレートする人間発話の回数（evaluation_horizon）
            plan: ロボットの介入プラン（Noneの場合は介入なし）

        Returns:
            (u_flip, logs): シミュレート後の不安定度と更新された会話ログ
        """
        logs = copy.deepcopy(snapshot)
        scorer = RelationScorer(**self._scorer_kwargs)

        # ロボットが介入する場合、ロボット発話を追加
        if plan and plan.get("intervene_now"):
            logs.append(self._render_intervention(plan, simulate=True))

        # human_turns回の人間発話を非同期で追加
        logs = await self._bootstrap_humans_async(logs, human_turns, scorer)

        # human_turns回後の関係性を計算
        participants = self._participants(logs)
        human_utterance_count = sum(1 for log in logs if log.get('speaker') != 'ロボット')
        if human_utterance_count >= self.start_relation_check_after_utterances and participants:
            # 関係性LLMにはロボット発話を除外して渡す
            filtered_logs = filter_logs_by_human_count(logs, self.max_history_relation, exclude_robot=True)
            # get_scoresを非同期で実行
            scores = await asyncio.to_thread(
                scorer.get_scores,
                filtered_logs,
                participants,
                return_trace=False,
                update_state=False
            )
            weights = self._edge_weights_from_scores(scores)
            u_flip, _, _ = self._compute_u_flip(weights)
        else:
            # 3発話未満の場合は不安定度を最大値（1.0）に設定
            u_flip = 1.0

        return u_flip, logs

    def _is_balanced(self) -> bool:
        # 現在の会話ログに基づき、関係が安定状態かどうかを判定
        snapshot = self.relation_snapshot()
        triangles = snapshot.get("triangles", {})
        if not triangles:
            return False
        return all(status == "S" for status in triangles.values())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def _validate_plan(self, plan: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        # 介入プランの妥当性を検証
        if not isinstance(plan, dict):
            return False, "plan_not_dict"
        if "intervene_now" not in plan:
            return False, "missing_intervene_now"
        try:
            recheck = int(plan.get("recheck_after_turns", self.evaluation_horizon))
        except Exception:
            return False, "invalid_recheck"
        if recheck <= 0:
            return False, "recheck_non_positive"
        if recheck < self.evaluation_horizon:
            return False, "recheck_too_short"

        intervene_now = bool(plan.get("intervene_now"))
        if intervene_now:
            edge = plan.get("edge_to_change")
            strategy = plan.get("strategy")
            target = plan.get("target_speaker")
            if edge not in _TARGET_EDGES:
                return False, "invalid_edge"
            if strategy not in _ALLOWED_STRATEGIES:
                return False, "invalid_strategy"
            if target not in _TARGET_SPEAKERS:
                return False, "invalid_target"
        return True, None

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def relation_snapshot(self) -> Dict[str, Any]:
        # 現在の会話ログに基づき、関係スナップショットを取得
        participants = self._participants(self.logs)
        scores = self.scorer.get_scores(self.logs, participants, return_trace=False, update_state=False)
        relations = analyze_relations_from_scores(scores, include_nodes=participants, verbose=self.debug)
        triangles = {}
        for tri in relations.get("triangles", []):
            nodes = tuple(sorted(tri.get("nodes", [])))
            if any(node == "ロボット" for node in nodes):
                continue
            struct = tri.get("struct")
            if struct in BALANCED:
                status = "S"
            elif struct in UNBALANCED:
                status = "U"
            else:
                status = "?"
            triangles[nodes] = status
        return {
            "participants": [p for p in participants if p != "ロボット"],
            "triangles": triangles,
            "scores": scores,
        }
