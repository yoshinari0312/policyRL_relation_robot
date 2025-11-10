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

_PLANNER_STRATEGIES = ("plan", "validate", "bridge")

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


def select_target_edge_and_speaker(
    edges: Dict[Any, float],
    debug: bool = False,
    debug_prefix: str = ""
) -> Tuple[str, str]:
    """
    関係性エッジから改善対象エッジとターゲット話者を選択する。
    
    ロジック:
    1. 絶対値が最小（最も0に近い）負のエッジを選択
    2. 負のエッジがない場合は最小スコアのエッジを選択
    3. ターゲット話者はエッジの2人からランダムに選択
    
    Args:
        edges: エッジ辞書 {(A,B): score, ...} または {"AB": score, ...}
        debug: デバッグ出力を有効にするか
        debug_prefix: デバッグメッセージのプレフィックス
        
    Returns:
        (edge_to_change, target_speaker) のタプル
        edge_to_change は正規化済み ("AB", "BC", "CA" のいずれか)
    """
    import random
    
    edge_to_change = None
    
    if debug:
        print(f"{debug_prefix} edges取得: {edges}")
    
    # 負のエッジのみを抽出し、絶対値でソート
    negative_edges = []
    for edge_key, score in edges.items():
        if score < 0:
            # edge_keyが文字列でない場合は変換
            if isinstance(edge_key, tuple):
                edge_str = "".join(str(c) for c in edge_key)
            else:
                edge_str = str(edge_key)
            negative_edges.append((edge_str, score))
            if debug:
                print(f"{debug_prefix} 負エッジ追加: {edge_str} = {score} (絶対値: {abs(score)})")
    
    if negative_edges:
        # 絶対値が最小（最も0に近い）負のエッジを選択
        negative_edges.sort(key=lambda x: abs(x[1]))
        edge_to_change = negative_edges[0][0]
        if debug:
            print(f"{debug_prefix} ソート後: {negative_edges}")
            print(f"{debug_prefix} 選択: {edge_to_change} (スコア: {negative_edges[0][1]})")
    else:
        # 負のエッジがない場合は、最小スコアのエッジを選択
        if edges:
            min_edge = min(edges.items(), key=lambda x: x[1])
            if isinstance(min_edge[0], tuple):
                edge_to_change = "".join(str(c) for c in min_edge[0])
            else:
                edge_to_change = str(min_edge[0])
            if debug:
                print(f"{debug_prefix} 負エッジなし、最小選択: {edge_to_change}")
    
    
    # edge_to_changeを正規化
    edge_before_norm = edge_to_change
    edge_to_change = edge_to_change.replace(",", "").replace(" ", "").upper()
    if edge_to_change not in {"AB", "BC", "CA"}:
        if len(edge_to_change) == 2 and edge_to_change[1] + edge_to_change[0] in {"AB", "BC", "CA"}:
            edge_to_change = edge_to_change[1] + edge_to_change[0]
            if debug:
                print(f"{debug_prefix} 反転: {edge_before_norm} → {edge_to_change}")
        else:
            edge_to_change = "AB"
            if debug:
                print(f"{debug_prefix} 不正なエッジ、ABに修正: {edge_before_norm}")
    else:
        if debug:
            print(f"{debug_prefix} 正規化: {edge_before_norm} → {edge_to_change}")
    
    # target_speaker: edge_to_changeの2人からランダムに選択
    target_speaker = random.choice(list(edge_to_change))
    
    return edge_to_change, target_speaker


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
        max_auto_skip: Optional[int] = None,
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
        
        # 後方互換性: max_historyが指定されていればそれを使用、なければ新しいパラメータを使用
        intervention_max_history_cfg = getattr(cfg.env, "intervention_max_history", None)
        robot_max_history_cfg = getattr(cfg.env, "robot_max_history", None)
        
        if intervention_max_history_cfg is not None:
            self.intervention_max_history = max(1, int(intervention_max_history_cfg))
        else:
            # フォールバック: max_history または引数から
            self.intervention_max_history = max(1, int(getattr(cfg.env, "max_history", max_history)))
        
        if robot_max_history_cfg is not None:
            self.robot_max_history = max(1, int(robot_max_history_cfg))
        else:
            # フォールバック: max_history または引数から
            self.robot_max_history = max(1, int(getattr(cfg.env, "max_history", max_history)))
        
        # 後方互換性のため、max_historyも残す（非推奨）
        self.max_history = self.intervention_max_history  # デフォルトは介入判定用と同じ
        
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

        # 新しい報酬パラメータ（感情ニーズベース）
        self.stable_bonus = float(getattr(cfg.env, "stable_bonus", 2.0))
        self.preference_match_bonus = float(getattr(cfg.env, "preference_match_bonus", 0.5))

        # その他のパラメータをYAMLから読み込み
        self.min_robot_intervention_lookback = int(getattr(cfg.env, "min_robot_intervention_lookback", 6))
        self.terminal_bonus_duration = int(getattr(cfg.env, "terminal_bonus_duration", 2))

        # auto-skip の最大試行回数（テストから参照するためインスタンス属性として保持）
        # コンストラクタ引数で指定されていなければ設定ファイルの値を参照し、無ければ 10 をデフォルトとする
        self.max_auto_skip = int(max_auto_skip) if max_auto_skip is not None else int(getattr(cfg.env, "max_auto_skip", 10))

        # ロボット介入履歴のトラッキング
        self.steps_since_last_intervention: int = 0  # 最後の介入からのステップ数
        self.stable_step_start: Optional[int] = None  # 安定状態が開始されたステップ（Noneなら未安定）
        self.terminal_bonus_given: bool = False  # terminal_bonusが既に与えられたかどうか

        # 前ステップの状態トラッキング
        self.previous_intervened: bool = False
        self.previous_rel_after_horizon: Optional[Dict[str, Any]] = None
        self.previous_rel_after_bonus: Optional[Dict[str, Any]] = None

        # personasの処理（List形式とDict形式の両方をサポート）
        if personas is None:
            # personasが指定されていない場合は、設定から読み込む
            personas_cfg = getattr(cfg.env, "personas", None)
            if debug:
                print(f"[ConvoEnv.__init__] personas_cfg type: {type(personas_cfg)}")
                print(f"[ConvoEnv.__init__] personas_cfg keys: {personas_cfg.keys() if isinstance(personas_cfg, dict) else 'N/A'}")
            if personas_cfg and isinstance(personas_cfg, dict):
                self.persona_pool: List[str] = sorted(personas_cfg.keys())
                self.persona_triggers: Dict[str, List[str]] = {
                    name: info.get("triggers", []) if isinstance(info, dict) else []
                    for name, info in personas_cfg.items()
                }
                if debug:
                    print(f"[ConvoEnv.__init__] persona_pool: {self.persona_pool}")
                    print(f"[ConvoEnv.__init__] persona_triggers: {self.persona_triggers}")
            else:
                self.persona_pool: List[str] = []
                self.persona_triggers: Dict[str, List[str]] = {}
                if debug:
                    print(f"[ConvoEnv.__init__] personas_cfg is None or not dict, persona_triggers set to empty")
        elif isinstance(personas, dict):
            # Dict形式: {"A": {"triggers": [...]}, "B": {...}, ...}
            self.persona_pool = sorted(personas.keys())
            self.persona_triggers = {
                name: info.get("triggers", []) if isinstance(info, dict) else []
                for name, info in personas.items()
            }
            if debug:
                print(f"[ConvoEnv.__init__] Dict personas provided")
                print(f"[ConvoEnv.__init__] persona_pool: {self.persona_pool}")
                print(f"[ConvoEnv.__init__] persona_triggers: {self.persona_triggers}")
        else:
            # List形式（後方互換性）: ["A", "B", "C"]
            persona_list = [p for p in personas if p]
            self.persona_pool = list(persona_list)
            self.persona_triggers = {}
            if debug:
                print(f"[ConvoEnv.__init__] List personas provided, persona_triggers set to empty")

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
        self.used_triggers: List[str] = []  # 既に使用した地雷のリスト
        self.current_topic: Optional[str] = None  # 現在のトピック
        self.current_topic_trigger: Optional[str] = None  # 現在のトピックの元になった地雷
        self._last_step_relations: Optional[Dict[str, Any]] = None  # 前ステップの最終関係性（evaluation_horizon後またはterminal_bonus_duration後）

        self._last_observation: Optional[str] = None
        self.reset()

    def reset(self) -> str:
        # 環境をリセットして初期状態を返す
        self.episode += 1
        self._episode_step = 0
        self._unstable_step = 0  # 不安定なターン（訓練データになるターン）のカウント
        self.t = 0
        self.logs = []
        self.scorer = RelationScorer(**self._scorer_kwargs)

        # ロボット介入履歴のリセット
        self.steps_since_last_intervention = 0
        self.stable_step_start = None
        self.terminal_bonus_given = False
        self._last_step_relations = None  # 前ステップの関係性をリセット

        # 前ステップ状態のリセット
        self.previous_intervened = False
        self.previous_rel_after_horizon = None
        self.previous_rel_after_bonus = None

        # エピソードごとに各話者の過激度を設定（50%の確率でマイルドまたは過激）
        import random
        self.speaker_aggressiveness = {}
        for speaker in self.persona_pool:
            self.speaker_aggressiveness[speaker] = random.random() < 0.5  # True=過激, False=マイルド

        if self.debug:
            aggr_str = ", ".join([f"{s}: {'過激' if is_aggr else 'マイルド'}"
                                   for s, is_aggr in self.speaker_aggressiveness.items()])
            print(f"[reset] Episode {self.episode} 話者過激度設定: {aggr_str}")

        # 感情ニーズをランダムに割り当て
        emotional_needs_pool = ["recognition", "mediation", "solution", "independence"]
        self.persona_emotional_needs = {}
        for speaker in self.persona_pool:
            self.persona_emotional_needs[speaker] = random.choice(emotional_needs_pool)

        if self.debug:
            needs_str = ", ".join([f"{s}: {need}" for s, need in self.persona_emotional_needs.items()])
            print(f"  💭 感情ニーズ割り当て: {needs_str}")

        # トピックを生成
        self.current_topic = self._get_topic_suggestion()
        self.used_topics.append(self.current_topic)

        # 初期会話を生成（evaluation_horizon回のターン = 通常1ラウンド）
        self._bootstrap_humans(self.evaluation_horizon)

        # 安定状態の自動スキップ（初期会話でも適用）
        # step()と同様に、安定であればmax_auto_skip回まで人間発話を生成
        auto_skip_count = 0
        max_auto_skip = int(getattr(self, "max_auto_skip", 10))
        initial_conversation_log = []  # 初期会話を記録

        if self.debug:
            print(f"  [reset auto_skip] max_auto_skip={max_auto_skip}")

        # 初期会話をログに記録
        for log_entry in self.logs:
            if log_entry.get("speaker") != "ロボット":
                initial_conversation_log.append({
                    "speaker": log_entry.get("speaker"),
                    "utterance": log_entry.get("utterance"),
                })

        while auto_skip_count < max_auto_skip:
            # 関係性を評価
            participants = self._participants(self.logs)
            human_utterance_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

            if human_utterance_count >= self.start_relation_check_after_utterances:
                filtered_logs = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                try:
                    scores, _ = self._relation_state(filtered_logs, update_state=True)
                    metrics, _ = self._metrics_state(scores, participants)
                    unstable_count = metrics.get("unstable_triads", 0)

                    # 不安定なら抜ける
                    if unstable_count > 0:
                        if self.debug:
                            print(f"  [reset auto_skip] 不安定になったのでループ終了 (unstable_triads={unstable_count}, auto_skip_count={auto_skip_count})")
                        break

                    # 安定なら人間発話を1つ生成
                    auto_skip_count += 1
                    if self.debug:
                        print(f"  [reset auto_skip] 安定状態を検出、人間発話を自動生成 ({auto_skip_count}/{max_auto_skip})")

                    replies = human_reply(self.logs, self.persona_pool, topic=self.current_topic, topic_trigger=self.current_topic_trigger, num_speakers=1, speaker_aggressiveness=self.speaker_aggressiveness)
                    if replies:
                        self.logs.extend(replies)
                        # 初期会話ログにも追加
                        for r in replies:
                            if r.get("speaker") != "ロボット":
                                initial_conversation_log.append({
                                    "speaker": r.get("speaker"),
                                    "utterance": r.get("utterance"),
                                })
                    else:
                        if self.debug:
                            print(f"  [reset auto_skip] 発話生成失敗、スキップ終了 (auto_skip_count={auto_skip_count})")
                        break

                except Exception as e:
                    if self.debug:
                        print(f"  [reset auto_skip] 関係性評価に失敗、スキップ終了: {e}")
                    break
            else:
                # まだ3発話に達していないので抜ける
                if self.debug:
                    print(f"  [reset auto_skip] まだ{self.start_relation_check_after_utterances}発話未満、スキップ終了")
                break

        # 初期会話ログを保存（step()で参照できるように）
        self.initial_conversation_log = initial_conversation_log

        # max_auto_skip回試しても不安定にならなかった場合は話題を切り替える
        if auto_skip_count >= max_auto_skip:
            if self.debug:
                print(f"  [reset auto_skip] {max_auto_skip}回試しても不安定にならず → 話題を切り替え")

            # 新しい話題を生成
            self.current_topic = self._get_topic_suggestion()
            self.used_topics.append(self.current_topic)

            # ログをクリアして新しいトピックで初期会話を生成
            self.logs = []
            self.scorer = RelationScorer(**self._scorer_kwargs)
            self._bootstrap_humans(self.evaluation_horizon)

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
            print(f"  previous_intervened: {self.previous_intervened}")

        # 前ステップの介入状態に基づいて処理
        human_replies_before = []  # このステップで生成した人間発話を保存

        # 人間発話生成をスキップする条件：
        # 1. 前のステップで介入した場合（evaluation_horizon後の発話が既に生成済み）
        # 2. ステップ1の場合（reset()で3発話が既に生成済み）
        skip_human_generation = self.previous_intervened or self._episode_step == 0

        if skip_human_generation:
            # 前のステップの最終関係性を引き継ぐ（rel_after_bonus を優先）
            if self.previous_intervened:
                if self.previous_rel_after_bonus:
                    rel_before_snapshot = self.previous_rel_after_bonus
                    if self.debug:
                        print(f"  使用: 前のrel_after_bonus")
                elif self.previous_rel_after_horizon:
                    rel_before_snapshot = self.previous_rel_after_horizon
                    if self.debug:
                        print(f"  使用: 前のrel_after_horizon")
                else:
                    rel_before_snapshot = {}
                    if self.debug:
                        print(f"  使用: 空のスナップショット")
            else:
                # ステップ1: reset()で既に発話が生成されている
                rel_before_snapshot = self.relation_snapshot()
                if self.debug:
                    print(f"  ステップ1: reset()で既に{len(self.logs)}発話生成済み")
        else:
            # 人間1発話生成
            replies = human_reply(self.logs, self.persona_pool, topic=self.current_topic, topic_trigger=self.current_topic_trigger, num_speakers=1, speaker_aggressiveness=self.speaker_aggressiveness)
            if replies:
                self.logs.extend(replies)
                human_replies_before = replies  # 生成した人間発話を保存
                if self.debug:
                    print(f"  生成した人間発話: {len(replies)}件")
            # 関係性の事前スナップショット
            rel_before_snapshot = self.relation_snapshot()
            if self.debug:
                print(f"  rel_before_snapshot取得")

        # 安定状態の自動スキップ（PPO学習の効率化）
        # filter_zero_rewards=trueの場合、安定状態では不安定になるまで人間発話を自動生成
        skip_stable = getattr(get_config().ppo, "filter_zero_rewards", False)
        auto_skip_count = 0
        # インスタンス属性を使う（コンストラクタや設定で変更可能）
        max_auto_skip = int(getattr(self, "max_auto_skip", 10))

        # skip_stableが無効の場合は、auto_skipを実行しない
        if not skip_stable:
            max_auto_skip = 0

        # auto-skip loop: 安定状態の間、人間発話を自動生成して関係性を再評価する
        # 不安定になるまで人間発話を生成し続け、human_replies_beforeに蓄積する
        # ループ終了後の関係性評価結果を保存
        final_rel_metrics = None
        final_rel_scores = None

        # ループ開始前に初期関係性を評価（previous_intervenedの場合は評価が必要）
        if skip_stable and rel_before_snapshot.get("metrics", {}).get("unstable_triads") is None:
            # metricsがない場合は新規評価
            try:
                participants = self._participants(self.logs)
                human_utterance_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')
                if human_utterance_count >= self.start_relation_check_after_utterances:
                    filtered_logs = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                    scores_init, _ = self._relation_state(filtered_logs, update_state=False)
                    metrics_init, _ = self._metrics_state(scores_init, participants)
                    rel_before_snapshot = {
                        "metrics": metrics_init,
                        "unstable_triads": metrics_init.get("unstable_triads", 0),
                        "edges": metrics_init.get("edges", {}),
                    }
                    if self.debug:
                        print(f"  [auto_skip] 初期関係性を評価: unstable_triads={metrics_init.get('unstable_triads', 0)}")
            except Exception as e:
                if self.debug:
                    print(f"  [auto_skip] 初期関係性評価に失敗: {e}")

        if self.debug:
            print(f"  [auto_skip] skip_stable={skip_stable}, max_auto_skip={max_auto_skip}")
            print(f"  [auto_skip] 現在の関係性: unstable_triads={rel_before_snapshot.get('metrics', {}).get('unstable_triads', 'N/A')}")

        while skip_stable and auto_skip_count < max_auto_skip:
            # rel_before_snapshot は前ループでの評価結果を表す
            current_rel = rel_before_snapshot
            if isinstance(current_rel, dict):
                if "metrics" in current_rel:
                    unstable_count = current_rel["metrics"].get("unstable_triads", 0)
                else:
                    unstable_count = current_rel.get("unstable_triads", 0)
            else:
                unstable_count = 0

            # 不安定なら抜ける
            if unstable_count > 0:
                if self.debug:
                    print(f"  [auto_skip] 不安定になったのでループ終了 (unstable_triads={unstable_count}, auto_skip_count={auto_skip_count})")
                break

            # 安定なら人間発話を1つ生成して再評価
            auto_skip_count += 1
            if self.debug:
                print(f"  [auto_skip] 安定状態を検出、人間発話を自動生成 ({auto_skip_count}/{max_auto_skip})")

            replies = human_reply(self.logs, self.persona_pool, topic=self.current_topic, topic_trigger=self.current_topic_trigger, num_speakers=1, speaker_aggressiveness=self.speaker_aggressiveness)
            if replies:
                # 安定時に生成された発話を self.logs に追加（実環境の会話履歴として保存）
                self.logs.extend(replies)
                # human_replies_before にも追加（info辞書で返すため）
                human_replies_before.extend(replies)

                # 関係性を再評価する（update_state=True で永続的に更新）
                try:
                    scores_new, _ = self._relation_state(self.logs, update_state=True)
                    metrics_new, _ = self._metrics_state(scores_new, self._participants(self.logs))
                    rel_before_snapshot = {
                        "metrics": metrics_new,
                        "unstable_triads": metrics_new.get("unstable_triads", 0),
                        "edges": metrics_new.get("edges", {}),
                    }
                    # ループ終了後に使うために保存
                    final_rel_metrics = metrics_new
                    final_rel_scores = scores_new

                    if self.debug:
                        print(f"  [auto_skip] 関係性再評価完了: unstable_triads={metrics_new.get('unstable_triads', 0)}, auto_skip_count={auto_skip_count}")
                except Exception as e:
                    # 万一スコアリングで失敗したらループを抜けて安全側に
                    if self.debug:
                        print(f"  [auto_skip] 関係性評価に失敗、スキップ終了: {e}")
                    break
            else:
                # 発話生成に失敗したら抜ける
                if self.debug:
                    print(f"  [auto_skip] 発話生成失敗、スキップ終了 (auto_skip_count={auto_skip_count})")
                break

        if self.debug:
            print(f"  [auto_skip] ループ終了: auto_skip_count={auto_skip_count}, human_replies_before={len(human_replies_before)}件")

        # auto_skipループの結果を記録（デバッグ用）
        reached_max_auto_skip = auto_skip_count >= max_auto_skip

        # 介入判定
        current_rel = rel_before_snapshot
        # rel_before_snapshotの構造に対応: relation_snapshot()はmetricsキーあり、info["rel_after_*"]はmetricsキーなし
        if isinstance(current_rel, dict):
            if "metrics" in current_rel:
                unstable_count = current_rel["metrics"].get("unstable_triads", 0)
                edges = current_rel["metrics"].get("edges", {})
            else:
                unstable_count = current_rel.get("unstable_triads", 0)
                edges = current_rel.get("edges", {})
        else:
            unstable_count = 0
            edges = {}

        # 関係性スコアリングが不完全な場合（edgesが空）は介入判定を行う
        # これにより、人間発話が生成された後に必ず介入判定が行われる
        is_relation_incomplete = (not edges) or (not current_rel)
        should_skip_intervention_check = (unstable_count == 0) and (not is_relation_incomplete)

        if self.debug:
            print(f"  関係性評価: 不安定トライアド数: {unstable_count}, edges: {len(edges)}, 関係性不完全: {is_relation_incomplete}")
            print(f"  介入判定スキップ: {should_skip_intervention_check}")

        if should_skip_intervention_check:
            # 安定状態 → 早期リターン（報酬なし）
            intervened = False  # 安定状態のため介入していない
            final_balanced = unstable_count == 0
            final_metrics = current_rel.get("metrics", {}) if isinstance(current_rel, dict) else {}
            reward = 0.0
            reward_breakdown = {}
            next_observation = self._make_observation()

            # ステップカウンタを更新
            self.steps_since_last_intervention += 1
            self._episode_step += 1
            self.t += 1
            self.total_steps += 1

            # 終了条件
            # 1. max_auto_skip回試して不安定にならなかった場合
            # 2. max_stepsに達した場合
            # 3. skip_stableが有効で安定状態が続く場合（filter_zero_rewardsの意図）
            if reached_max_auto_skip:
                done = True
                if self.debug:
                    print(f"  [auto_skip] {max_auto_skip}回試しても不安定にならず → エピソード終了")
            elif self.t >= self.max_steps:
                done = True
                if self.debug:
                    print(f"  max_stepsに達したためエピソード終了 (t={self.t}, max_steps={self.max_steps})")
            elif skip_stable and max_auto_skip > 0:
                # filter_zero_rewardsが有効な場合、安定状態で報酬0のサンプルを避けるため終了
                done = True
                if self.debug:
                    print(f"  [filter_zero_rewards] 安定状態のためエピソード終了（報酬0をスキップ）")
            else:
                done = False

            # info辞書
            info: Dict[str, Any] = {
                "plan": None,
                "plan_error": None,
                "intervened": False,
                "balanced": final_balanced,
                "robot_utterance": None,
                "replies": human_replies_before,  # 生成した人間発話を追加（後方互換性のため残す）
                "human_utterance_before_relation": human_replies_before,  # ログ順序調整用
                "rel": final_metrics,
                "reward_breakdown": reward_breakdown,
                "personas": list(self.persona_pool),
                # デバッグ用: ステップ開始時の関係性スナップショット（metricsを含む）
                "rel_before": rel_before_snapshot.get("metrics") if isinstance(rel_before_snapshot, dict) else {},
                "next_observation": next_observation,
            }

            # ステップ開始時の関係性状態を追加
            # rel_before_snapshotの構造に対応
            if isinstance(rel_before_snapshot, dict):
                if "metrics" in rel_before_snapshot:
                    rel_before_metrics = rel_before_snapshot["metrics"]
                else:
                    rel_before_metrics = rel_before_snapshot
            else:
                rel_before_metrics = {}
            unstable_count_before = rel_before_metrics.get("unstable_triads", 0)
            info["status"] = {
                "is_stable": unstable_count_before == 0,
                "edges": rel_before_metrics.get("edges", {}),
            }

            # 報酬の内訳が空の場合は注記
            if not reward_breakdown:
                info.setdefault("reward_notes", "no_breakdown_or_stable_no_reward")

            # previous更新
            self.previous_intervened = intervened
            self.previous_rel_after_horizon = None
            self.previous_rel_after_bonus = None

            self._last_observation = next_observation

            if self.debug:
                print(f"  ✅ 安定状態 → 早期リターン（報酬なし）")
                print(f"    最終報酬: {reward:.4f}")
                print(f"    done: {done}")

            return next_observation, reward, done, info

        # スナップショットを保存（人間発話生成後、ロボット介入前）
        snapshot_logs = [dict(entry) for entry in self.logs]

        # 関係性を評価（人間発話生成後のログを使用）
        participants = self._participants(self.logs)
        human_utterance_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

        # 3発話以上の場合のみ関係性評価を行い、安定判定に使用
        if human_utterance_count >= self.start_relation_check_after_utterances:
            # auto_skipループで既に評価済みの場合はそれを使用（再評価を避ける）
            if final_rel_metrics is not None and final_rel_scores is not None:
                rel = final_rel_scores
                metrics = final_rel_metrics
                trace_scores = []
                trace_metrics = []
                if self.debug:
                    print(f"  [auto_skip] ループ内で評価した関係性を使用（再評価なし）")
            else:
                # ループが実行されなかった場合のみ新規評価
                filtered_logs = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                rel, trace_scores = self._relation_state(filtered_logs, update_state=True)
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
                    print(f"    生成された人間発話: {len(human_replies_before)}件")
                    print(f"    edges: {metrics.get('edges', {})}")

                # steps_since_last_interventionを更新（介入しなかったのでインクリメント）
                self.steps_since_last_intervention += 1

                self._episode_step += 1
                self.t += 1
                self.total_steps += 1

                # 終了条件: 安定状態ではエピソードを終了しない
                # FiniteOnlineDataset.__iter__()でスキップされ、不安定になるまで継続
                done = False

                next_observation = self._make_observation()
                self._last_observation = next_observation

                info = {
                    "plan": None,
                    "plan_error": None,
                    "intervened": False,
                    "balanced": True,
                    "robot_utterance": None,
                    "replies": [entry for entry in self.logs[len(snapshot_logs):]],
                    "human_utterance_before_relation": human_replies_before,  # ログ順序調整用
                    "rel": metrics,
                    "reward_breakdown": {},
                    "personas": list(self.persona_pool),
                    "next_observation": next_observation,
                    "status": {
                        "is_stable": True,
                        "edges": metrics.get("edges", {}),
                    },
                }

                return next_observation, 0.0, done, info
        else:
            # 3発話未満の場合は関係性評価をスキップし、介入判定に進む
            rel = {}
            trace_scores = []
            metrics = {}
            trace_metrics = []
            is_stable = False

            if self.debug:
                print(f"  ⏭️  関係性評価をスキップ（人間発話数 {human_utterance_count} < {self.start_relation_check_after_utterances}）")
                print(f"    介入判定に進みます")

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

        # planから戦略情報を取得
        edge_to_change = plan.get("edge_to_change", "AB") if plan else "AB"
        target_speaker = plan.get("target_speaker", "A") if plan else "A"
        strategy = plan.get("strategy", "plan") if plan else "plan"

        # ロボット発話を追加（介入する場合のみ）
        if intervened:
            self.logs.append(robot_entry)

        # evaluation_horizon回の人間発話を生成
        if self.debug:
            print(f"  🔄 {self.evaluation_horizon}回の人間発話を生成")

        # 感情ニーズと正解戦略フラグを準備
        emotional_needs = self.persona_emotional_needs.copy()
        is_correct_strategy_flags = {}

        # 各話者について正解戦略かどうかをチェック
        for speaker in self.persona_pool:
            preferred = self._get_human_preferred_strategy(speaker)
            # 対象話者の場合は実際の戦略と比較、それ以外はFalse
            if speaker == target_speaker:
                is_correct_strategy_flags[speaker] = (strategy == preferred)
            else:
                is_correct_strategy_flags[speaker] = False

        self._bootstrap_humans(
            self.evaluation_horizon,
            emotional_needs=emotional_needs,
            is_correct_strategy_flags=is_correct_strategy_flags
        )

        # evaluation_horizon後の関係性を評価
        participants_after = self._participants(self.logs)
        human_utterance_count_after = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

        if human_utterance_count_after >= self.start_relation_check_after_utterances:
            filtered_logs_after = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
            rel_after, _ = self._relation_state(filtered_logs_after, update_state=True)
        else:
            rel_after = {}

        metrics_after, _ = self._metrics_state(rel_after, participants_after)
        is_stable_after = metrics_after.get("unstable_triads", 0) == 0

        # 対象エッジのスコアを取得（stable_bonus判定に使用）
        def get_edge_score(edge_str: str, scores_dict: Dict[Tuple[str, str], float]) -> float:
            """エッジ文字列（"AB"など）からスコアを取得"""
            if len(edge_str) >= 2:
                # ("A", "B") または ("B", "A") を試す
                edge_tuple1 = (edge_str[0], edge_str[1])
                edge_tuple2 = (edge_str[1], edge_str[0])
                return scores_dict.get(edge_tuple1, scores_dict.get(edge_tuple2, 0.0))
            return 0.0

        target_edge_score_after = get_edge_score(edge_to_change, rel_after)
        target_edge_positive_after = target_edge_score_after > 0

        if self.debug:
            print(f"  📊 evaluation_horizon後の関係性:")
            print(f"    安定状態: {is_stable_after}")
            print(f"    対象エッジ（{edge_to_change}）スコア: {target_edge_score_after:.4f}")
            print(f"    対象エッジが正: {target_edge_positive_after}")

        # 報酬計算: stable_bonus + preference_match_bonus
        rel_after_bonus = None  # terminal_bonus_duration後の関係性を保存する変数

        # stable_bonus付与条件: 対象エッジが正 AND 全体が安定
        if is_stable_after and target_edge_positive_after:
            if self.debug:
                print(f"  🎯 安定達成 & 対象エッジ正 → terminal_bonusチェック開始")
                print(f"    追加で{self.terminal_bonus_duration}人間発話分の安定性を確認")
        elif self.debug:
            if not is_stable_after:
                print(f"  ⚠️  全体が不安定 → stable_bonusなし")
            elif not target_edge_positive_after:
                print(f"  ⚠️  対象エッジ（{edge_to_change}）が正でない（{target_edge_score_after:.4f}） → stable_bonusなし")

        if is_stable_after and target_edge_positive_after:

            # terminal_bonus_duration人間発話を生成（実際には1ラウンド = 3人間発話が最小単位）
            # 感情ニーズと正解戦略フラグを引き継ぐ
            self._bootstrap_humans(
                self.terminal_bonus_duration,
                emotional_needs=emotional_needs,
                is_correct_strategy_flags=is_correct_strategy_flags
            )

            # 最後の関係性を再評価
            participants_check = self._participants(self.logs)
            human_count_check = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

            stability_maintained = True
            target_edge_positive_check = False
            if human_count_check >= self.start_relation_check_after_utterances:
                filtered_check = filter_logs_by_human_count(self.logs, self.max_history_relation, exclude_robot=True)
                rel_check, _ = self._relation_state(filtered_check, update_state=True)
                metrics_check, _ = self._metrics_state(rel_check, participants_check)

                # 対象エッジのスコアをチェック
                target_edge_score_check = get_edge_score(edge_to_change, rel_check)
                target_edge_positive_check = target_edge_score_check > 0

                if metrics_check.get("unstable_triads", 0) > 0:
                    # 不安定に戻った
                    stability_maintained = False
                    if self.debug:
                        print(f"    ❌ 不安定に戻った")
                elif not target_edge_positive_check:
                    # 対象エッジが負または0になった
                    stability_maintained = False
                    if self.debug:
                        print(f"    ❌ 対象エッジ（{edge_to_change}）が負または0に: {target_edge_score_check:.4f}")
                else:
                    if self.debug:
                        print(f"    ✅ 安定維持 & 対象エッジ正（{target_edge_score_check:.4f}）")

            # terminal_bonus_duration人間発話後も安定が続き、対象エッジも正の場合
            if stability_maintained and target_edge_positive_check:
                if self.debug:
                    print(f"  🎁 安定が持続 & 対象エッジ正 → stable_bonus付与: +{self.stable_bonus:.4f}")
                reward += self.stable_bonus
                reward_breakdown["stable_bonus"] = self.stable_bonus
                # terminal_bonus_duration後の関係性を保存（後でinfoに追加）
                rel_after_bonus = metrics_check
            elif self.debug:
                if not stability_maintained:
                    print(f"  ⚠️  安定が持続せず → stable_bonusなし")
                elif not target_edge_positive_check:
                    print(f"  ⚠️  対象エッジが正でない → stable_bonusなし")

        # 正解戦略の場合、preference_match_bonusを付与
        preferred_strategy = self._get_human_preferred_strategy(target_speaker)
        is_correct_strategy = (strategy == preferred_strategy)
        if is_correct_strategy:
            reward += self.preference_match_bonus
            reward_breakdown["preference_match_bonus"] = self.preference_match_bonus
            if self.debug:
                print(f"  ✅ 正解戦略（{strategy}） → preference_match_bonus付与: +{self.preference_match_bonus:.4f}")

        if self.debug:
            print(f"  💰 最終報酬: {reward:.4f}")

        if intervened:
            self.steps_since_last_intervention = 0
        else:
            self.steps_since_last_intervention += 1

        # ステップカウンタを更新
        self._episode_step += 1
        self._unstable_step += 1  # 不安定なターンをカウント
        self.t += 1
        self.total_steps += 1

        # 最終的な関係性とobservation
        final_participants = self._participants(self.logs)
        final_human_count = sum(1 for log in self.logs if log.get('speaker') != 'ロボット')

        # 終了条件: ステップ数がmax_stepsに達したか確認
        done = self.t >= self.max_steps

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
            "human_utterance_before_relation": human_replies_before,  # ログ順序調整用
            "rel": final_metrics,
            "reward_breakdown": reward_breakdown,
            "personas": list(self.persona_pool),
            # デバッグ用: ステップ開始時の関係性スナップショット（metricsを含む）
            "rel_before": rel_before_snapshot.get("metrics") if isinstance(rel_before_snapshot, dict) else {},
            "next_observation": next_observation,
            # 感情ニーズと戦略情報
            "emotional_needs": dict(self.persona_emotional_needs),
            "target_speaker": target_speaker,
            "chosen_strategy": strategy,
            "preferred_strategy": preferred_strategy,
            "preference_match": is_correct_strategy,
            # 対象エッジ情報（stable_bonus判定用）
            "edge_to_change": edge_to_change,
            "target_edge_score_after": target_edge_score_after,
            "target_edge_positive_after": target_edge_positive_after,
        }

        # ステップ開始時の関係性を追加（ログ用）
        # rel_before_snapshotの構造に対応
        if isinstance(rel_before_snapshot, dict):
            if "metrics" in rel_before_snapshot:
                rel_before_metrics = rel_before_snapshot["metrics"]
            else:
                rel_before_metrics = rel_before_snapshot
        else:
            rel_before_metrics = {}
        unstable_count_before = rel_before_metrics.get("unstable_triads", 0)
        info["status"] = {
            "is_stable": unstable_count_before == 0,
            "edges": rel_before_metrics.get("edges", {}),
        }

        if self.debug:
            print(f"[ConvoEnv.step] DEBUG - info['status']:")
            print(f"  is_stable: {unstable_count_before == 0}")
            print(f"  edges: {rel_before_metrics.get('edges', {})}")

        # 介入した場合、evaluation_horizon後の関係性を追加
        if intervened:
            info["rel_after_horizon"] = metrics_after
            info["stable_after_horizon"] = is_stable_after

            # terminal_bonus_duration後の関係性を追加（該当する場合）
            if rel_after_bonus is not None:
                info["rel_after_bonus"] = rel_after_bonus

        # 報酬の内訳が空の場合は注記を追加（安定で報酬計算がスキップされた等の理由）
        if not reward_breakdown:
            info.setdefault("reward_notes", "no_breakdown_or_stable_no_reward")

        # previous更新
        self.previous_intervened = intervened
        self.previous_rel_after_horizon = info.get("rel_after_horizon") if intervened else None
        self.previous_rel_after_bonus = info.get("rel_after_bonus") if intervened and info.get("rel_after_bonus") else None

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
        # ランダムに1つだけ選択（既に選択済みの地雷は除外）
        selected_trigger = None
        if self.persona_triggers and "{trigger_examples}" in system:
            # 全personaのtriggersを平坦化（重複を除く）
            all_triggers = []
            for triggers in self.persona_triggers.values():
                all_triggers.extend(triggers)
            # 重複を除去
            all_triggers = list(set(all_triggers))

            if all_triggers:
                # まだ選択していない地雷を取得
                available_triggers = [t for t in all_triggers if t not in self.used_triggers]

                # 全ての地雷が選択済みの場合はリセット
                if not available_triggers:
                    self.used_triggers = []
                    available_triggers = all_triggers

                # ランダムに1つだけ選択
                selected_trigger = random.choice(available_triggers)
                self.used_triggers.append(selected_trigger)
                trigger_examples = selected_trigger
                system = system.replace("{trigger_examples}", trigger_examples)

                if self.debug:
                    print(f"[topic] 選択された地雷: {selected_trigger}")
                    print(f"[topic] 地雷を持つペルソナ: {[p for p, ts in self.persona_triggers.items() if selected_trigger in ts]}")
            else:
                # triggersが空の場合はプレースホルダーを削除
                system = system.replace("{trigger_examples}", "様々なテーマ")
                if self.debug:
                    print(f"[topic] 地雷リストが空です")
        else:
            if self.debug:
                print(f"[topic] persona_triggers: {bool(self.persona_triggers)}, has placeholder: {'{trigger_examples}' in system}")

        # 選択された地雷を保存（人間LLMで使用するため）
        self.current_topic_trigger = selected_trigger

        if self.debug:
            print(f"[topic] current_topic_trigger: {self.current_topic_trigger}")

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
        # intervention_max_history個の人間発話 + その間のロボット発話を取得（介入判定用）
        filtered_logs = filter_logs_by_human_count(self.logs, self.intervention_max_history)
        history_lines = [f"[{item.get('speaker', '?')}] {item.get('utterance', '').strip()}" for item in filtered_logs]
        if not history_lines:
            history_lines = ["(履歴なし)"]

        scores = context.get("scores", {}) or {"AB": 0.0, "BC": 0.0, "CA": 0.0}
        
        # 改善すべきエッジとターゲット話者を特定（共通関数を使用）
        # ※重要: ターゲット話者はここで1回だけランダムに選択し、
        #   後続の_parse_planで再利用する（2回選択すると不整合が生じるため）
        # relation_snapshot()から取得したエッジ情報を使用
        rel = self.relation_snapshot()
        if isinstance(rel, dict):
            metrics = rel.get("metrics", rel)
            edges = metrics.get("edges", {})
            if edges:
                target_edge, target_speaker = select_target_edge_and_speaker(
                    edges, 
                    debug=False,  # observation生成時はデバッグ出力不要
                    debug_prefix=""
                )
                # 選択したターゲット話者を保存（_parse_planで再利用）
                self._current_target_edge = target_edge
                self._current_target_speaker = target_speaker
                
                # エッジのスコアを取得
                # edgesのキーはタプルの可能性があるので、両方の形式を試す
                target_score = None
                for edge_key, score in edges.items():
                    if isinstance(edge_key, tuple):
                        edge_str = "".join(str(c) for c in edge_key)
                    else:
                        edge_str = str(edge_key).replace(",", "").replace(" ", "").upper()
                    if edge_str == target_edge or edge_str == target_edge[::-1]:
                        target_score = score
                        break
                
                if target_score is None:
                    target_score = 0.0
            else:
                target_edge = "AB"
                target_score = 0.0
                target_speaker = "A"
                self._current_target_edge = target_edge
                self._current_target_speaker = target_speaker
        else:
            target_edge = "AB"
            target_score = 0.0
            target_speaker = "A"
            self._current_target_edge = target_edge
            self._current_target_speaker = target_speaker

        # プロンプト最適化: 可変要素（会話履歴、関係スコア、改善対象）を返す
        # 固定説明（タスク、制約、戦略）はシステムプロンプトに移動済み（build_robot_messages参照）
        prompt_lines = [
            "履歴:",
            *history_lines,
            "",
            "現在の関係スコア（-1..1）: " + ", ".join(f"w_{edge}={value:+.2f}" for edge, value in scores.items()),
            "",
            f"改善すべきエッジ: {target_edge} (現在: {target_score:+.2f})",
            f"発話対象（ターゲット）: {target_speaker}",
        ]
        return "\n".join(prompt_lines)

    def _parse_plan(self, action: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
        # 先頭・末尾の引用符や括弧を削る
        if action is None:
            return None, "empty_action", None
        text = action.strip()
        if not text:
            return None, "empty_action", None

        # 新形式: 数字1-4のパース
        strategy_num = None
        for char in text:
            if char in '1234':
                strategy_num = int(char)
                break
        
        # JSONパース（旧形式との後方互換性のため）
        json_parsed = False
        payload = None
        try:
            payload = json.loads(text)
            json_parsed = True
        except json.JSONDecodeError:
            pass

        plan: Dict[str, Any] = {}

        # 新形式（数字）の処理
        if strategy_num is not None:
            strategy_map = {
                1: "validate",
                2: "bridge",
                3: "plan",
                4: "no_intervention"
            }
            
            strategy = strategy_map.get(strategy_num)
            if not strategy:
                return None, "invalid_strategy_number", text

            # edge_to_changeとtarget_speakerは_make_observationで既に選択済み
            # ※重要: ターゲット話者をここで再選択すると、LLMに提示した値と異なる値になる
            #   可能性があるため、_make_observationで選択した値を再利用する
            edge_to_change = getattr(self, '_current_target_edge', None)
            target_speaker = getattr(self, '_current_target_speaker', None)

            if strategy == "no_intervention":
                plan["intervene_now"] = False
                # 介入なしの場合でもedge_to_changeとtarget_speakerを記録（ログ出力用）
                plan["edge_to_change"] = edge_to_change
                plan["target_speaker"] = target_speaker
                return plan, None, None

            # 介入ありの場合
            plan["intervene_now"] = True
            plan["strategy"] = strategy
            plan["edge_to_change"] = edge_to_change
            plan["target_speaker"] = target_speaker

            return plan, None, None

        # 旧形式（JSON）の処理（後方互換性）
        elif json_parsed and isinstance(payload, dict):
            intervene_now = _coerce_bool(payload.get("intervene_now"))
            if intervene_now is None:
                intervene_now = bool(payload.get("intervene_now"))
            plan["intervene_now"] = bool(intervene_now)

            edge_raw = str(payload.get("edge_to_change", "AB")).upper().strip()
            if edge_raw not in {"AB", "BC", "CA"}:
                edge_raw = "AB"
            plan["edge_to_change"] = edge_raw

            strategy_raw = str(payload.get("strategy", "plan")).strip()
            if strategy_raw not in _PLANNER_STRATEGIES:
                strategy_raw = "plan"
            plan["strategy"] = strategy_raw

            target_raw = str(payload.get("target_speaker", "A")).upper().strip()
            if target_raw not in {"A", "B", "C"}:
                target_raw = "A"
            plan["target_speaker"] = target_raw

            return plan, None, None
        
        else:
            return None, "not_json_or_number", text

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

        # no_interventionの場合は「見守り」発話
        if strategy == "no_intervention":
            import random
            utterances = [
                "（静かに聞いています）",
                "（うなずいて見守っています）",
                "（話を聞いています）",
                "（黙って耳を傾けています）"
            ]
            utterance = random.choice(utterances)
            return {
                "speaker": "ロボット",
                "utterance": utterance
            }

        # robot_max_history個の人間発話 + その間のロボット発話を取得（ロボット発話生成用）
        filtered_logs = filter_logs_by_human_count(self.logs, self.robot_max_history)
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
                    # Azure OpenAIのcontent filterエラーの場合は即座にフォールバック
                    if "content_filter" in str(exc) or "ResponsibleAIPolicyViolation" in str(exc):
                        if getattr(get_config().env, "debug", False):
                            print(f"[robot_utterance] Azure content filter triggered, using fallback immediately")
                        break  # すぐにフォールバックに移行

                    if getattr(get_config().env, "debug", False):
                        print(f"[robot_utterance] attempt {attempt} failed:", exc)
                    if attempt < max_attempts:
                        time.sleep(base_backoff * (2 ** (attempt - 1)))
                    else:
                        if getattr(get_config().env, "debug", False):
                            print("[robot_utterance] all attempts failed, falling back to local heuristic")
        return {"speaker": "ロボット", "utterance": fallback_text}

    def _fallback_intervention_text(self, target_name: str, partner_name: str, strategy: str) -> str:
        # フォールバック用の介入発話テンプレート
        templates = {
            "plan": f"{target_name}さん、次の一歩として、まず小さなことから始めてみませんか？",
            "validate": f"{target_name}さん、あなたの意見はとても大切です。もっと聞かせてください。",
            "bridge": f"{target_name}さん、{partner_name}さん、お二人には共通の目標があると思います。",
        }
        return templates.get(strategy, templates["plan"])

    def _bootstrap_humans(
        self,
        target_turns: int,
        emotional_needs: Optional[Dict[str, str]] = None,
        is_correct_strategy_flags: Optional[Dict[str, bool]] = None
    ) -> None:
        # 人間参加者の発話を追加して会話を進める
        turns_added = 0
        safety_limit = max(1, self.max_steps * max(1, len(self.persona_pool)) * 3)
        while turns_added < target_turns and turns_added < safety_limit:
            # 1人間発話ずつ生成
            replies = human_reply(
                self.logs,
                self.persona_pool,
                topic=self.current_topic,
                topic_trigger=self.current_topic_trigger,
                num_speakers=1,
                speaker_aggressiveness=self.speaker_aggressiveness,
                emotional_needs=emotional_needs,
                is_correct_strategy_flags=is_correct_strategy_flags
            )
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

    def _get_human_preferred_strategy(self, target_speaker: str) -> str:
        """対象話者の感情ニーズから好まれる戦略を返す"""
        need = self.persona_emotional_needs.get(target_speaker, "recognition")
        need_to_strategy = {
            "recognition": "validate",
            "mediation": "bridge",
            "solution": "plan",
            "independence": "no_intervention"
        }
        return need_to_strategy[need]

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

        # 戻り値を構築（後方互換性のため、metricsの内容もトップレベルに含める）
        result = {
            "participants": [p for p in participants if p != "ロボット"],
            "triangles": triangles,
            "scores": scores,
            "metrics": relations,  # metricsキーを追加（edgesなどを含む）
            # 後方互換性のため、metricsの主要なキーをトップレベルにも追加
            "unstable_triads": relations.get("unstable_triads", 0),
            "balanced_triads": relations.get("balanced_triads", 0),
            "edges": relations.get("edges", {}),
            "iso_nodes": relations.get("iso_nodes", 0),
        }
        return result
