"""
バッチサマリーの集計と記録を管理するモジュール
"""
from collections import defaultdict
from typing import Dict, Any
import math


class BatchSummaryCollector:
    """PPO訓練中のバッチごとの統計を集計"""

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.batch_count = 0
        self.reset()

    def reset(self):
        """バッチごとの統計をリセット"""
        self.total_reward = 0.0
        self.intervention_count = 0
        self.unstable_no_intervention_count = 0
        self.plan_error_count = 0
        self.decision_counter = 0

        # 選択肢のカウント（バッチ全体で蓄積）
        self.choice_counts = {
            "intervene_now": defaultdict(int),
            "edge_to_change": defaultdict(int),
            "strategy": defaultdict(int),
            "target_speaker": defaultdict(int),
        }

    def add_decision(self, plan_obj: Dict[str, Any]):
        """モデルの決定を記録"""
        self.decision_counter += 1

        intervene_now = plan_obj.get("intervene_now", False)
        self.choice_counts["intervene_now"][str(intervene_now)] += 1

        # strategyを常に記録（output_errorやno_interventionを含む）
        strategy = plan_obj.get("strategy", None)
        if strategy:
            self.choice_counts["strategy"][strategy] += 1
        elif not intervene_now:
            # intervene_now=falseでstrategyがない場合はoutput_errorとして記録
            self.choice_counts["strategy"]["output_error"] += 1

        # intervene_now=trueの場合のみedgeとtargetを記録
        if intervene_now:
            edge = plan_obj.get("edge_to_change", "unknown")
            target = plan_obj.get("target_speaker", "unknown")
            self.choice_counts["edge_to_change"][edge] += 1
            self.choice_counts["target_speaker"][target] += 1
    
    def get_strategy_distribution(self) -> Dict[str, int]:
        """戦略の分布を返す（エントロピー計算用）"""
        return dict(self.choice_counts["strategy"])

    def add_step(self, reward: float, intervened: bool, is_stable_before: bool, plan_error: bool):
        """ステップの結果を記録"""
        self.total_reward += reward
        if intervened:
            self.intervention_count += 1
        if not is_stable_before and not intervened:
            self.unstable_no_intervention_count += 1
        if plan_error:
            self.plan_error_count += 1

    def is_batch_ready(self) -> bool:
        """バッチが完了したかチェック"""
        return self.decision_counter >= self.batch_size

    def get_summary(self) -> Dict[str, Any]:
        """バッチサマリーを取得"""
        self.batch_count += 1

        # エントロピーを計算
        choice_entropy = {
            "intervene_now": self._compute_entropy(self.choice_counts["intervene_now"]),
            "edge_to_change": self._compute_entropy(self.choice_counts["edge_to_change"]),
            "strategy": self._compute_entropy(self.choice_counts["strategy"]),
            "target_speaker": self._compute_entropy(self.choice_counts["target_speaker"]),
        }

        # 分布を辞書に変換
        choice_distributions = {
            "intervene_now": dict(self.choice_counts["intervene_now"]),
            "edge_to_change": dict(self.choice_counts["edge_to_change"]),
            "strategy": dict(self.choice_counts["strategy"]),
            "target_speaker": dict(self.choice_counts["target_speaker"]),
        }

        summary = {
            "batch_id": self.batch_count,
            "batch_size": self.batch_size,
            "total_reward": self.total_reward,
            "total_interventions": self.intervention_count,
            "unstable_no_intervention": self.unstable_no_intervention_count,
            "plan_errors": self.plan_error_count,
            "choice_entropy": choice_entropy,
            "choice_distributions": choice_distributions,
        }

        # 次のバッチのためにリセット
        self.reset()

        return summary

    @staticmethod
    def _compute_entropy(counts_dict: Dict[str, int]) -> float:
        """カテゴリ分布からエントロピーを計算（bits）"""
        if not counts_dict:
            return 0.0
        total = sum(counts_dict.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counts_dict.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
