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
        self.no_intervention_count = 0
        self.output_error_count = 0
        self.decision_counter = 0

        # 戦略のカウント（バッチ全体で蓄積）
        self.strategy_counts = defaultdict(int)

    def add_reward(self, reward: float):
        """報酬を記録"""
        self.total_reward += reward

    def add_decision(self, plan_obj: Dict[str, Any]):
        """モデルの決定を記録

        Args:
            plan_obj: 決定内容の辞書
        """
        self.decision_counter += 1

        strategy = plan_obj.get("strategy", None)

        if strategy == "output_error":
            # LLMの出力が不正だった場合
            self.strategy_counts["output_error"] += 1
            self.output_error_count += 1
        elif strategy in ["validate", "bridge", "plan"]:
            # 実際に介入した場合
            self.strategy_counts[strategy] += 1
            self.intervention_count += 1
        elif strategy is None and not plan_obj.get("intervene_now", False):
            # モデルが"4"（no_intervention）を出力した場合
            self.strategy_counts["no_intervention"] += 1
            self.no_intervention_count += 1
        else:
            # 予期しないケース（デバッグ用）
            print(f"[BatchSummaryCollector] 予期しないplan_obj: {plan_obj}")
    
    def get_strategy_distribution(self) -> Dict[str, int]:
        """戦略の分布を返す（エントロピー計算用）"""
        return dict(self.strategy_counts)

    def is_batch_ready(self) -> bool:
        """バッチが完了したかチェック"""
        return self.decision_counter >= self.batch_size

    def get_summary(self, kl_coef: float = None, current_lr: float = None) -> Dict[str, Any]:
        """バッチサマリーを取得

        Args:
            kl_coef: 現在のKL係数（オプション）
            current_lr: 現在の学習率（オプション）
        """
        self.batch_count += 1

        # エントロピーを計算（strategyのみ）
        choice_entropy = {
            "strategy": self._compute_entropy(self.strategy_counts),
        }

        summary = {
            "batch_id": self.batch_count,
            "batch_size": self.batch_size,
            "total_reward": self.total_reward,
            "total_interventions": self.intervention_count,
            "no_intervention_count": self.no_intervention_count,
            "output_error_count": self.output_error_count,
            "choice_entropy": choice_entropy,
            "strategy_counts": dict(self.strategy_counts),
        }

        # KL係数と学習率を記録（指定されている場合）
        if kl_coef is not None:
            summary["kl_coef"] = kl_coef
        if current_lr is not None:
            summary["learning_rate"] = current_lr

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
