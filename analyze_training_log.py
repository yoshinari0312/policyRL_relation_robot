#!/usr/bin/env python3
"""
学習ログの詳細分析スクリプト
ppo_run-20251030-172826のデータを分析
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_log(log_dir: str):
    log_path = Path(log_dir)

    # メトリクスファイルの読み込み
    metrics_file = log_path / "metrics.jsonl"
    conv_summary_file = log_path / "conversation_summary.jsonl"
    events_file = log_path / "run_events.jsonl"

    print("=" * 80)
    print(f"学習ログ分析: {log_dir}")
    print("=" * 80)
    print()

    # 1. メトリクスの分析
    if metrics_file.exists():
        print("## 1. メトリクス分析")
        print("-" * 80)

        metrics = []
        with open(metrics_file) as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except:
                    pass

        print(f"総メトリクス数: {len(metrics)}")

        # 報酬の統計
        rewards = []
        kl_values = []
        entropy_values = []
        approxkl_values = []

        for m in metrics:
            if "objective/rlhf_reward" in m:
                rewards.append(m["objective/rlhf_reward"])
            if "objective/kl" in m:
                kl_values.append(m["objective/kl"])
            if "policy/entropy_avg" in m:
                entropy_values.append(m["policy/entropy_avg"])
            if "policy/approxkl_avg" in m:
                approxkl_values.append(m["policy/approxkl_avg"])

        if rewards:
            print(f"\n報酬統計:")
            print(f"  平均: {np.mean(rewards):.4f}")
            print(f"  標準偏差: {np.std(rewards):.4f}")
            print(f"  最小値: {np.min(rewards):.4f}")
            print(f"  最大値: {np.max(rewards):.4f}")
            print(f"  中央値: {np.median(rewards):.4f}")
            print(f"  パーセンタイル [25%, 50%, 75%]: {np.percentile(rewards, [25, 50, 75])}")

            # 報酬が0の割合
            zero_rewards = sum(1 for r in rewards if abs(r) < 1e-6)
            print(f"  報酬=0の割合: {zero_rewards}/{len(rewards)} ({100*zero_rewards/len(rewards):.1f}%)")

            # 報酬の分布（ヒストグラム）
            print(f"\n報酬分布（10ビン）:")
            hist, bins = np.histogram(rewards, bins=10)
            for i, (count, bin_start, bin_end) in enumerate(zip(hist, bins[:-1], bins[1:])):
                bar = "#" * int(50 * count / max(hist))
                print(f"  [{bin_start:7.3f}, {bin_end:7.3f}): {count:4d} {bar}")

        if kl_values:
            print(f"\nKLダイバージェンス統計:")
            print(f"  平均: {np.mean(kl_values):.4f}")
            print(f"  標準偏差: {np.std(kl_values):.4f}")
            print(f"  最小値: {np.min(kl_values):.4f}")
            print(f"  最大値: {np.max(kl_values):.4f}")
            print(f"  中央値: {np.median(kl_values):.4f}")

            # KL > 0.3 の割合
            high_kl = sum(1 for k in kl_values if k > 0.3)
            print(f"  KL > 0.3 の割合: {high_kl}/{len(kl_values)} ({100*high_kl/len(kl_values):.1f}%)")

        if entropy_values:
            print(f"\nエントロピー統計:")
            print(f"  平均: {np.mean(entropy_values):.4f}")
            print(f"  標準偏差: {np.std(entropy_values):.4f}")
            print(f"  最小値: {np.min(entropy_values):.4f}")
            print(f"  最大値: {np.max(entropy_values):.4f}")
            print(f"  中央値: {np.median(entropy_values):.4f}")

        if approxkl_values:
            print(f"\napproxKL統計:")
            print(f"  平均: {np.mean(approxkl_values):.4f}")
            print(f"  標準偏差: {np.std(approxkl_values):.4f}")
            print(f"  最小値: {np.min(approxkl_values):.4f}")
            print(f"  最大値: {np.max(approxkl_values):.4f}")
            print(f"  中央値: {np.median(approxkl_values):.4f}")

        print()

    # 2. 会話サマリーの分析（介入率など）
    if conv_summary_file.exists():
        print("## 2. 会話サマリー分析（介入率・報酬の詳細）")
        print("-" * 80)

        summaries = []
        with open(conv_summary_file) as f:
            for line in f:
                try:
                    summaries.append(json.loads(line))
                except:
                    pass

        print(f"総会話ステップ数: {len(summaries)}")

        # 介入判定の分析
        intervention_decision_count = 0
        intervened_count = 0
        not_intervened_count = 0
        stable_skipped_count = 0

        rewards_when_stable = []
        rewards_when_intervened = []
        rewards_when_not_intervened = []

        reward_breakdowns = defaultdict(list)

        for s in summaries:
            reward = s.get("reward", 0)
            intervened = s.get("intervened", False)

            # 報酬内訳の収集
            breakdown = s.get("reward_breakdown", {})
            for key, value in breakdown.items():
                reward_breakdowns[key].append(value)

            # 安定状態（報酬0）のチェック
            if abs(reward) < 1e-6:
                stable_skipped_count += 1
                rewards_when_stable.append(reward)
            else:
                intervention_decision_count += 1
                if intervened:
                    intervened_count += 1
                    rewards_when_intervened.append(reward)
                else:
                    not_intervened_count += 1
                    rewards_when_not_intervened.append(reward)

        print(f"\n介入判定の内訳:")
        print(f"  安定状態（介入判定スキップ）: {stable_skipped_count}/{len(summaries)} ({100*stable_skipped_count/len(summaries):.1f}%)")
        print(f"  介入判定実施: {intervention_decision_count}/{len(summaries)} ({100*intervention_decision_count/len(summaries):.1f}%)")
        if intervention_decision_count > 0:
            print(f"    → 介入した: {intervened_count}/{intervention_decision_count} ({100*intervened_count/intervention_decision_count:.1f}%)")
            print(f"    → 介入しなかった: {not_intervened_count}/{intervention_decision_count} ({100*not_intervened_count/intervention_decision_count:.1f}%)")

        print(f"\n報酬の状態別統計:")
        if rewards_when_stable:
            print(f"  安定状態時:")
            print(f"    平均: {np.mean(rewards_when_stable):.4f}")
            print(f"    標準偏差: {np.std(rewards_when_stable):.4f}")

        if rewards_when_intervened:
            print(f"  介入時:")
            print(f"    平均: {np.mean(rewards_when_intervened):.4f}")
            print(f"    標準偏差: {np.std(rewards_when_intervened):.4f}")
            print(f"    最小値: {np.min(rewards_when_intervened):.4f}")
            print(f"    最大値: {np.max(rewards_when_intervened):.4f}")

        if rewards_when_not_intervened:
            print(f"  介入しなかった時:")
            print(f"    平均: {np.mean(rewards_when_not_intervened):.4f}")
            print(f"    標準偏差: {np.std(rewards_when_not_intervened):.4f}")
            print(f"    最小値: {np.min(rewards_when_not_intervened):.4f}")
            print(f"    最大値: {np.max(rewards_when_not_intervened):.4f}")

        # 報酬内訳の統計
        print(f"\n報酬内訳の統計:")
        for key, values in sorted(reward_breakdowns.items()):
            if values:
                print(f"  {key}:")
                print(f"    平均: {np.mean(values):.4f}")
                print(f"    標準偏差: {np.std(values):.4f}")
                print(f"    最小値: {np.min(values):.4f}")
                print(f"    最大値: {np.max(values):.4f}")
                # 非ゼロの割合
                non_zero = sum(1 for v in values if abs(v) > 1e-6)
                print(f"    非ゼロ: {non_zero}/{len(values)} ({100*non_zero/len(values):.1f}%)")

        print()

    # 3. イベントログの分析
    if events_file.exists():
        print("## 3. イベントログ分析")
        print("-" * 80)

        events = []
        with open(events_file) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except:
                    pass

        # イベントタイプごとのカウント
        event_counts = defaultdict(int)
        for e in events:
            event_type = e.get("event", "unknown")
            event_counts[event_type] += 1

        print(f"イベントタイプ別カウント:")
        for event_type, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            print(f"  {event_type}: {count}")

        # adaptive_kl_updateイベントの分析
        adaptive_kl_events = [e for e in events if e.get("event") == "adaptive_kl_update"]
        if adaptive_kl_events:
            print(f"\nAdaptive KL更新イベント: {len(adaptive_kl_events)}回")
            increase_count = sum(1 for e in adaptive_kl_events if e.get("action") == "increase")
            decrease_count = sum(1 for e in adaptive_kl_events if e.get("action") == "decrease")
            print(f"  増加: {increase_count}回")
            print(f"  減少: {decrease_count}回")

        print()

    # 4. 問題点の診断
    print("=" * 80)
    print("## 診断と推奨事項")
    print("=" * 80)
    print()

    issues = []
    recommendations = []

    # Issue 1: 報酬のsparseness
    if stable_skipped_count / len(summaries) > 0.5:
        issues.append(f"⚠️  報酬のSparseness: {100*stable_skipped_count/len(summaries):.1f}%のステップで報酬=0（安定状態）")
        recommendations.append("→ 安定状態でも小さな報酬シグナルを与える（例: 維持報酬）")
        recommendations.append("→ または、安定状態のサンプルをバッファから除外する")

    # Issue 2: KL divergenceの爆発
    if kl_values and np.max(kl_values) > 5.0:
        issues.append(f"⚠️  KLダイバージェンスの爆発: 最大値 {np.max(kl_values):.2f} (target: 0.05)")
        recommendations.append("→ 学習率を下げる（現在の推奨: 1e-5）")
        recommendations.append("→ KL係数を増やす（現在の推奨: 0.3）")
        recommendations.append("→ cliprangeを小さくする（現在の推奨: 0.15）")

    # Issue 3: エントロピーの低下
    if entropy_values and np.min(entropy_values) < 0.02:
        issues.append(f"⚠️  エントロピーの極端な低下: 最小値 {np.min(entropy_values):.4f}")
        recommendations.append("→ temperatureを上げる（多様性向上）")
        recommendations.append("→ entropy_floorを設定して早期停止")

    # Issue 4: whiten_rewardsの影響
    if rewards and np.std(rewards) < 0.1:
        issues.append(f"ℹ️  報酬の標準偏差が小さい: {np.std(rewards):.4f}")
        recommendations.append("→ whiten_rewards=trueが正規化している（これ自体は問題ではない）")
        recommendations.append("→ 元の報酬スケールを確認するため、reward_breakdownを参照")

    if issues:
        print("検出された問題:")
        for issue in issues:
            print(f"  {issue}")
        print()

    if recommendations:
        print("推奨事項:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        print()

    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_log.py <log_dir>")
        sys.exit(1)

    analyze_log(sys.argv[1])
