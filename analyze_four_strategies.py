#!/usr/bin/env python3
"""
4戦略(validate, bridge, plan, 介入なし)の分析スクリプト

conversation_summary.jsonlから:
- 戦略使用頻度
- 平均報酬
- 安定化成功率（不安定→安定への遷移率）
を分析する
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_conversation_data(jsonl_path: Path) -> List[Dict]:
    """JSONLファイルを読み込む（複数行JSONオブジェクト対応）"""
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # '='*80 で分割してJSONオブジェクトを取得
    entries = content.split('=' * 80)
    
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        
        try:
            data.append(json.loads(entry))
        except json.JSONDecodeError as e:
            # デバッグ用に最初の100文字を表示
            preview = entry[:100].replace('\n', ' ')
            print(f"Warning: JSONデコードエラー: {e} - Preview: {preview}...")
            continue
    
    return data


def analyze_strategies(data: List[Dict]) -> Dict:
    """
    5戦略の分析を実行（validate, bridge, plan, no_intervention, output_error）
    
    Returns:
        Dict with keys: 'validate', 'bridge', 'plan', 'no_intervention', 'output_error'
        Each value contains: count, total_reward, success_count, attempts
    """
    stats = {
        'validate': {'count': 0, 'total_reward': 0.0, 'success_count': 0, 'attempts': 0},
        'bridge': {'count': 0, 'total_reward': 0.0, 'success_count': 0, 'attempts': 0},
        'plan': {'count': 0, 'total_reward': 0.0, 'success_count': 0, 'attempts': 0},
        'no_intervention': {'count': 0, 'total_reward': 0.0, 'success_count': 0, 'attempts': 0},
        'output_error': {'count': 0, 'total_reward': 0.0, 'success_count': 0, 'attempts': 0}
    }
    
    for entry in data:
        # 介入情報を取得
        intervention = entry.get('intervention', {})
        
        # interventionキーが存在しない場合はスキップ（安定状態で介入判定が行われなかったケース）
        if not intervention:
            continue
            
        intervened = intervention.get('intervened', False)
        
        # 戦略を決定
        if intervened:
            strategy = intervention.get('strategy', '').lower()
            if strategy not in ['validate', 'bridge', 'plan', 'output_error']:
                print(f"Warning: 未知の戦略 '{strategy}' (interaction_id={entry.get('interaction_id')})")
                continue
        else:
            # intervened: false の場合は、strategyを確認してno_interventionまたはoutput_errorを判定
            strategy = intervention.get('strategy', 'no_intervention').lower()
            if strategy not in ['no_intervention', 'output_error']:
                strategy = 'output_error'
        
        # 関係性の変化をチェック
        relation_before = entry.get('relation', {})
        relation_after = entry.get('relations_after_horizon', {})
        
        # 報酬を取得
        reward_info = entry.get('reward', {})
        total_reward = reward_info.get('total', 0.0)
        
        # カウント
        stats[strategy]['count'] += 1
        stats[strategy]['total_reward'] += total_reward
        
        # 安定化成功判定（不安定→安定への遷移）
        if relation_before and relation_after:
            was_unstable = not relation_before.get('is_stable', True)
            became_stable = relation_after.get('is_stable', False)
            
            if was_unstable:
                stats[strategy]['attempts'] += 1
                if became_stable:
                    stats[strategy]['success_count'] += 1
    
    return stats


def calculate_metrics(stats: Dict) -> Dict:
    """統計情報から各種メトリクスを計算"""
    metrics = {}
    
    for strategy, data in stats.items():
        count = data['count']
        total_reward = data['total_reward']
        success_count = data['success_count']
        attempts = data['attempts']
        
        avg_reward = total_reward / count if count > 0 else 0.0
        success_rate = (success_count / attempts * 100) if attempts > 0 else 0.0
        
        metrics[strategy] = {
            'count': count,
            'avg_reward': avg_reward,
            'success_count': success_count,
            'attempts': attempts,
            'success_rate': success_rate
        }
    
    return metrics


def print_analysis(metrics: Dict, total_interactions: int):
    """分析結果を表示"""
    print("=" * 80)
    print("5戦略分析結果 (validate, bridge, plan, no_intervention, output_error)")
    print("=" * 80)
    print()
    
    # 戦略の順序を定義
    strategy_order = ['validate', 'bridge', 'plan', 'no_intervention', 'output_error']
    strategy_labels = {
        'validate': 'VALIDATE',
        'bridge': 'BRIDGE',
        'plan': 'PLAN',
        'no_intervention': '介入なし',
        'output_error': 'OUTPUT_ERROR'
    }
    
    # サマリーテーブル
    print(f"{'戦略':<15} {'使用回数':>10} {'使用率':>10} {'平均報酬':>12} {'成功回数':>10} {'成功率':>10}")
    print("-" * 80)
    
    for strategy in strategy_order:
        if strategy not in metrics:
            continue
        
        m = metrics[strategy]
        label = strategy_labels.get(strategy, strategy)
        count = m['count']
        usage_rate = (count / total_interactions * 100) if total_interactions > 0 else 0.0
        avg_reward = m['avg_reward']
        success_count = m['success_count']
        success_rate = m['success_rate']
        
        print(f"{label:<15} {count:>10} {usage_rate:>9.1f}% {avg_reward:>12.4f} "
              f"{success_count:>10} {success_rate:>9.1f}%")
    
    print()
    print(f"総インタラクション数: {total_interactions}")
    print()
    
    # 詳細情報
    print("=" * 80)
    print("詳細情報")
    print("=" * 80)
    print()
    
    for strategy in strategy_order:
        if strategy not in metrics:
            continue
        
        m = metrics[strategy]
        label = strategy_labels.get(strategy, strategy)
        
        print(f"【{label}】")
        print(f"  使用回数: {m['count']} 回 ({m['count']/total_interactions*100:.1f}%)")
        print(f"  平均報酬: {m['avg_reward']:.4f}")
        print(f"  総報酬: {m['avg_reward'] * m['count']:.4f}")
        
        if m['attempts'] > 0:
            print(f"  安定化成功回数: {m['success_count']} 回")
            print(f"  安定化成功率: {m['success_rate']:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='4戦略(validate, bridge, plan, 介入なし)の分析'
    )
    parser.add_argument(
        'jsonl_path',
        type=str,
        nargs='?',
        default='app/logs/ppo_run-20251105-193813/conversation_summary.jsonl',  #ここを変える
        help='conversation_summary.jsonlのパス (デフォルト: ppo_run-20251105-193813)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='結果をJSONファイルに保存 (オプション)'
    )
    
    args = parser.parse_args()
    
    jsonl_path = Path(args.jsonl_path)
    
    if not jsonl_path.exists():
        print(f"Error: ファイルが見つかりません: {jsonl_path}")
        return
    
    print(f"データ読み込み中: {jsonl_path}")
    data = load_conversation_data(jsonl_path)
    print(f"読み込み完了: {len(data)} エントリ")
    print()
    
    # 分析実行
    stats = analyze_strategies(data)
    metrics = calculate_metrics(stats)
    
    # 結果表示
    print_analysis(metrics, len(data))
    
    # JSON出力（オプション）
    if args.output:
        output_path = Path(args.output)
        result = {
            'total_interactions': len(data),
            'metrics': metrics,
            'raw_stats': stats
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"結果を保存しました: {output_path}")


if __name__ == '__main__':
    main()
