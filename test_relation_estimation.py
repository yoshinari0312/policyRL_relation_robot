#!/usr/bin/env python3
"""関係性推定のテストスクリプト"""

import sys
import os

# appディレクトリをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from relation_scorer import RelationScorer
from config import get_config


def test_relation_estimation():
    """関係性推定の動作をテスト"""
    print("=" * 80)
    print("関係性推定テスト")
    print("=" * 80)
    print()

    cfg = get_config()
    llm_cfg = getattr(cfg, "llm", None)

    if not llm_cfg:
        print("❌ エラー: LLM設定が見つかりません")
        return

    print("【設定情報】")
    print("-" * 80)
    print(f"Provider: {getattr(llm_cfg, 'provider', 'N/A')}")
    print(f"Endpoint: {getattr(llm_cfg, 'azure_endpoint', 'N/A')}")
    print(f"Relation Model: {getattr(llm_cfg, 'relation_model', 'N/A')}")
    print(f"Backend: {getattr(cfg.scorer, 'backend', 'N/A')}")
    print()

    # RelationScorerを初期化（設定から use_ema などを読み込む）
    print("【RelationScorer 初期化】")
    print("-" * 80)

    # 設定からスコアラーのパラメータを取得
    scorer_cfg = getattr(cfg, "scorer", None)
    use_ema = getattr(scorer_cfg, "use_ema", True) if scorer_cfg else True
    if use_ema is None:
        use_ema = False
    decay_factor = getattr(scorer_cfg, "decay_factor", 1.5) if scorer_cfg else 1.5
    backend = getattr(scorer_cfg, "backend", "azure") if scorer_cfg else "azure"

    print(f"use_ema: {use_ema}")
    print(f"decay_factor: {decay_factor}")
    print(f"backend: {backend}")
    print()

    try:
        scorer = RelationScorer(
            backend=backend,
            use_ema=use_ema,
            decay_factor=decay_factor,
            verbose=True
        )
        print("✓ RelationScorer初期化成功")
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return

    print()

    # テストケース1: 穏やかな会話（+++パターン）
    print("【テストケース1】穏やかな会話（期待: +++パターン）")
    print("-" * 80)
    test_logs_1 = [
        {"speaker": "A", "utterance": "今日はいい天気ですね。"},
        {"speaker": "B", "utterance": "本当ですね。散歩日和ですよ。"},
        {"speaker": "C", "utterance": "公園に行きませんか？"},
        {"speaker": "A", "utterance": "いいですね、行きましょう。"},
    ]

    print("会話履歴:")
    for log in test_logs_1:
        print(f"  [{log['speaker']}] {log['utterance']}")
    print()

    try:
        participants = ["A", "B", "C"]
        result = scorer.get_scores(test_logs_1, participants, return_trace=True, update_state=True)

        # return_trace=Trueの場合は(scores, trace)のタプルが返る
        if isinstance(result, tuple):
            scores, trace = result
            print(f"トレース: {trace}")
        else:
            scores = result

        print("関係性スコア:")
        for (p1, p2), score in sorted(scores.items()):
            print(f"  {p1}-{p2}: {score:+.3f}")
        print()

        # 構造バランス分析
        from network_metrics import analyze_relations_from_scores
        analysis = analyze_relations_from_scores(scores)
        print(f"不安定トライアド数: {analysis['unstable_triads']}")
        print(f"安定トライアド数: {analysis['balanced_triads']}")
        print(f"安定状態: {'はい' if analysis['unstable_triads'] == 0 else 'いいえ'}")
        print()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        print()

    # テストケース2: 対立のある会話（---パターン）
    print("【テストケース2】対立のある会話（期待: ---パターン）")
    print("-" * 80)
    test_logs_2 = [
        {"speaker": "A", "utterance": "お前ら本当にうざいな。"},
        {"speaker": "B", "utterance": "何だよ、お前こそ黙れよ。"},
        {"speaker": "C", "utterance": "2人とも最悪だな。話にならねえ。"},
        {"speaker": "A", "utterance": "Cも大概だろ。"},
    ]

    print("会話履歴:")
    for log in test_logs_2:
        print(f"  [{log['speaker']}] {log['utterance']}")
    print()

    try:
        result = scorer.get_scores(test_logs_2, participants, return_trace=True, update_state=True)
        # return_trace=Trueの場合は(scores, trace)のタプルが返る
        if isinstance(result, tuple):
            scores, trace = result
            print(f"トレース: {trace}")
        else:
            scores = result

        print("関係性スコア:")
        for (p1, p2), score in sorted(scores.itaaaems()):
            print(f"  {p1}-{p2}: {score:+.3f}")
        print()

        # 構造バランス分析
        analysis = analyze_relations_from_scores(scores)
        print(f"不安定トライアド数: {analysis['unstable_triads']}")
        print(f"安定トライアド数: {analysis['balanced_triads']}")
        print(f"安定状態: {'はい' if analysis['unstable_triads'] == 0 else 'いいえ'}")
        print()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        print()

    # テストケース3: 2人が仲良く1人が孤立（++-パターン）
    print("【テストケース3】2人が仲良く1人が孤立（期待: ++-パターン）")
    print("-" * 80)
    test_logs_3 = [
        {"speaker": "A", "utterance": "Bさん、一緒に行きましょう。"},
        {"speaker": "B", "utterance": "いいですね、Aさん。"},
        {"speaker": "C", "utterance": "私も行きたいんですが..."},
        {"speaker": "A", "utterance": "Cさんは来なくていいです。"},
        {"speaker": "B", "utterance": "そうですね、Aさんと2人で行きましょう。"},
    ]

    print("会話履歴:")
    for log in test_logs_3:
        print(f"  [{log['speaker']}] {log['utterance']}")
    print()

    try:
        result = scorer.get_scores(test_logs_3, participants, return_trace=True, update_state=True)

        # return_trace=Trueの場合は(scores, trace)のタプルが返る
        if isinstance(result, tuple):
            scores, trace = result
            print(f"トレース: {trace}")
        else:
            scores = result

        print("関係性スコア:")
        for (p1, p2), score in sorted(scores.items()):
            print(f"  {p1}-{p2}: {score:+.3f}")
        print()

        # 構造バランス分析
        analysis = analyze_relations_from_scores(scores)
        print(f"不安定トライアド数: {analysis['unstable_triads']}")
        print(f"安定トライアド数: {analysis['balanced_triads']}")
        print(f"安定状態: {'はい' if analysis['unstable_triads'] == 0 else 'いいえ'}")
        print()

    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        print()

    print("=" * 80)
    print("テスト完了！")
    print("=" * 80)


if __name__ == "__main__":
    test_relation_estimation()
