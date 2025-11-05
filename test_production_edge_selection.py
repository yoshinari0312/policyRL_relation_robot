#!/usr/bin/env python3
"""
プロダクション環境でのエッジ選択テスト

ユーザーが報告したバグ:
  入力: {"A,B": -0.8, "A,C": -0.7, "B,C": -0.6}
  期待: BC (abs=0.6, 最小)
  実際: CA (abs=0.7, 不正解)

このスクリプトは実際の環境でエッジ選択の動作を確認します。
"""
import sys
sys.path.insert(0, '/home/yoshinari/projects/rl-convo-policy/app')

from config import get_config
from env.convo_env import ConversationEnv

def test_production_edge_selection():
    """プロダクション環境でエッジ選択をテスト"""
    
    print("=" * 80)
    print("プロダクション環境でのエッジ選択テスト")
    print("=" * 80)
    
    # 設定を読み込み
    cfg = get_config()
    
    # 環境を初期化（デバッグモード有効）
    env = ConversationEnv(
        max_steps=8,
        personas=getattr(cfg.env, "personas", {}),
        include_robot=True,
        max_history=6,
        backend=getattr(cfg.scorer, "backend", "azure"),
        decay_factor=getattr(cfg.scorer, "decay_factor", 1.5),
        debug=True,  # デバッグモード有効 - エッジ選択のログが出力される
        reward_backend=getattr(cfg.env, "reward_backend", "rule"),
        evaluation_horizon=getattr(cfg.env, "evaluation_horizon", 3),
        time_penalty=getattr(cfg.env, "time_penalty", 0.01),
        terminal_bonus=getattr(cfg.env, "terminal_bonus", 0.25),
        intervention_cost=getattr(cfg.env, "intervention_cost", 0.02),
    )
    
    print("\n1. 環境をリセット")
    observation = env.reset()
    print(f"   話題: {env.current_topic}")
    print(f"   ペルソナ: {env.persona_pool}")
    
    # 関係性スナップショットを取得
    print("\n2. 現在の関係性スナップショットを取得")
    snapshot = env.relation_snapshot()
    print(f"   スナップショットのキー: {list(snapshot.keys())}")
    
    # エッジを抽出
    edges = snapshot.get('edges', {})
    if isinstance(edges, dict) and 'edges' in edges:
        # ネストされている場合
        edges = edges['edges']
    
    print(f"   エッジ数: {len(edges)}")
    if edges:
        print(f"   エッジキーの型: {type(list(edges.keys())[0])}")
    
    # エッジの内容を表示
    print("\n3. 全エッジの内容:")
    for edge_key, score in edges.items():
        if isinstance(score, (int, float)):
            print(f"   {edge_key} ({type(edge_key).__name__}): {score:.4f}")
    
    # マイナスのエッジを抽出
    print("\n4. マイナスのエッジを抽出:")
    negative_edges = [(k, v) for k, v in edges.items() if v < 0]
    for edge_key, score in negative_edges:
        print(f"   {edge_key}: {score:.4f} (abs={abs(score):.4f})")
    
    if negative_edges:
        # 絶対値でソート
        print("\n5. 絶対値でソート:")
        sorted_edges = sorted(negative_edges, key=lambda x: abs(x[1]))
        for edge_key, score in sorted_edges:
            print(f"   {edge_key}: {score:.4f} (abs={abs(score):.4f})")
        
        # 最小絶対値のエッジを選択
        selected_edge = sorted_edges[0]
        print(f"\n6. 選択されたエッジ: {selected_edge[0]} (score={selected_edge[1]:.4f}, abs={abs(selected_edge[1]):.4f})")
        
        # 正規化
        if isinstance(selected_edge[0], tuple):
            normalized = "".join(sorted(selected_edge[0]))
        else:
            normalized = "".join(sorted(str(selected_edge[0]).replace(",", "").replace(" ", "")))
        print(f"   正規化後: {normalized}")
    else:
        print("\n   マイナスのエッジがありません")
    
    # 実際に介入を実行して、_parse_plan()のデバッグ出力を見る
    print("\n" + "=" * 80)
    print("実際に介入を実行（strategy=3=plan）")
    print("=" * 80)
    observation, reward, done, info = env.step('3')
    
    print(f"\n結果:")
    print(f"  - 介入: {info.get('intervened', False)}")
    print(f"  - edge_to_change: {info.get('plan', {}).get('edge_to_change', 'N/A')}")
    print(f"  - target_speaker: {info.get('plan', {}).get('target_speaker', 'N/A')}")
    print(f"  - 報酬: {reward:.4f}")
    
    print("\n" + "=" * 80)
    print("テスト完了")
    print("=" * 80)


if __name__ == "__main__":
    test_production_edge_selection()
