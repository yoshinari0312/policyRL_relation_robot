#!/usr/bin/env python3
"""
edge_to_change自動選択ロジックのテスト

問題:
1. エッジキーが"A,B"形式で来ているのに"AB"形式を想定している
2. 絶対値が最小のエッジが正しく選択されていない
"""

def test_edge_selection():
    """エッジ選択ロジックのテスト"""
    
    # テストケース1: カンマ付きエッジキー
    edges_case1 = {
        "A,B": -0.8,
        "A,C": -0.7,
        "B,C": -0.6  # <- これが選ばれるべき（絶対値最小）
    }
    
    print("テストケース1: カンマ付きエッジキー")
    print(f"エッジ: {edges_case1}")
    
    # 負のエッジのみを抽出し、絶対値でソート
    negative_edges = []
    for edge_key, score in edges_case1.items():
        if score < 0:
            # edge_keyが文字列でない場合は変換
            if isinstance(edge_key, tuple):
                edge_str = "".join(str(c) for c in edge_key)
            else:
                edge_str = str(edge_key)
            negative_edges.append((edge_str, score))
            print(f"  負エッジ追加: {edge_str} = {score} (絶対値: {abs(score)})")
    
    if negative_edges:
        # 絶対値が最小（最も0に近い）負のエッジを選択
        negative_edges.sort(key=lambda x: abs(x[1]))
        edge_to_change = negative_edges[0][0]
        print(f"  選択されたエッジ（ソート前）: {negative_edges}")
        print(f"  選択されたエッジ: {edge_to_change} (スコア: {negative_edges[0][1]}, 絶対値: {abs(negative_edges[0][1])})")
    
    # 正規化
    edge_to_change_normalized = edge_to_change.replace(",", "").upper()
    print(f"  正規化後: {edge_to_change_normalized}")
    
    # 期待値チェック
    expected = "BC"
    if edge_to_change_normalized == expected:
        print(f"✅ テスト1成功: {edge_to_change_normalized} == {expected}")
    else:
        print(f"❌ テスト1失敗: {edge_to_change_normalized} != {expected}")
    
    print()
    
    # テストケース2: タプル形式のエッジキー
    edges_case2 = {
        ("A", "B"): -0.8,
        ("A", "C"): -0.7,
        ("B", "C"): -0.6  # <- これが選ばれるべき
    }
    
    print("テストケース2: タプル形式のエッジキー")
    print(f"エッジ: {edges_case2}")
    
    negative_edges2 = []
    for edge_key, score in edges_case2.items():
        if score < 0:
            if isinstance(edge_key, tuple):
                edge_str = "".join(str(c) for c in edge_key)
            else:
                edge_str = str(edge_key)
            negative_edges2.append((edge_str, score))
            print(f"  負エッジ追加: {edge_str} = {score} (絶対値: {abs(score)})")
    
    if negative_edges2:
        negative_edges2.sort(key=lambda x: abs(x[1]))
        edge_to_change2 = negative_edges2[0][0]
        print(f"  選択されたエッジ: {edge_to_change2} (スコア: {negative_edges2[0][1]}, 絶対値: {abs(negative_edges2[0][1])})")
    
    edge_to_change2_normalized = edge_to_change2.replace(",", "").upper()
    print(f"  正規化後: {edge_to_change2_normalized}")
    
    if edge_to_change2_normalized == expected:
        print(f"✅ テスト2成功: {edge_to_change2_normalized} == {expected}")
    else:
        print(f"❌ テスト2失敗: {edge_to_change2_normalized} != {expected}")
    
    print()
    
    # テストケース3: 逆順のエッジキー（B,C vs C,B）
    edges_case3 = {
        "A,B": -0.8,
        "C,A": -0.7,  # <- CAの逆順
        "B,C": -0.6   # <- これが選ばれるべき
    }
    
    print("テストケース3: 逆順のエッジキー")
    print(f"エッジ: {edges_case3}")
    
    negative_edges3 = []
    for edge_key, score in edges_case3.items():
        if score < 0:
            if isinstance(edge_key, tuple):
                edge_str = "".join(str(c) for c in edge_key)
            else:
                edge_str = str(edge_key)
            negative_edges3.append((edge_str, score))
            print(f"  負エッジ追加: {edge_str} = {score} (絶対値: {abs(score)})")
    
    if negative_edges3:
        negative_edges3.sort(key=lambda x: abs(x[1]))
        edge_to_change3 = negative_edges3[0][0]
        print(f"  選択されたエッジ: {edge_to_change3} (スコア: {negative_edges3[0][1]}, 絶対値: {abs(negative_edges3[0][1])})")
    
    edge_to_change3_normalized = edge_to_change3.replace(",", "").upper()
    print(f"  正規化前: {edge_to_change3}")
    print(f"  正規化後: {edge_to_change3_normalized}")
    
    # 正規化処理の確認
    if edge_to_change3_normalized not in {"AB", "BC", "CA"}:
        print(f"  ⚠️ 正規化が必要: {edge_to_change3_normalized}")
        if len(edge_to_change3_normalized) == 2 and edge_to_change3_normalized[1] + edge_to_change3_normalized[0] in {"AB", "BC", "CA"}:
            edge_to_change3_normalized = edge_to_change3_normalized[1] + edge_to_change3_normalized[0]
            print(f"  → 反転: {edge_to_change3_normalized}")
    
    if edge_to_change3_normalized == expected:
        print(f"✅ テスト3成功: {edge_to_change3_normalized} == {expected}")
    else:
        print(f"❌ テスト3失敗: {edge_to_change3_normalized} != {expected}")


if __name__ == "__main__":
    test_edge_selection()
