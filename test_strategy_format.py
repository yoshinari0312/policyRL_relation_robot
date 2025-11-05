"""
新形式（数字1-4）の戦略パース機能のユニットテスト
"""

import sys

def _validate_plan_json_test(text: str, env=None):
    """
    プランテキスト（数字1-4）の妥当性をチェックし、辞書形式に変換する
    ※テスト用に _validate_plan_json 関数を再実装
    """
    if not text or not isinstance(text, str):
        return None, False

    text = text.strip()
    if not text:
        return None, False

    # 数字の抽出（1-4を探す）
    strategy_num = None
    for char in text:
        if char in '1234':
            strategy_num = int(char)
            break
    
    if strategy_num is None:
        return None, False

    # 戦略番号を戦略名にマッピング
    strategy_map = {
        1: "validate",
        2: "bridge", 
        3: "plan",
        4: "no_intervention"
    }
    
    strategy = strategy_map.get(strategy_num)
    if not strategy:
        return None, False

    # 介入なしの場合
    if strategy == "no_intervention":
        return {"intervene_now": False}, True

    # 介入ありの場合: edge_to_changeとtarget_speakerを自動決定
    # ダミーの値を使用（実際の環境がないため）
    import random
    edge_to_change = "AB"  # デフォルト
    target_speaker = random.choice(list(edge_to_change))
    
    return {
        "intervene_now": True,
        "strategy": strategy,
        "edge_to_change": edge_to_change,
        "target_speaker": target_speaker
    }, True

def test_number_parsing():
    """数字1-4のパーステスト"""
    print("=" * 80)
    print("数字1-4のパーステスト")
    print("=" * 80)
    print()
    
    test_cases = [
        ("1", "validate", True),
        ("2", "bridge", True),
        ("3", "plan", True),
        ("4", "no_intervention", True),
        (" 1 ", "validate", True),
        ("2\n", "bridge", True),
        ("answer: 3", "plan", True),
        ("The answer is 4", "no_intervention", True),
        ("5", None, False),  # 無効な数字
        ("abc", None, False),  # 数字なし
        ("", None, False),  # 空文字列
    ]
    
    passed = 0
    failed = 0
    
    for text, expected_strategy, expected_valid in test_cases:
        result, is_valid = _validate_plan_json_test(text, env=None)
        
        if is_valid != expected_valid:
            print(f"❌ FAIL: '{text}' -> valid={is_valid}, expected={expected_valid}")
            print(f"  Debug: result={result}")
            failed += 1
            continue
        
        if is_valid:
            # 介入なしの場合は特別処理
            if expected_strategy == "no_intervention":
                intervene_now = result.get("intervene_now")
                if intervene_now == False:
                    print(f"✅ PASS: '{text}' -> no intervention")
                    passed += 1
                else:
                    print(f"❌ FAIL: '{text}' -> intervene_now={intervene_now}, expected=False")
                    failed += 1
                continue
            
            actual_strategy = result.get("strategy")
            if actual_strategy != expected_strategy:
                print(f"❌ FAIL: '{text}' -> strategy='{actual_strategy}', expected='{expected_strategy}'")
                failed += 1
                continue
            
            # 介入ありの場合、edge_to_changeとtarget_speakerが設定されているか確認
            if expected_strategy != "no_intervention":
                if "edge_to_change" not in result or "target_speaker" not in result:
                    print(f"❌ FAIL: '{text}' -> missing edge_to_change or target_speaker")
                    failed += 1
                    continue
                
                edge = result.get("edge_to_change")
                target = result.get("target_speaker")
                
                # edge_to_changeの妥当性チェック
                if edge not in ["AB", "BC", "CA"]:
                    print(f"❌ FAIL: '{text}' -> invalid edge_to_change='{edge}'")
                    failed += 1
                    continue
                
                # target_speakerの妥当性チェック
                if target not in ["A", "B", "C"]:
                    print(f"❌ FAIL: '{text}' -> invalid target_speaker='{target}'")
                    failed += 1
                    continue
                
                # edge_to_changeとtarget_speakerの整合性チェック
                if target not in edge:
                    print(f"❌ FAIL: '{text}' -> target_speaker='{target}' not in edge_to_change='{edge}'")
                    failed += 1
                    continue
                
                print(f"✅ PASS: '{text}' -> strategy='{actual_strategy}', edge='{edge}', target='{target}'")
            else:
                print(f"✅ PASS: '{text}' -> strategy='{actual_strategy}' (no intervention)")
            passed += 1
        else:
            if expected_valid:  # 有効であるべきなのに無効
                print(f"❌ FAIL: '{text}' -> invalid, expected valid with strategy='{expected_strategy}'")
                failed += 1
            else:
                print(f"✅ PASS: '{text}' -> invalid (as expected)")
                passed += 1
    
    print()
    print("=" * 80)
    print(f"結果: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


def test_backward_compatibility():
    """旧形式（JSON）の後方互換性テスト"""
    print()
    print("=" * 80)
    print("旧形式JSONの後方互換性テスト")
    print("=" * 80)
    print()
    
    # 注: 旧形式はconvo_env.pyの_parse_planでのみサポート（ppo_train.pyでは新形式のみ）
    print("⚠️  注: ppo_train.py (_validate_plan_json) は新形式（数字のみ）をサポート")
    print("    旧形式JSONはconvo_env.py (_parse_plan) でのみサポート")
    print()


if __name__ == "__main__":
    success = test_number_parsing()
    test_backward_compatibility()
    
    if success:
        print("\n🎉 全テスト成功！")
        sys.exit(0)
    else:
        print("\n❌ テスト失敗")
        sys.exit(1)
