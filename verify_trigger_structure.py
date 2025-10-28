#!/usr/bin/env python3
"""地雷構造の検証スクリプト"""

from app.config import get_config


def verify_trigger_structure():
    """地雷の構造を検証"""
    print("=" * 60)
    print("地雷構造の検証")
    print("=" * 60)
    print()

    cfg = get_config()
    personas_cfg = getattr(cfg.env, "personas", None)

    if not personas_cfg or not isinstance(personas_cfg, dict):
        print("エラー: personas設定が見つかりません")
        return

    # 各ペルソナの地雷を取得
    all_triggers = {}
    for persona_name in personas_cfg.keys():
        persona_info = personas_cfg.get(persona_name, {})
        if isinstance(persona_info, dict):
            persona_triggers = persona_info.get("triggers", []) or []
            all_triggers[persona_name] = [t for t in persona_triggers if t and t.strip()]

    # セットに変換
    sets = {name: set(triggers) for name, triggers in all_triggers.items()}

    print("【基本情報】")
    print("-" * 60)
    for name, triggers in all_triggers.items():
        print(f"ペルソナ {name}: {len(triggers)}個の地雷")
    print()

    # 共通地雷の計算
    print("【共通地雷の分析】")
    print("-" * 60)

    # 3人共通
    all_three = sets['A'] & sets['B'] & sets['C']
    print(f"3人共通（---）: {len(all_three)}個")
    print(f"  {sorted(all_three)}")
    print()

    # AB共通（Cにはない）
    ab_only = (sets['A'] & sets['B']) - sets['C']
    print(f"AB共通のみ（--+）: {len(ab_only)}個")
    print(f"  {sorted(ab_only)}")
    print()

    # BC共通（Aにはない）
    bc_only = (sets['B'] & sets['C']) - sets['A']
    print(f"BC共通のみ（+--）: {len(bc_only)}個")
    print(f"  {sorted(bc_only)}")
    print()

    # CA共通（Bにはない）
    ca_only = (sets['C'] & sets['A']) - sets['B']
    print(f"CA共通のみ（-+-）: {len(ca_only)}個")
    print(f"  {sorted(ca_only)}")
    print()

    # 1人だけの地雷（安定パターン+--を形成）
    a_only = sets['A'] - sets['B'] - sets['C']
    b_only = sets['B'] - sets['A'] - sets['C']
    c_only = sets['C'] - sets['A'] - sets['B']

    print("【1人だけの地雷（避けるべき）】")
    print("-" * 60)
    print(f"Aのみ（+--パターン形成）: {len(a_only)}個")
    if a_only:
        print(f"  {sorted(a_only)}")
    print(f"Bのみ（-+-パターン形成）: {len(b_only)}個")
    if b_only:
        print(f"  {sorted(b_only)}")
    print(f"Cのみ（--+パターン形成）: {len(c_only)}個")
    if c_only:
        print(f"  {sorted(c_only)}")
    print()

    # 各ペアの共通地雷数
    print("【各ペアの共通地雷数】")
    print("-" * 60)
    ab_common = sets['A'] & sets['B']
    bc_common = sets['B'] & sets['C']
    ca_common = sets['C'] & sets['A']
    print(f"AB共通: {len(ab_common)}個 (3人共通 {len(all_three)} + AB専用 {len(ab_only)})")
    print(f"BC共通: {len(bc_common)}個 (3人共通 {len(all_three)} + BC専用 {len(bc_only)})")
    print(f"CA共通: {len(ca_common)}個 (3人共通 {len(all_three)} + CA専用 {len(ca_only)})")
    print()

    # 検証結果
    print("【検証結果】")
    print("-" * 60)

    all_ok = True

    # 1. ---と++-系が同数か
    two_only_total = len(ab_only) + len(bc_only) + len(ca_only)
    if len(all_three) == two_only_total:
        print(f"✓ ---パターン({len(all_three)}個) と ++-系パターン({two_only_total}個) が同数")
    else:
        print(f"✗ ---パターン({len(all_three)}個) と ++-系パターン({two_only_total}個) が異なる")
        all_ok = False

    # 2. AB, BC, CAの共通数が同じか
    if len(ab_common) == len(bc_common) == len(ca_common):
        print(f"✓ AB/BC/CA共通地雷数が同じ（各{len(ab_common)}個）")
    else:
        print(f"✗ AB/BC/CA共通地雷数が異なる（AB:{len(ab_common)}, BC:{len(bc_common)}, CA:{len(ca_common)}）")
        all_ok = False

    # 3. 1人だけの地雷がないか
    if len(a_only) == 0 and len(b_only) == 0 and len(c_only) == 0:
        print(f"✓ 1人だけの地雷がない（安定パターン+--を回避）")
    else:
        print(f"✗ 1人だけの地雷がある（A:{len(a_only)}, B:{len(b_only)}, C:{len(c_only)}）")
        all_ok = False

    # 4. 各ペルソナの地雷数が同じか
    trigger_counts = [len(triggers) for triggers in all_triggers.values()]
    if len(set(trigger_counts)) == 1:
        print(f"✓ 各ペルソナの地雷数が同じ（各{trigger_counts[0]}個）")
    else:
        print(f"✗ 各ペルソナの地雷数が異なる: {trigger_counts}")
        all_ok = False

    print()
    if all_ok:
        print("✓✓✓ すべての検証に合格しました！")
    else:
        print("✗✗✗ 一部の検証に失敗しました")

    print()


if __name__ == "__main__":
    verify_trigger_structure()
