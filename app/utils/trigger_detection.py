"""地雷（トリガー）検出ユーティリティ"""

from typing import Dict, List, Set


def get_common_triggers_for_others(
    current_speaker: str,
    all_triggers: Dict[str, List[str]]
) -> List[str]:
    """
    現在の話者以外の2人に共通する地雷を検出する（自分の地雷を除外）

    Args:
        current_speaker: 現在の話者（例: "A"）
        all_triggers: 各話者の地雷リスト（例: {"A": [...], "B": [...], "C": [...]}）

    Returns:
        他の2人に共通する地雷のリスト（自分の地雷は除外、空文字列も除外）

    Example:
        >>> triggers = {
        ...     "A": ["お金", "勉強", "テクノロジー"],
        ...     "B": ["お金", "勉強", "ゲーム"],
        ...     "C": ["お金", "テクノロジー", "ゲーム"]
        ... }
        >>> get_common_triggers_for_others("A", triggers)
        ['ゲーム']  # BとCに共通する地雷で、Aにはない地雷
    """
    # 現在の話者以外の話者を取得
    other_speakers = [s for s in all_triggers.keys() if s != current_speaker]

    if len(other_speakers) < 2:
        # 他の話者が2人未満の場合は共通地雷を検出できない
        return []

    # 各話者の地雷セットを作成（空文字列を除外）
    trigger_sets = [
        set(t.strip() for t in all_triggers.get(speaker, []) if t and t.strip())
        for speaker in other_speakers
    ]

    if not trigger_sets:
        return []

    # 2人の共通地雷を取得（積集合）
    common_triggers_set = trigger_sets[0]
    for trigger_set in trigger_sets[1:]:
        common_triggers_set = common_triggers_set.intersection(trigger_set)

    # 自分の地雷を除外
    my_triggers = set(
        t.strip() for t in all_triggers.get(current_speaker, []) if t and t.strip()
    )
    common_triggers_set = common_triggers_set - my_triggers

    # リストとして返す
    return sorted(list(common_triggers_set))


def format_common_triggers_message(common_triggers: List[str], other_speakers: List[str]) -> str:
    """
    共通地雷に関するメッセージをフォーマットする

    Args:
        common_triggers: 共通地雷のリスト
        other_speakers: 他の話者のリスト（例: ["B", "C"]）

    Returns:
        フォーマットされたメッセージ文字列

    Example:
        >>> format_common_triggers_message(["お金", "ゲーム"], ["B", "C"])
        '他の参加者（B, C）に共通する不機嫌トリガー：お金, ゲーム。
         これらの話題になった時は、B と C に優しく穏やかに接してください。'
    """
    if not common_triggers:
        return ""

    speakers_str = ", ".join(other_speakers)
    triggers_str = ", ".join(common_triggers)

    return (
        f"他の参加者（{speakers_str}）に共通する不機嫌トリガー：{triggers_str}。"
        f"これらの話題になった時は、{' と '.join(other_speakers)} に優しく穏やかに接してください。"
    )
