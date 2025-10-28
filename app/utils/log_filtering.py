"""会話ログのフィルタリングユーティリティ"""

from typing import Dict, List


def filter_logs_by_human_count(
    logs: List[Dict],
    human_count: int,
    exclude_robot: bool = False
) -> List[Dict]:
    """
    人間発話数を基準にログをフィルタリング。
    直近 human_count 個の人間発話と、その間に含まれるロボット発話を全て返す。

    Args:
        logs: 全会話ログ
        human_count: 含める人間発話の数（max_history）
        exclude_robot: Trueの場合、ロボット発話を除外して人間発話のみ返す

    Returns:
        フィルタリングされたログ（人間 human_count 個 + その間のロボット発話）
        exclude_robot=Trueの場合は人間発話のみ

    Examples:
        >>> logs = [
        ...     {"speaker": "A", "utterance": "こんにちは"},
        ...     {"speaker": "ロボット", "utterance": "はい"},
        ...     {"speaker": "B", "utterance": "よろしく"},
        ... ]
        >>> filter_logs_by_human_count(logs, 1)  # 直近1人間発話+ロボット
        [{"speaker": "ロボット", "utterance": "はい"}, {"speaker": "B", "utterance": "よろしく"}]
        >>> filter_logs_by_human_count(logs, 1, exclude_robot=True)  # 直近1人間発話のみ
        [{"speaker": "B", "utterance": "よろしく"}]
    """
    if not logs:
        return []

    # 人間発話のインデックスを取得
    human_indices = [i for i, log in enumerate(logs) if log.get("speaker") != "ロボット"]

    if len(human_indices) <= human_count:
        # 人間発話が指定数以下なら全ログを返す（またはロボット除外）
        if exclude_robot:
            return [log for log in logs if log.get("speaker") != "ロボット"]
        else:
            return logs

    # 直近 human_count 個の人間発話の開始位置
    start_idx = human_indices[-human_count]

    # start_idx 以降のログを返す
    filtered = logs[start_idx:]

    # ロボット除外が指定されている場合は人間発話のみ
    if exclude_robot:
        return [log for log in filtered if log.get("speaker") != "ロボット"]

    return filtered
