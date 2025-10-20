import os
from env.convo_env import ConversationEnv
from config import get_config

def main():
    cfg = get_config()
    backend  = cfg.scorer.backend
    personas = cfg.env.personas

    env = ConversationEnv(
        max_steps=cfg.env.max_steps,
        personas=personas,
        include_robot=cfg.env.include_robot,
        ema_alpha=cfg.scorer.ema_alpha,
        decay_factor=cfg.scorer.decay_factor,
        backend=backend,
        debug=cfg.env.debug,
    )

    state = env.reset()
    print("=== START ===")
    print(state)

    done = False
    step_count = 0

    while not done:
        step_count += 1
        action = "最近楽しかったことはなんですか？"
        state, reward, done, info = env.step(action)
        dbg = info.get("debug") or {}

        # ====== 表示はここで一括。内部は print しないため、順序が乱れません ======
        # print(f"\n=== ステップ {step_count} ===")
        # # 事前状態
        # print("［事前状態（行動前）］")
        # print("  参加者:", dbg.get("participants_before"))
        # # metrics のトレース（日本語整形）
        # for line in dbg.get("trace_metrics_before", []):
        #     print("  " + line)
        # # ロボット行動
        # print("［ロボットの発話］")
        # print(f"  🤖 {action}")
        # # 事後状態
        # print("［事後状態（行動後）］")
        # print("  参加者:", dbg.get("participants_after"))
        # print("  発話数:", dbg.get("utterance_counts_after"))
        # # スコアリングのトレース（α計算など）
        # print("  〈関係スコアの計算過程〉")
        # for line in dbg.get("trace_scores_after", []):
        #     print("   ", line)
        # print("  〈ネットワーク指標〉")
        # for line in dbg.get("trace_metrics_after", []):
        #     print("   ", line)
        # # 会話ログ
        # print("［会話履歴（全体）］")
        # for log in env.logs:
        #     print(f"  [{log['speaker']}] {log['utterance']}")
        # # 報酬
        # print(f"［報酬］ {reward}")
        before = dbg.get("rel_before") or {}
        after  = dbg.get("rel_after") or {}
        tri_before = before.get("triangles", [])
        tri_after  = after.get("triangles", [])
        ub_before = before.get("unstable_triads", 0)
        ub_after  = after.get("unstable_triads", 0)

        print(f"\n=== ステップ {step_count} ===")
        print("［ロボットの発話］")
        print(f"  🤖 {action}")
        print("［事前→事後の変化］")
        print(f"  不安定三角形: {ub_before} → {ub_after}（Δ {ub_after - ub_before}）")
        if tri_after != tri_before:
            print(f"  三角形（事後）: {tri_after}")
        # α計算などは“事後”のみを表示（重複回避）
        trace_after = dbg.get("trace_scores_after", [])
        if trace_after:
            print("  関係スコア更新（事後）:")
            for line in trace_after:
                print("   ", line)
        # 会話は直近の追加分のみを表示
        added = env.logs[-(1 + len(personas)):] if len(env.logs) >= 1 + len(personas) else env.logs
        print("［直近の発話（ロボ＋人間の応答）］")
        for log in added:
            print(f"  [{log['speaker']}] {log['utterance']}")
        print(f"［報酬］ {reward}")
    print("DONE")

if __name__ == "__main__":
    main()
