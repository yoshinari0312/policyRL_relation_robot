#!/bin/bash
# エントロピーボーナスの動作確認スクリプト

echo "=== エントロピーボーナス動作確認 ==="
echo ""

# 最新のログディレクトリを取得
LATEST_LOG=$(ls -dt app/logs/ppo_run-* 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ ログディレクトリが見つかりません"
    exit 1
fi

echo "📁 ログディレクトリ: $LATEST_LOG"
echo ""

# Dockerコンテナのログからエントロピーボーナス情報を確認
echo "--- Dockerログ（エントロピーボーナス）---"
docker logs rl-convo-policy-app-1 2>&1 | grep "\[ENTROPY_BONUS\]" | tail -10
echo ""

# metrics.jsonlからtrain/loss/total_with_entropyを確認
echo "--- metrics.jsonl (train/loss/total_with_entropy) ---"
if [ -f "$LATEST_LOG/metrics.jsonl" ]; then
    grep "train/loss/total_with_entropy" "$LATEST_LOG/metrics.jsonl" | tail -5
else
    echo "❌ metrics.jsonl が見つかりません"
fi
echo ""

# wandbの状態を確認
echo "--- wandb 設定 ---"
grep -A 3 "^wandb:" app/config.local.yaml
echo ""

echo "=== 確認項目 ==="
echo "1. [ENTROPY_BONUS] ログが出力されているか"
echo "2. entropy_avg が 0.5 以上か（理想的な探索）"
echo "3. train/loss/total_with_entropy がメトリクスに記録されているか"
echo ""
echo "=== entropy_coef 調整の目安 ==="
echo "- entropy_avg < 0.3 → entropy_coef を 0.1 に増やす"
echo "- 0.3 ≤ entropy_avg < 0.5 → entropy_coef を 0.07 に増やす"
echo "- 0.5 ≤ entropy_avg < 1.0 → 現状維持（0.05）"
echo "- entropy_avg ≥ 1.0 → entropy_coef を 0.03 に減らす"
echo "- KL divergence > 0.3 → entropy_coef を減らす"
