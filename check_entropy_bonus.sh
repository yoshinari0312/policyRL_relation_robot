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

# Dockerコンテナのログから全てのデバッグ情報を確認
echo "--- Dockerログ（初期化）---"
docker logs rl-convo-policy-app-1 2>&1 | grep "\[PPOTrainerWithEntropyBonus\]" | tail -5
echo ""

echo "--- Dockerログ（Trainer初期化後）---"
docker logs rl-convo-policy-app-1 2>&1 | grep "\[TRAINER_DEBUG\]" | tail -10
echo ""

echo "--- Dockerログ（学習開始）---"
docker logs rl-convo-policy-app-1 2>&1 | grep "\[TRAIN_DEBUG\]" | tail -10
echo ""

echo "--- Dockerログ（compute_loss呼び出し）---"
docker logs rl-convo-policy-app-1 2>&1 | grep "\[COMPUTE_LOSS_DEBUG\]" | tail -15
echo ""

echo "--- Dockerログ（エントロピーデバッグ）---"
docker logs rl-convo-policy-app-1 2>&1 | grep "\[ENTROPY_DEBUG\]" | tail -20
echo ""

echo "--- Dockerログ（エントロピーボーナス）---"
docker logs rl-convo-policy-app-1 2>&1 | grep "\[ENTROPY_BONUS\]" | tail -10
echo ""

# metrics.jsonlからtrain/loss/total_with_entropyを確認
echo "--- metrics.jsonl (train/loss/total_with_entropy) ---"
if [ -f "$LATEST_LOG/metrics.jsonl" ]; then
    grep "train/loss/total_with_entropy" "$LATEST_LOG/metrics.jsonl" | tail -5
    if [ $? -ne 0 ]; then
        echo "❌ train/loss/total_with_entropy が見つかりません"
    fi
else
    echo "❌ metrics.jsonl が見つかりません"
fi
echo ""

echo "=== トラブルシューティング ==="
echo "1. [PPOTrainerWithEntropyBonus] が出ているか → Trainerが初期化されている"
echo "2. [TRAINER_DEBUG] が出ているか → Trainer初期化後の状態"
echo "3. [TRAIN_DEBUG] が出ているか → trainer.train() が呼ばれている"
echo "4. [COMPUTE_LOSS_DEBUG] が出ているか → compute_loss() が呼ばれている"
echo "5. [ENTROPY_DEBUG] が出ているか → log_historyの中身を確認できる"
echo "6. [ENTROPY_BONUS] が出ているか → エントロピーボーナスが適用されている"
