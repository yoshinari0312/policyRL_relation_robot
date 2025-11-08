# CLAUDE.md

このファイルは、Claude Code (claude.ai/code) がこのリポジトリのコードを扱う際のガイダンスを提供します。

## 概要

このプロジェトは、PPO（Proximal Policy Optimization）強化学習を用いた会話ロボット方策の訓練システムです。ロボットは3人の人間による会話に介入し、反実仮想推論と遅延報酬を用いて対人関係の安定化を学習します。

**コアコンセプト**: ペルソナA、B、Cによる3者会話をシミュレートします。ロボット方策（LoRAチューニングされたLLM）は、構造バランス理論とLLMベースの関係性スコアリングで評価される対人関係を安定化させるために、いつどのように介入するかを学習します。

## プロジェクト構成

### 主要コンポーネント

1. **環境** (`app/env/convo_env.py`)
   - `ConversationEnv`: 3者会話のためのGymnasiumスタイルのRL環境
   - 会話状態、関係性スコアリング、報酬計算を管理
   - 反実仮想シミュレーションを実装（what-ifシナリオの評価）
   - 遅延報酬機構: ロボット介入のN発話後に関係性の安定度を評価

2. **PPO訓練** (`app/ppo_train.py`)
   - TRL（Transformers Reinforcement Learning）を使用したメイン訓練ループ
   - 大規模言語モデルの効率的なファインチューニングのためのLoRA/PEFT
   - カスタムデバイスマッピング戦略によるマルチGPUサポート
   - 方策崩壊を防ぐための適応的KLダイバージェンス係数

3. **関係性スコアリング** (`app/relation_scorer.py`)
   - `RelationScorer`: 会話参加者間の関係性品質を推定
   - 複数のバックエンドをサポート: Azure OpenAI、Ollama、ルールベース
   - LLMプロンプティングを使用してペアワイズ関係性スコア（-1〜+1）を抽出
   - 時間的一貫性のためのEMA（指数移動平均）平滑化

4. **人間シミュレーション** (`app/humans_ollama.py`)
   - 異なるペルソナを持つ人間の会話パートナーをシミュレート
   - 各ペルソナは設定可能な「地雷」キーワードを持ち、ネガティブな反応を引き起こす
   - Ollama-hostedのLLMを使用（異なるGPU上の複数インスタンスをサポート）

5. **設定** (`app/config.py`, `app/config.local.yaml`)
   - 一元化されたYAMLベースの設定システム
   - 環境パラメータ、報酬シェーピング、LLM設定、PPOハイパーパラメータを制御

### 報酬メカニズム

報酬システムは**反実仮想推論**を使用:

- **実世界（actual）**: ロボットが選択した行動（介入するorしない）後の実際の会話
- **反実仮想世界（counterfactual）**: ロボットが反対の行動を取った場合の仮想的な会話
- **報酬の計算式**: `(反実仮想の不安定度 - 実世界の不安定度) - 時間ペナルティ - 介入コスト + 安定ボーナス`

主要な報酬パラメータ（`config.local.yaml`で設定）:
- `evaluation_horizon`: 関係性の安定度を評価する前にシミュレートする人間の発話数
- `start_relation_check_after_utterances`: 最初のN発話では関係性チェックをスキップ（デフォルト: 3）
- `time_penalty`: ステップごとのコストで効率化を促進
- `intervention_cost`: ロボットが発話するコスト（過剰介入を抑制）
- `terminal_bonus`: 安定した関係性達成時のボーナス

### Docker構成

このリポジトリ付属の `docker-compose.yml` には主に訓練用コンテナが定義されています。現状の compose 定義では Trainer コンテナのみが含まれており、Ollama 系サービス (`ollama_a`/`ollama_b`/`ollama_c`/`ollama_d`) は含まれていません。

- `trainer`: PPOモデル訓練用のメイン訓練コンテナ（`container_name: rl-trainer`）

注意:
- Ollama インスタンスやその他の補助サービスは別途起動する必要があります（ローカルで個別起動するか、別の compose ファイル／ホストを用意してください）。
- Ollama のエンドポイントは `app/config.local.yaml` の `ollama.bases` や環境変数 `OLLAMA_BASES` / `OLLAMA_BASE` で指定します。

環境変数（利用に応じて設定）:
- `OLLAMA_BASES` / `OLLAMA_BASE`: 人間ペルソナ用の Ollama エンドポイント一覧または単一エンドポイント
- `OLLAMA_SCORER_BASE`: 関係性スコアリング専用の Ollama エンドポイント（優先度高）
- `CUDA_VISIBLE_DEVICES`: 訓練用の GPU 割り当て

## よく使うコマンド

### 環境変数のセットアップ（初回のみ）

機密情報（APIキー、エンドポイントなど）は `.env` ファイルで管理します。

```bash
# .env.example をコピー
cp .env.example .env

# .env を編集して実際の値を入力
nano .env  # または vim, code など
```

**必須の環境変数**:
- `AZURE_ENDPOINT`: Azure OpenAI のエンドポイントURL
- `AZURE_API_KEY`: Azure OpenAI の APIキー
- `WANDB_ENTITY`: Weights & Biases のユーザー名またはチーム名

**推奨の環境変数**:
- `AZURE_API_VERSION`: Azure OpenAI の APIバージョン（デフォルト: `2024-12-01-preview`）
- `AZURE_MODEL`: デフォルトモデル名（デフォルト: `gpt-5-chat`）
- `HUMAN_MODEL`, `RELATION_MODEL`, `ROBOT_MODEL`, `TOPIC_MODEL`, `INTERVENTION_MODEL`: 各用途別のモデル

詳細は `.env.example` を参照してください。

**注意**:
- `.env` ファイルは `.gitignore` に含まれており、Gitにコミットされません
- `config.local.yaml` は機密情報を含まないため、Gitにコミット可能です
- 環境変数が設定されていない場合、`config.local.yaml` のデフォルト値が使用されます

### Docker セットアップ

```bash
# コンテナのビルドと起動
docker-compose up -d

# trainerコンテナにアクセス
docker exec -it rl-trainer bash

# ログを表示
docker-compose logs -f trainer
```

### 訓練

```bash
# trainerコンテナ内（/workspace）
cd app

# PPO訓練を実行
python ppo_train.py

# ドライラン（訓練なしでCUDAとモデルロードを検証）
python ppo_train.py --dry-run
```

### テスト

```bash
# Azure ベースの人間シミュレーションで環境をテスト
python app/test_azure_human_convo.py

# 遅延報酬メカニズムをテスト
python test_delayed_reward.py

# 環境の変更をテスト
python test_env_changes.py
```

### GPT5介入判定シミュレーション

学習済みモデル（Qwen3）との比較のため、GPT5（Azure OpenAI）で介入判定した時の報酬や会話の様子を確認できます。

```bash
# 基本的な使用方法（5セッション、各セッション6ステップ）
python app/simulate_gpt5_intervention.py

# セッション数とステップ数をカスタマイズ
python app/simulate_gpt5_intervention.py --num-sessions 10 --max-steps 8

# 詳細な出力を抑制（統計のみ表示）
python app/simulate_gpt5_intervention.py --quiet
```

**機能**:
- GPT5を使った介入判定（いつ・どのように介入するか）
- セッションごとに新しい話題を自動生成
- 会話内容、介入判定、関係性スコア、報酬の詳細を出力
- 全セッションの統計（総報酬、平均報酬、介入回数など）を表示

**出力内容**:
- セッションごとの会話履歴（人間3人 + ロボット）
- ステップごとの介入判定内容（対象エッジ、戦略、対象話者）
- 関係性スコア（不安定トライアド数、安定状態）
- 報酬の内訳（時間ペナルティ、介入コスト、安定ボーナスなど）
- セッション統計（総報酬、介入回数、平均報酬など）

### 設定

設定は階層的に管理されています：

1. **`.env` ファイル（最優先）**: 機密情報（APIキー、エンドポイントなど）
2. **`app/config.local.yaml`**: 非機密の設定パラメータ（コミット可能）
3. **`app/config.py`**: デフォルト値とバリデーション

**設定の優先順位**:
- `.env` ファイルの環境変数が最優先
- `.env` に設定がない場合、`config.local.yaml` の値を使用
- どちらにもない場合、`config.py` のデフォルト値を使用

**`config.local.yaml` の主要なセクション**:
- `env.*`: 環境設定（max_steps、報酬パラメータ、ペルソナの地雷）
- `ppo.*`: PPOハイパーパラメータ（学習率、バッチサイズ、KL係数）
- `llm.*`: Azure OpenAI設定（機密情報は `.env` で設定）
- `ollama.*`: Ollamaエンドポイント設定とモデル選択
- `scorer.*`: 関係性スコアリングのバックエンドとパラメータ
- `wandb.*`: Weights & Biases可視化設定（エンティティは `.env` で設定）

**注意**:
- `config.local.yaml` は機密情報を含まないため、Gitにコミット可能です
- 機密情報は必ず `.env` ファイルで管理してください
- `.env` ファイルは `.gitignore` に含まれており、Gitにコミットされません

### 現在の設定値（2025-11-01時点）

#### 環境設定 (`env`)
```yaml
max_rounds: 6  # 1ラウンド = 3人間発話、6ラウンド = 18発話
max_auto_skip: 10  # 安定状態が続いた場合の自動スキップ回数
evaluation_horizon: 3  # ロボット介入後、N発話後の関係性を評価
start_relation_check_after_utterances: 3  # 最初のN発話は関係性チェックをスキップ
time_penalty: 0.02  # ステップごとのペナルティ（0.05 → 0.02に削減）
terminal_bonus: 2.0  # 安定達成時のボーナス（1.5 → 2.0に増加）
intervention_cost: 0.3  # ロボット介入コスト（0.45 → 0.3に削減）
```

#### 関係性スコアラー設定 (`scorer`)
```yaml
backend: "azure"   # 例: "azure" | "ollama" | "rule"
use_ema: false      # EMA(指数移動平均)は現在 config で無効化されています
decay_factor: 1.5
```

#### PPOハイパーパラメータ (`ppo`)
```yaml
# KL安定化
lr: 1.0e-5  # 学習率（2e-5 → 1e-5、KL爆発防止）
batch_size: 16  # バッチサイズ（8 → 16、統計的安定性向上）
grad_accum_steps: 2  # 勾配累積（実効バッチサイズ16）
kl_coef: 0.3  # KL係数（0.15 → 0.3、方策崩壊防止）
kl_estimator: "k3"  # KL推定器（k1 → k3、分散低減）
target_kl: 0.08  # 目標KL（0.05 → 0.08、探索の余地確保）
num_mini_batches: 4  # ミニバッチ数（1 → 4、更新の粒度向上）

# 探索促進
temperature: 2.0  # 温度（1.5 → 2.0、探索性向上）
top_p: 0.85  # nucleus sampling（0.90 → 0.85）
entropy_floor: 0.02  # エントロピー閾値（0.008 → 0.02）
entropy_patience: 10  # 早期停止の猶予（5 → 10）

# 価値関数
vf_coef: 0.05  # 価値関数係数（0.1 → 0.05、過学習防止）
gamma: 0.9  # 割引率（1.0 → 0.9、長期報酬の割引）

# 学習制御
filter_zero_rewards: true  # 安定状態の自動スキップを有効化
whiten_rewards: true  # 報酬の正規化
```

#### Weights & Biases設定 (`wandb`)
```yaml
enabled: true
project: "rl-convo-policy"
entity: "yoshinari0312-keio-jp"
table_log_frequency: 30  # 30介入ごとにwandb.Tableをログ
```

## 重要な実装の詳細

### 安定状態の自動スキップ (`filter_zero_rewards`)

**目的**: 報酬0のサンプル（安定状態で介入しない場合）を訓練データから除外し、学習効率を向上させる。

**実装** (`app/env/convo_env.py`):
- `reset()`: 初期会話生成後、安定状態であれば最大`max_auto_skip`回まで人間発話を生成し、不安定になるまで待つ。それでも安定なら話題を切り替えて再生成。
- `step()`: 介入判定前に安定状態をチェック。安定であれば同様に人間発話を自動生成。`max_auto_skip`回試しても不安定にならない場合はエピソード終了。

**設定**:
```yaml
ppo:
  filter_zero_rewards: true  # 有効化
env:
  max_auto_skip: 10  # 最大自動スキップ回数
```

**効果**:
- 訓練データの質向上（報酬シグナルが明確なサンプルのみ）
- 学習の高速化
- 方策崩壊のリスク低減

### 会話履歴のフィルタリング

システムは異なるLLM呼び出しに対して異なるフィルタリング戦略を使用:

- **人間/介入判断/ロボット発言LLM**: 直近N個の人間発話 + その範囲内のロボット発話をすべて含む
- **関係性スコアラー**: 直近N個の人間発話のみ、ロボット発話は完全に除外

これは`convo_env.py`の`_filter_logs_by_human_count()`で`exclude_robot`パラメータを使って処理されます。

### トピック生成

環境はAzure OpenAIを使用して会話トピックを動的に生成します（`topic_manager`セクションで設定）。トピックは`used_topics`で追跡され、重複を避けます。

### 関係性の安定性

構造バランス理論に基づく:
- **安定パターン**: すべて正（+++）または1つ正2つ負（+--/-+-/--+）
- **不安定**: その他すべてのパターン
- 不安定度スコア: 安定に到達するために必要な符号反転の重み付き合計

### マルチGPU戦略

PPO訓練は複数のデバイスマッピング戦略をサポート（`ppo.device_map_strategy`で設定）:

- `auto`: Hugging Faceの自動配置
- `balanced`: GPU間での均等分散
- `balanced_low_0`: GPU 0を優先、その後他を均等化
- `sequential`: レイヤーの順次割り当て

参照モデルは単一GPUに配置されます（`ppo.ref_device`で設定可能）。

### Weights & Biases可視化

訓練中の方策の振る舞いをリアルタイムで監視するため、wandb.Tableによる可視化を実装。

**記録内容**:
1. **戦略テーブル** (`strategy_table`):
   - 各介入での戦略選択（validate/bridge/reframe）
   - 対象エッジ、安定状態の変化、報酬

2. **出力テーブル** (`output_table`):
   - モデルの詳細な出力（発話内容、関係性スコア）
   - 最新100件を保持

3. **戦略分布メトリクス**:
   - `strategy/validate_ratio`, `strategy/bridge_ratio`, `strategy/reframe_ratio`
   - リアルタイムで方策崩壊を検出

**使い方**:
```bash
# wandbにログイン（初回のみ）
wandb login

# 訓練実行
python app/ppo_train.py

# ダッシュボードで確認
# https://wandb.ai/yoshinari0312-keio-jp/rl-convo-policy
```

**方策崩壊の検出**:
- `strategy/validate_ratio > 0.8` → 方策崩壊の兆候
- `strategy/reframe_ratio = 0` → reframe戦略が消失

### バッチサマリー集計

**BatchSummaryCollector** (`app/batch_summary.py`):
- バッチごとの統計を正確に集計
- 選択肢の分布とエントロピーを計算
- `batch_summary.jsonl`に記録（`conversation_summary.jsonl`と分離）

**記録内容**:
```json
{
  "batch_summary": {
    "batch_id": 1,
    "batch_size": 16,
    "total_reward": 2.5,
    "total_interventions": 12,
    "choice_entropy": {
      "strategy": 1.2,  // エントロピー（bits）
      ...
    },
    "choice_distributions": {
      "strategy": {
        "validate": 8,
        "bridge": 5,
        "reframe": 3
      }
    }
  }
}
```

## ファイル構造

```
app/
├── env/
│   └── convo_env.py               # メインRL環境
├── config.py                      # 設定ローダー
├── config.local.yaml              # ローカル設定（コミットしない）
├── ppo_train.py                   # PPO訓練スクリプト
├── batch_summary.py               # バッチサマリー集計モジュール
├── relation_scorer.py             # 関係性スコアリングモジュール
├── humans_ollama.py               # 人間ペルソナシミュレーション
├── azure_clients.py               # Azure OpenAIクライアントセットアップ
├── context_reward.py              # コンテキストベース報酬評価
├── network_metrics.py             # 構造バランス計算
├── run_robot_inference.py         # ロボット方策推論
├── test_azure_human_convo.py      # 環境のテストスクリプト
└── simulate_gpt5_intervention.py  # GPT5介入判定シミュレーションスクリプト

docker-compose.yml           # マルチコンテナセットアップ
Dockerfile                  # コンテナ定義
CHANGES_PPO_INTEGRATION.md  # PPO変更のドキュメント
```

### ログファイル構造

訓練実行ごとに`app/logs/ppo_run-YYYYMMDD-HHMMSS/`ディレクトリが作成され、以下のファイルが生成されます：

- **`conversation_summary.jsonl`**: 各ステップの詳細な会話ログ（介入判定、報酬、関係性など）
- **`batch_summary.jsonl`**: バッチごとの統計（戦略分布、エントロピー、報酬合計など）
- **`metrics.jsonl`**: 訓練メトリクス（損失、KL、エントロピーなど）
- **`run_events.jsonl`**: エピソード開始/終了などのイベントログ
- **`config.json`**: 実行時の全設定のスナップショット

## 重要な動作の注意点

1. **3発話の閾値**: 最初の3回の人間発話では関係性スコアリングがスキップされ、会話がコンテキストを確立できるようにします。この期間中は報酬は計算されません。

2. **ペルソナの地雷**: 各ペルソナ（A/B/C）は設定可能な地雷キーワードを持ち、言及されるとネガティブな反応を引き起こします。これらは`config.local.yaml`の`env.personas.{A|B|C}.triggers`で定義されます。

3. **終了ボーナスの条件**: ロボットは、安定性が`terminal_bonus_duration`発話の間維持され、かつロボットが直近`min_robot_intervention_lookback`発話の間介入していない場合にのみ終了ボーナスを受け取ります（無関係な安定化でクレジットを受け取ることを防ぐため）。

4. **モデルパッチ**: コードには、量子化されたUnslothバリアントとの互換性を確保するためのGPT-OSS MoE（Mixture of Experts）モデル用のカスタムパッチが含まれています。

## 開発時の注意事項

- **recheck_after_turnsは廃止**: 不安定な場合は常に関係性チェックと介入判定を実行します
- **会話履歴は用途別に異なるフィルタリング**: 関係性LLMはロボット発話を除外、他のLLMは含める
- **反実仮想シミュレーションと実世界の区別**: 報酬計算では、実際に展開された会話（actual）から関係性を計算し、もう一方の選択肢（counterfactual）はシミュレーションで評価します
- **filter_zero_rewardsの動作**: `ppo.filter_zero_rewards=true`の場合、安定状態では自動的に人間発話を生成し、不安定になるまで待ちます。`max_auto_skip`回試しても不安定にならない場合はエピソードを終了します。

## 方策崩壊の問題と対策

### 問題の兆候

訓練ログで以下のパターンが見られる場合、方策崩壊が発生している可能性があります：

1. **戦略の偏り**: `strategy/validate_ratio > 0.8`（validate戦略に異常に偏る）
2. **戦略の消失**: `strategy/reframe_ratio = 0`（reframe戦略が完全に消失）
3. **連続使用**: 同じ戦略が数十回連続で選ばれる
4. **エントロピー崩壊**: `train/policy/entropy_avg < 0.05`
5. **KLダイバージェンス暴走**: `train/objective/kl`が0〜12の範囲で激しく変動

### 原因

- **バッチサイズ不足**: 統計的に不安定なKL推定
- **温度が低すぎ**: 探索不足
- **報酬シグナルが弱い**: 一部の戦略のみが短期的に有利に見える

### 対策（実装済み）

1. **KL安定化**:
   - `batch_size: 16`（8 → 16）
   - `kl_estimator: "k3"`（分散の低い推定器）
   - `num_mini_batches: 4`（更新の粒度向上）

2. **探索促進**:
   - `temperature: 2.0`（1.5 → 2.0）
   - `entropy_floor: 0.02`（0.008 → 0.02）
   - `entropy_patience: 10`（5 → 10）

3. **報酬シグナル強化**:
   - `intervention_cost: 0.3`（0.45 → 0.3、介入のハードル低減）
   - `terminal_bonus: 2.0`（1.5 → 2.0、成功報酬増加）

### 監視方法

**wandbダッシュボード**で以下を確認：
- 戦略分布グラフ（各戦略が20-40%の範囲に分布しているか）
- エントロピーグラフ（> 0.2を維持しているか）
- KLグラフ（< 1.0で安定しているか）

**ログファイル**で確認：
```bash
# 戦略分布を確認
grep -o '"strategy": "[^"]*"' app/logs/<run>/conversation_summary.jsonl | \
  cut -d'"' -f4 | sort | uniq -c
```
