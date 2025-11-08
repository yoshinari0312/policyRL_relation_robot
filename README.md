# RL-based Conversation Policy for Social Robots

**PPO強化学習を用いた会話ロボット方策の訓練システム**

3者間会話における対人関係の安定化を学習する社会的ロボットの研究プロジェクト

---

## 📋 目次

- [研究概要](#研究概要)
- [コアコンセプト](#コアコンセプト)
- [システムアーキテクチャ](#システムアーキテクチャ)
- [主要コンポーネント](#主要コンポーネント)
- [報酬メカニズム](#報酬メカニズム)
- [セットアップ](#セットアップ)
- [使い方](#使い方)
- [ファイル構成](#ファイル構成)
- [設定ガイド](#設定ガイド)
- [可視化とモニタリング](#可視化とモニタリング)
- [トラブルシューティング](#トラブルシューティング)

---

## 研究概要

### 目的

このプロジェクトは、**3人の人間による会話に介入し、対人関係の安定化を学習する社会的ロボット**を強化学習で訓練するシステムです。ロボットは以下を学習します：

- **どのように介入すべきか** (Strategy): どのような戦略で介入するか（介入しない選択も含む）
  - **validate**: 対象者の感情や意見を承認し、心理的安全性を構築する
  - **bridge**: 対立する相手との共通点や協力の軸を見つけ、関係を再接続する
  - **plan**: 対象者に次の行動や方針を示し、前向きな関係改善を促す
  - **no_intervention**: 対立が軽度で、介入が逆効果になりそうなときや自然回復が見込める時に選ぶ

### 研究の意義

- **社会的AI**: 人間関係の微妙なダイナミクスを理解し、適切に介入するAIシステムの実現
- **強化学習の応用**: 遅延報酬と反実仮想推論を用いた複雑な社会的タスクへのPPOの適用
- **構造バランス理論**: 社会心理学の理論を計算モデルに統合

### 技術的特徴

1. **反実仮想推論 (Counterfactual Reasoning)**:
   - ロボットが介入した場合としなかった場合の両方をシミュレート
   - 介入の真の効果を評価

2. **遅延報酬 (Delayed Reward)**:
   - 介入後N発話後の関係性を評価
   - 短期的な反応ではなく、持続的な効果を重視

3. **構造バランス理論 (Structural Balance Theory)**:
   - 3者間関係の安定性を数学的に評価
   - 社会心理学の理論に基づく報酬設計

4. **マルチモーダルLLM**:
   - 人間シミュレーション: Ollama (Qwen3/GPT-OSS)
   - 関係性評価: Azure OpenAI (GPT-4.1/GPT-5)
   - ロボット方策: LoRAチューニングされたLlama-3.1-Swallow

---

## コアコンセプト

### 3者会話シミュレーション

```
┌─────────────────────────────────────────────────┐
│         3者会話 (Persona A, B, C)               │
│  - 各ペルソナは「地雷」キーワードを持つ        │
│  - 地雷が踏まれるとネガティブな反応           │
│  - 会話は動的に生成される                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│      関係性スコアリング (Relation Scorer)       │
│  - LLMで各ペア間の親密度を評価 (-1.0 ~ +1.0)  │
│  - 構造バランス理論で安定性を計算             │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│    対象エッジ・話者の自動選択 (Program)        │
│  - 負の中で全体値が小さい関係性エッジを選択                 │
│  - そのエッジの2人からランダムに話者選択      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│       ロボットの介入戦略選択 (Policy/LLM)      │
│  - 戦略のみをLLMで選択（1桁の数字出力）       │
│    1: validate, 2: bridge,                     │
│    3: plan, 4: no_intervention                 │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│  反実仮想シミュレーション (Counterfactual)      │
│  ※介入する場合（no_intervention以外）のみ     │
│  - 実世界: ロボットが介入した結果             │
│  - 反実世界: 介入しなかった場合の結果         │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│            報酬計算 (Reward)                    │
│  reward = (反実の不安定度 - 実世界の不安定度)  │
│           - 時間ペナルティ                      │
│           - 介入コスト                          │
│           + 安定ボーナス                        │
└─────────────────────────────────────────────────┘
```

### 構造バランス理論

3者間関係の安定パターン：

- **安定パターン**:
  - `(+, +, +)`: 全員が仲良し
  - `(+, -, -)`: AとBが仲良く、両者ともCと対立
  - `(-, +, -)`: BとCが仲良く、両者ともAと対立
  - `(-, -, +)`: CとAが仲良く、両者ともBと対立

- **不安定パターン**: その他すべて（例: `(+, +, -), (-, -, -)`）

**不安定度スコア**: 安定状態に到達するために必要な関係性の符号反転の最小コスト

---

## システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Loop (PPO)                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Policy Model (LoRA-tuned Llama-3.1-Swallow)       │    │
│  │  - 介入戦略のみを選択（1桁の数字: 1-4）           │    │
│  └────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Conversation Environment (ConversationEnv)        │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ Human Simulation (Ollama/Azure)              │ │    │
│  │  │ - Persona A, B, C の発話生成                │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ Relation Scorer (Azure GPT-4.1/GPT-5)        │ │    │
│  │  │ - ペアワイズ関係性スコアリング               │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ Counterfactual Simulator                     │ │    │
│  │  │ - 実世界・反実世界の並列シミュレーション     │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  │  ┌──────────────────────────────────────────────┐ │    │
│  │  │ Reward Calculator                            │ │    │
│  │  │ - 構造バランス理論ベースの報酬計算           │ │    │
│  │  └──────────────────────────────────────────────┘ │    │
│  └────────────────────────────────────────────────────┘    │
│                           ↓                                  │
│  ┌────────────────────────────────────────────────────┐    │
│  │  PPO Trainer (TRL)                                 │    │
│  │  - Policy/Value 損失計算                           │    │
│  │  - KL発散制約                                      │    │
│  │  - エントロピー正則化                              │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│              Monitoring & Logging (Weights & Biases)         │
│  - 戦略分布の可視化                                          │
│  - 報酬の時系列変化                                          │
│  - KL発散とエントロピーの監視                                │
│  - 会話ログのテーブル記録                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 主要コンポーネント

### 1. 環境 (`app/env/convo_env.py`)

**`ConversationEnv`**: Gymnasiumスタイルの強化学習環境

- **状態管理**: 会話履歴、関係性スコア、安定状態を管理
- **反実仮想シミュレーション**: 実世界と反実世界を並列実行
- **遅延報酬**: 介入後N発話後の関係性を評価
- **自動スキップ**: 安定状態が続く場合、不安定になるまで人間発話を自動生成

**主要メソッド**:
- `reset()`: エピソード初期化、初期会話生成
- `step(action)`: ロボットの行動を受け取り、報酬を返す
- `_bootstrap_humans(n)`: N発話の人間会話を生成
- `_simulate_counterfactual()`: 反実世界のシミュレーション

### 2. PPO訓練 (`app/ppo_train.py`)

**メイン訓練ループ**: TRL (Transformers Reinforcement Learning) を使用

- **介入戦略の学習**: LLMは戦略のみを学習（1桁の数字出力）
  - 1: validate, 2: bridge, 3: plan, 4: no_intervention
  - 対象エッジ・話者はプログラムが自動選択
- **LoRA/PEFT**: 効率的なファインチューニング (rank=16)
- **マルチGPU対応**: 最大6 GPUで並列訓練
- **KL発散制約**: 方策崩壊を防ぐ適応的KL係数
- **エントロピー正則化**: 探索を促進（自動調整機能付き）

**主要機能**:
- デバイスマップ戦略 (balanced_low_0): GPU間の負荷分散
- 決定的評価モード: 訓練中の方策性能を定期評価
- バッチサマリー: 戦略分布とエントロピーの集計

### 3. 関係性スコアリング (`app/relation_scorer.py`)

**`RelationScorer`**: 会話参加者間の関係性を推定

- **複数バックエンド**: Azure OpenAI, Ollama, ルールベース
- **LLMプロンプティング**: ペアワイズ関係性スコア (-1.0 ~ +1.0) を抽出
- **EMA平滑化** (オプション): 時間的一貫性の確保

**評価プロセス**:
1. 会話履歴をLLMに入力
2. 各ペア (A-B, B-C, C-A) の親密度をスコアリング
3. 構造バランス理論で安定性を計算

### 4. 人間シミュレーション (`app/humans_ollama.py`)

**異なるペルソナを持つ人間パートナーをシミュレート**

- **ペルソナA, B, C**: 各25個の「地雷」キーワードを持つ
  - 3人共通: 15個（お金、外見、健康、仕事、恋愛など）
  - AB共通: 5個（勉強、研究、読書など）
  - BC共通: 5個（ゲーム、アニメ、音楽など）
  - CA共通: 5個（旅行、温泉、食べ物など）

- **地雷メカニズム**: 地雷が踏まれるとネガティブな反応を生成
- **LLMバックエンド**: Ollama (複数インスタンス) または Azure OpenAI

### 5. 設定管理 (`app/config.py`, `app/config.local.yaml`)

**階層的設定システム**:

1. `.env` ファイル（最優先）: 機密情報
2. `config.local.yaml`: 非機密の設定パラメータ
3. `config.py`: デフォルト値とバリデーション

**主要設定セクション**:
- `env.*`: 環境パラメータ（max_steps, 報酬パラメータなど）
- `ppo.*`: PPOハイパーパラメータ（学習率、バッチサイズなど）
- `llm.*`: Azure OpenAI設定
- `ollama.*`: Ollamaエンドポイント設定
- `scorer.*`: 関係性スコアリング設定
- `wandb.*`: Weights & Biases可視化設定

---

## 報酬メカニズム

### 反実仮想推論ベースの報酬

```python
# 実世界: ロボットが選んだ行動（介入 or 非介入）
actual_world = simulate(robot_action)
actual_instability = calculate_instability(actual_world)

# 反実世界: 逆の行動を取った場合
counterfactual_world = simulate(opposite_action)
counterfactual_instability = calculate_instability(counterfactual_world)

# 報酬 = 反実世界との差分
reward = (counterfactual_instability - actual_instability)
         - time_penalty
         - intervention_cost
         + stability_bonus
```

### 報酬の構成要素

1. **不安定度の差分**: `(反実の不安定度 - 実世界の不安定度)`
   - 正: ロボットの介入が関係性を改善
   - 負: ロボットの介入が関係性を悪化

2. **時間ペナルティ**: `time_penalty = 0.1` (デフォルト)
   - 効率的なエピソード完了を促進

3. **介入コスト**: `intervention_cost = 0.35` (デフォルト)
   - 過剰介入を抑制
   - 介入が必要な場面でのみ発言

4. **安定ボーナス**: `terminal_bonus = 2.0` (デフォルト)
   - 安定状態の達成時に付与
   - 条件:
     - ロボット介入後、安定が`terminal_bonus_duration`発話継続

### 遅延報酬

介入の効果は即座には現れないため、**介入後N発話後**の関係性を評価：

- `evaluation_horizon = 3` (デフォルト)
- ロボットが介入した場合、3発話後の会話状態を評価
- 短期的な反応ではなく、持続的な効果を重視

---

## セットアップ

### 前提条件

- Docker & Docker Compose
- NVIDIA GPU (CUDA 12.1対応)
- Azure OpenAI アカウント（または Ollama のみでも可）
- Weights & Biases アカウント（オプション、可視化用）

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd rl-convo-policy
```

### 2. 環境変数の設定

機密情報は `.env` ファイルで管理します。

```bash
# .env.example をコピー
cp .env.example .env

# .env を編集して実際の値を入力
nano .env  # または vim, code など
```

**必須の環境変数**:
```bash
# Azure OpenAI
AZURE_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_API_KEY=your-api-key
AZURE_API_VERSION=2024-12-01-preview
AZURE_MODEL=gpt-5-chat

# Embedding
AZURE_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_EMBEDDING_API_VERSION=2024-12-01-preview

# Weights & Biases
WANDB_ENTITY=your-username-or-team
WANDB_PROJECT=rl-convo-policy
```

**推奨の環境変数**（各用途別のモデル）:
```bash
HUMAN_MODEL=gpt-4.1          # 人間ペルソナ用
RELATION_MODEL=gpt-4.1       # 関係性スコアリング用
ROBOT_MODEL=gpt-5-chat       # ロボット発話生成用
TOPIC_MODEL=gpt-5-chat       # 話題生成用
INTERVENTION_MODEL=gpt-5-chat # 介入判定用
```

詳細は `.env.example` を参照してください。

### 3. Dockerコンテナの起動

```bash
# コンテナのビルドと起動
docker compose up -d

# ログを確認
docker compose logs -f trainer
```

### 4. コンテナにアクセス

```bash
docker exec -it rl-trainer bash
```

---

## 使い方

### 訓練の実行

```bash
# trainerコンテナ内 (/workspace)
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

### GPT-5介入判定シミュレーション

学習済みモデルとの比較のため、GPT-5で介入判定した場合の報酬や会話を確認：

```bash
# 基本的な使用方法（5セッション、各6ステップ）
python app/simulate_gpt5_intervention.py

# カスタマイズ
python app/simulate_gpt5_intervention.py --num-sessions 10 --max-steps 8

# 詳細出力を抑制（統計のみ）
python app/simulate_gpt5_intervention.py --quiet
```

**出力内容**:
- セッションごとの会話履歴（人間3人 + ロボット）
- ステップごとの介入判定内容（対象エッジ、戦略、対象話者）
- 関係性スコア（不安定トライアド数、安定状態）
- 報酬の内訳（時間ペナルティ、介入コスト、安定ボーナスなど）
- 全セッションの統計（総報酬、介入回数、平均報酬など）

---

## ファイル構成

```
rl-convo-policy/
├── app/                              # メインアプリケーション
│   ├── env/
│   │   └── convo_env.py              # 会話環境（Gymnasium準拠）
│   ├── config.py                     # 設定ローダー
│   ├── config.local.yaml             # ローカル設定（非機密、コミット可）
│   ├── ppo_train.py                  # PPO訓練メインスクリプト
│   ├── batch_summary.py              # バッチサマリー集計
│   ├── relation_scorer.py            # 関係性スコアリングモジュール
│   ├── humans_ollama.py              # 人間ペルソナシミュレーション
│   ├── azure_clients.py              # Azure OpenAIクライアント
│   ├── context_reward.py             # コンテキストベース報酬評価
│   ├── network_metrics.py            # 構造バランス計算
│   ├── run_robot_inference.py        # ロボット方策推論
│   ├── test_azure_human_convo.py     # 環境テストスクリプト
│   ├── simulate_gpt5_intervention.py # GPT-5介入シミュレーション
│   └── eval_deterministic.py         # 決定的評価モジュール
│
├── models/                           # 訓練済みモデル（.gitignore）
│   └── ppo_robot/                    # PPOモデル保存先
│
├── logs/                             # ログファイル（.gitignore）
│   └── ppo_run-YYYYMMDD-HHMMSS/      # 実行ごとのログディレクトリ
│       ├── conversation_summary.jsonl # 会話ログ
│       ├── batch_summary.jsonl       # バッチ統計
│       ├── metrics.jsonl             # 訓練メトリクス
│       ├── run_events.jsonl          # イベントログ
│       └── config.json               # 実行時設定スナップショット
│
├── docker-compose.yml                # Dockerサービス定義
├── Dockerfile                        # コンテナイメージ定義
├── .env.example                      # 環境変数テンプレート
├── .env                              # 機密情報（.gitignore、手動作成）
├── .gitignore                        # Git除外設定
├── README.md                         # このファイル
├── CLAUDE.md                         # 開発者向け詳細ドキュメント
└── CHANGES_PPO_INTEGRATION.md        # PPO統合の変更履歴
```

### ログファイル構造

各訓練実行ごとに `app/logs/ppo_run-YYYYMMDD-HHMMSS/` ディレクトリが作成されます：

- **`conversation_summary.jsonl`**: 各ステップの詳細（介入判定、報酬、関係性）
- **`batch_summary.jsonl`**: バッチごとの統計（戦略分布、エントロピー、報酬合計）
- **`metrics.jsonl`**: 訓練メトリクス（損失、KL、エントロピー）
- **`run_events.jsonl`**: エピソード開始/終了イベント
- **`config.json`**: 実行時の全設定スナップショット

---

## 設定ガイド

### 現在の主要設定値

#### 環境設定 (`config.local.yaml` - `env`)

```yaml
max_steps: 8                           # 1エピソードの最大ステップ数
max_rounds: 6                          # 最大ラウンド数（未使用）
evaluation_horizon: 3                  # 介入後N発話後の関係性を評価
start_relation_check_after_utterances: 3  # 最初N発話は関係性チェックスキップ
max_auto_skip: 10                      # 安定状態の自動スキップ回数

# 報酬パラメータ
time_penalty: 0.1                      # ステップごとのペナルティ
terminal_bonus: 2.0                    # 安定達成時のボーナス
intervention_cost: 0.35                # ロボット介入コスト
output_error_penalty: 1.0              # LLM出力失敗時のペナルティ
min_robot_intervention_lookback: 6     # 安定ボーナス判定範囲
terminal_bonus_duration: 2             # 安定継続必要発話数
```

#### PPOハイパーパラメータ (`config.local.yaml` - `ppo`)

```yaml
# 学習率
lr: 5.0e-5                             # 学習率
lr_scheduler_type: "cosine"            # スケジューラータイプ
warmup_ratio: 0.01                     # ウォームアップ期間

# バッチサイズ
per_device_train_batch_size: 8         # デバイスあたりバッチサイズ
grad_accum_steps: 4                    # 勾配累積ステップ
ppo_epochs: 3                          # PPOエポック数
num_mini_batches: 2                    # ミニバッチ数

# サンプリング
temperature: 0.7                       # 温度（Qwen3推奨値）
top_p: 0.9                             # nucleus sampling
max_new_tokens: 2                      # 生成トークン数

# エントロピー正則化
entropy_auto_adjust: true              # エントロピー係数自動調整
entropy_target: 0.7                    # 目標エントロピー
entropy_coef_initial: 0.05             # 初期エントロピー係数
entropy_floor: 0.03                    # エントロピー閾値
entropy_patience: 20                   # 早期停止猶予

# KL発散制約
kl_coef: 0.05                          # 初期KL係数
target_kl: 0.05                        # 目標KL
kl_estimator: "k3"                     # KL推定器（分散低減）
min_kl_coef: 0.01                      # KL係数下限
max_kl_coef: 0.3                       # KL係数上限

# クリッピング
cliprange: 0.3                         # 方策クリッピング範囲
cliprange_value: 0.2                   # 価値関数クリッピング
max_grad_norm: 0.5                     # 勾配クリッピング

# 価値関数
vf_coef: 0.1                           # 価値関数係数
gamma: 0.9                             # 割引率
lam: 0.95                              # GAE lambda

# その他
filter_zero_rewards: true              # 報酬0サンプルを除外
whiten_rewards: true                   # 報酬正規化
total_updates: 50                      # 総更新回数
```

#### デバイス設定

```yaml
cuda_visible_devices: "0,1,2,3,4,5"    # 使用GPU
device_map_strategy: "balanced_low_0"  # デバイスマップ戦略
ref_device: 5                          # 参照モデルのGPU
max_memory_per_device_gib: 44          # GPU最大メモリ (GiB)
```

### 設定の優先順位

1. **`.env` ファイル**（最優先）: 機密情報
2. **`config.local.yaml`**: 非機密の設定
3. **`config.py`**: デフォルト値

### 方策崩壊の防止

**問題の兆候**:
- `strategy/validate_ratio > 0.8` (validate戦略に偏る)
- `strategy/reframe_ratio = 0` (reframe戦略が消失)
- `train/policy/entropy_avg < 0.05` (エントロピー崩壊)
- KL発散が激しく変動 (0〜12の範囲)

**対策** (既に実装済み):
1. **KL安定化**: batch_size=64, kl_estimator="k3", num_mini_batches=2
2. **探索促進**: temperature=0.7, entropy_target=0.7, entropy_auto_adjust=true
3. **報酬シグナル強化**: intervention_cost=0.35, terminal_bonus=2.0

---

## 可視化とモニタリング

### Weights & Biases統合

訓練中の方策の振る舞いをリアルタイムで監視：

```bash
# wandbにログイン（初回のみ）
wandb login

# 訓練実行（自動的にログ記録）
python app/ppo_train.py

# ダッシュボードで確認
# https://wandb.ai/<your-entity>/rl-convo-policy
```

### 記録内容

1. **戦略テーブル** (`strategy_table`):
   - 各介入での戦略選択（validate/bridge/plan）
   - 対象エッジ、安定状態の変化、報酬

2. **出力テーブル** (`output_table`):
   - モデルの詳細な出力（発話内容、関係性スコア）
   - 最新100件を保持

3. **戦略分布メトリクス**:
   - `strategy/validate_ratio`, `strategy/bridge_ratio`, `strategy/reframe_ratio`
   - リアルタイムで方策崩壊を検出

4. **訓練メトリクス**:
   - `train/policy/loss_avg`: 方策損失
   - `train/value/loss_avg`: 価値関数損失
   - `train/objective/kl`: KL発散
   - `train/policy/entropy_avg`: エントロピー
   - `train/policy/clipfrac_avg`: クリッピング割合

### ローカルログ分析

```bash
# 戦略分布を確認
grep -o '"strategy": "[^"]*"' app/logs/ppo_run-*/conversation_summary.jsonl | \
  cut -d'"' -f4 | sort | uniq -c

# 報酬の統計
jq '.reward' app/logs/ppo_run-*/conversation_summary.jsonl | \
  awk '{sum+=$1; n++} END {print "平均報酬:", sum/n}'
```

---

## トラブルシューティング

### よくある問題

#### 1. Azure OpenAI接続エラー

**エラー**: `RuntimeError: Azure OpenAI クライアントの取得に失敗しました。`

**解決策**:
```bash
# .envファイルを確認
cat .env | grep AZURE

# 必要な変数が設定されているか確認
echo $AZURE_ENDPOINT
echo $AZURE_API_KEY
```

#### 2. CUDA Out of Memory

**エラー**: `RuntimeError: CUDA out of memory`

**解決策**:
```yaml
# config.local.yaml を編集
ppo:
  per_device_train_batch_size: 4  # 8 → 4に削減
  grad_accum_steps: 8              # 4 → 8に増加（実効バッチサイズ維持）
  max_memory_per_device_gib: 40    # 44 → 40に削減
```

#### 3. 方策崩壊

**症状**: 特定の戦略に偏る、エントロピーが低下

**解決策**:
```yaml
# config.local.yaml を編集
ppo:
  entropy_coef_initial: 0.1        # 0.05 → 0.1に増加
  temperature: 1.0                 # 0.7 → 1.0に増加
  kl_coef: 0.1                     # 0.05 → 0.1に増加
```

#### 4. Docker環境変数が読み込まれない

**症状**: コンテナ内で環境変数が空

**解決策**:
```bash
# コンテナを再起動
docker compose down
docker compose up -d

# 環境変数を確認
docker exec rl-trainer bash -c "echo \$AZURE_ENDPOINT"
```

#### 5. wandbログインエラー

**エラー**: `wandb: ERROR authentication failed`

**解決策**:
```bash
# コンテナ内でログイン
docker exec -it rl-trainer bash
wandb login

# または .envファイルにAPIキーを設定
echo "WANDB_API_KEY=your-api-key" >> .env
```

### デバッグモード

```yaml
# config.local.yaml
env:
  debug: true                        # デバッグログを有効化
ppo:
  prompt_feed_debug: true            # プロンプト全文をダンプ
```

### ログの確認

```bash
# 訓練ログをリアルタイムで確認
tail -f app/logs/ppo_train.log

# 特定のエラーを検索
grep -i "error" app/logs/ppo_run-*/conversation_summary.jsonl

# バッチサマリーを確認
jq '.batch_summary' app/logs/ppo_run-*/batch_summary.jsonl | less
```

---

## 参考資料

### 関連論文

- **PPO**: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
- **構造バランス理論**: Heider (1946) "Attitudes and Cognitive Organization"
- **反実仮想推論**: Pearl (2009) "Causality: Models, Reasoning, and Inference"

### 使用ライブラリ

- **TRL** (Transformers Reinforcement Learning): PPO実装
- **PEFT** (Parameter-Efficient Fine-Tuning): LoRAサポート
- **Transformers**: Hugging Faceモデルライブラリ
- **Gymnasium**: RL環境インターフェース
- **Weights & Biases**: 実験管理と可視化

### 開発者向けドキュメント

詳細な実装ガイドは [CLAUDE.md](./CLAUDE.md) を参照してください。

---

## ライセンス

研究用途のみ。商用利用については別途相談。

## 貢献

プルリクエストやイシューは歓迎します。大きな変更の場合は、事前にイシューで議論してください。

## 連絡先

質問や提案は GitHub Issues でお願いします。
