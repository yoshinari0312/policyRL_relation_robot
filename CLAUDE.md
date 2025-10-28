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

システムは複数のDockerコンテナを使用（`docker-compose.yml`参照）:

- `trainer`: PPOモデル訓練用のメイン訓練コンテナ（GPU 0-3使用）
- `ollama_a`, `ollama_b`, `ollama_c`: 人間ペルソナをシミュレートする3つのOllamaインスタンス（GPU 4-5使用）
- `ollama_d`: 関係性スコアリング用のOllamaインスタンス（GPU 5使用）

環境変数:
- `OLLAMA_BASES`: 人間ペルソナ用のOllamaエンドポイントのカンマ区切りリスト
- `CUDA_VISIBLE_DEVICES`: 訓練用のGPU割り当て
- `OLLAMA_SCORER_BASE`: 関係性スコアリング専用のエンドポイント

## よく使うコマンド

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

すべての設定は`app/config.local.yaml`にあります。主要なセクション:

- `env.*`: 環境設定（max_steps、報酬パラメータ、ペルソナの地雷）
- `ppo.*`: PPOハイパーパラメータ（学習率、バッチサイズ、KL係数）
- `llm.*`: Azure OpenAI設定（エンドポイント、APIキー、モデルデプロイメント）
- `ollama.*`: Ollamaエンドポイント設定とモデル選択
- `scorer.*`: 関係性スコアリングのバックエンドとパラメータ

**注意**: `config.local.yaml`にはAPIキーが含まれています。このファイルは絶対にコミットしないでください。

## 重要な実装の詳細

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

## ファイル構造

```
app/
├── env/
│   └── convo_env.py               # メインRL環境
├── config.py                      # 設定ローダー
├── config.local.yaml              # ローカル設定（コミットしない）
├── ppo_train.py                   # PPO訓練スクリプト
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

## 重要な動作の注意点

1. **3発話の閾値**: 最初の3回の人間発話では関係性スコアリングがスキップされ、会話がコンテキストを確立できるようにします。この期間中は報酬は計算されません。

2. **ペルソナの地雷**: 各ペルソナ（A/B/C）は設定可能な地雷キーワードを持ち、言及されるとネガティブな反応を引き起こします。これらは`config.local.yaml`の`env.personas.{A|B|C}.triggers`で定義されます。

3. **終了ボーナスの条件**: ロボットは、安定性が`terminal_bonus_duration`発話の間維持され、かつロボットが直近`min_robot_intervention_lookback`発話の間介入していない場合にのみ終了ボーナスを受け取ります（無関係な安定化でクレジットを受け取ることを防ぐため）。

4. **モデルパッチ**: コードには、量子化されたUnslothバリアントとの互換性を確保するためのGPT-OSS MoE（Mixture of Experts）モデル用のカスタムパッチが含まれています。

## 開発時の注意事項

- **recheck_after_turnsは廃止**: 不安定な場合は常に関係性チェックと介入判定を実行します
- **会話履歴は用途別に異なるフィルタリング**: 関係性LLMはロボット発話を除外、他のLLMは含める
- **反実仮想シミュレーションと実世界の区別**: 報酬計算では、実際に展開された会話（actual）から関係性を計算し、もう一方の選択肢（counterfactual）はシミュレーションで評価します
