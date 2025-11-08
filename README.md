# policyRL_relation_robot

PPO強化学習を用いた会話ロボット方策の訓練システム

## セットアップ

### 1. 環境変数の設定

機密情報（APIキーなど）は `.env` ファイルで管理します。

```bash
# .env.example をコピー
cp .env.example .env

# .env を編集して、実際の値を入力
nano .env  # または vim, code など
```

必須の環境変数：
- `AZURE_ENDPOINT`: Azure OpenAI のエンドポイントURL
- `AZURE_API_KEY`: Azure OpenAI の APIキー
- `WANDB_ENTITY`: Weights & Biases のユーザー名またはチーム名

その他の環境変数については `.env.example` を参照してください。

### 2. 依存パッケージのインストール

```bash
pip install python-dotenv pyyaml
```

### 3. 使い方

詳細な使用方法については [CLAUDE.md](./CLAUDE.md) を参照してください。

## ディレクトリ構成

- `app/`: メインのアプリケーションコード
  - `env/`: 強化学習環境
  - `config.py`: 設定ローダー
  - `config.local.yaml`: ローカル設定（機密情報なし、コミット可能）
  - `ppo_train.py`: PPO訓練スクリプト
- `.env.example`: 環境変数のテンプレート
- `.env`: 機密情報を含む環境変数（gitignore済み、手動作成）
