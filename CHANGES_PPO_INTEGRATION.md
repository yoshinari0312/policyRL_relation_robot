# PPO強化学習対応の変更まとめ

test_azure_human_convo.pyで実装したルールをPPO強化学習でも使えるように、以下のファイルを変更しました。

## 1. config.local.yaml の変更

### 追加した設定:
- `env.start_relation_check_after_utterances`: 3発話後から関係性推定を開始
- `env.evaluation_horizon`: ロボットの何発話後の関係性を報酬に使うか
- `env.time_penalty`: 毎ステップのペナルティ（デフォルト: 0.01）
- `env.terminal_bonus`: 安定達成時のボーナス（デフォルト: 0.25）
- `env.intervention_cost`: ロボット介入時のコスト（デフォルト: 0.02）
- `personas.A/B/C.triggers`: 各人間の地雷キーワードリスト（複数指定可能）
- `topic_manager.enable`: トピック提案機能の有効化フラグ
- `topic_manager.generation_prompt`: トピック生成用のプロンプト

### 地雷設定の例:
```yaml
personas:
  A:
    triggers:
      - "お金"
      - "給料"
      - "費用"
  B:
    triggers:
      - "時間"
      - "締め切り"
      - "期限"
  C:
    triggers:
      - "旅行"
      - "お出かけ"
      - "外出"
```

## 2. env/convo_env.py の変更

### 追加・変更した機能:

1. **トピック管理**:
   - `self.used_topics`: 既に使用したトピックのリスト
   - `self.current_topic`: 現在のトピック
   - `_get_topic_suggestion()`: LLMでトピックを生成

2. **会話履歴フィルタリング**:
   - `_filter_human_logs()`: 関係性LLM用にロボット発言を除外
   - 過去6人間発言まで、ロボット発言の数だけ追加

3. **3発話後からの関係性チェック**:
   - `start_relation_check_after_utterances`: 設定から読み込み
   - `step()`メソッドで発話数をカウントし、3発話未満の場合は関係性推定をスキップ
   - `_bootstrap_humans()`でも同様のチェック

4. **recheck_after_turnsの削除**:
   - `step()`メソッドから`recheck_after_turns`の処理を削除
   - 常に`evaluation_horizon`を使用
   - 不安定な場合は毎ターン関係性推定と介入判定を実行

5. **reset()の変更**:
   - トピック生成を追加
   - `used_topics`にトピックを追加

6. **報酬計算の変更**:
   - 報酬パラメータ（time_penalty、terminal_bonus、intervention_cost）をYAMLから読み込み
   - **3発話未満または安定状態の場合は報酬を計算しない**（time_penaltyなども適用しない）
   - 不安定時のみ報酬計算を実行
   - 安定達成時のみterminal_bonusを適用

7. **不安定度の修正**:
   - 3発話未満の場合の不安定度を`float('inf')`から`1.0`に修正（最大値は1.0）

## 3. humans_ollama.py の変更

### 追加・変更した機能:

1. **地雷設定の動的読み込み**:
   - `build_human_prompt()`でconfigからpersonas設定を読み込み
   - 各話者の`triggers`リストを取得
   - ハードコードされた地雷設定を削除

2. **トピックの反映**:
   - `build_human_prompt()`に`topic`パラメータを追加
   - プロンプトにトピックを含める
   - `human_reply()`に`topic`パラメータを追加

## 4. ppo_train.py の変更

- 特に変更なし（ConversationEnvの変更が自動的に反映される）

## 動作の流れ

### 初期化 (reset):
1. トピックをLLMで生成
2. `used_topics`に追加
3. 人間発話を生成（トピックと地雷設定を反映）
4. 不安定状態になるまで発話を追加

### ステップ実行 (step):
1. 3発話未満の場合は関係性推定をスキップ
2. 3発話以降かつ不安定な場合のみ:
   - 人間発話のみをフィルタリングして関係性推定
   - 介入判定を実行
   - 介入する場合はロボット発話生成
3. ロボット発話がある場合、その分履歴を増やす
4. 報酬計算と次の観測を返す

### 会話履歴の扱い:
全てのLLMで人間発話数を`max_history`で固定し、その間のロボット発話の扱いを用途別に調整:

**人間LLM・介入判断LLM・ロボット発言LLM:**
- **人間発話数を`max_history`で固定**
- **その間にロボット発話がn回あれば、合計`max_history + n`個の会話履歴をLLMに渡す**
- `_filter_logs_by_human_count(logs, max_history, exclude_robot=False)`を使用

**関係性LLM（RelationScorer）:**
- **人間発話数を`max_history`で固定**
- **ロボット発話は除外し、人間発話のみ`max_history`個をLLMに渡す**
- `_filter_logs_by_human_count(logs, max_history, exclude_robot=True)`を使用

`_filter_logs_by_human_count()`の動作:
- 直近`max_history`個の人間発話を特定
- `exclude_robot=False`: その範囲に含まれるロボット発話も全て含める
  - 例: max_history=8で、直近8個の人間発話の間にロボット発話が3回あれば、合計11個の会話履歴を返す
- `exclude_robot=True`: ロボット発話を除外し、人間発話のみ`max_history`個を返す
  - 例: max_history=8で、直近8個の人間発話のみを返す（ロボット発話は含めない）

## テスト方法

```bash
# test_azure_human_convo.pyで動作確認
python app/test_azure_human_convo.py

# PPO学習の実行
python app/ppo_train.py
```

## 注意点

1. **トピック生成にはAzure OpenAIが必要**: `config.local.yaml`の`llm`設定が正しく設定されていることを確認
2. **地雷設定はYAMLで管理**: ハードコードではなく設定ファイルで管理
3. **3発話後からチェック**: 初期の会話では関係性推定をスキップして効率化
4. **recheck_after_turnsは廃止**: 不安定な場合は常にチェック
