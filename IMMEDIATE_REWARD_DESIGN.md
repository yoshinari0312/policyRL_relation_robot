# 即座報酬システム設計（Immediate Reward with Internal Lookahead）

## 概要

このドキュメントは、会話ロボットの強化学習における報酬システムの設計を定義します。

**設計方針**: PPOが学習可能なように、`step(N)`の返り値として即座に報酬を返します。ただし、報酬計算のために、`step(N)`内部でevaluation_horizon回（+必要に応じてterminal_bonus_duration回）の人間発話を先読み生成します。

これにより、遅延報酬の問題（credit assignment problem）を回避しつつ、介入の長期的効果を評価できます。

## パラメータ

- `evaluation_horizon`: 介入後に内部生成する人間発話の回数（**3推奨**: 3人全員の発話を反映）
- `terminal_bonus_duration`: 安定状態が続いた場合にボーナスを与えるために必要な連続安定ステップ数
- `time_penalty`: 不安定状態が続くことに対するペナルティ（1ステップごと）
- `intervention_cost`: ロボットが介入する際のコスト
- `terminal_bonus`: 安定状態がterminal_bonus_duration回続いた場合のボーナス

## step(N)の処理フロー（確定版）

### 基本フロー

```
step(N):
  1. 人間発話を1回生成
  2. 関係性を推定（3人全員の発話が揃っていない場合はスキップ）
  3. 関係性で分岐:
     ├─ 安定（不安定トライアド数 = 0）
     │   └─ 報酬なし、観測を返して終了
     │
     └─ 不安定（不安定トライアド数 > 0）
         └─ エージェントに観測を渡し、介入判定（action）を受け取る
            ├─ 介入しない → Case A
            └─ 介入する → Case B
```

---

## Case A: 介入しない場合

### 処理フロー

```
step(N):
  1. タイムペナルティを計算
     reward = -time_penalty

  2. step(N)の返り値として即座に返す
     return (observation, reward, done, info)
```

### 報酬

| 報酬要素 | 値 | 付与タイミング |
|---------|-----|--------------|
| タイムペナルティ | -time_penalty | step(N)の返り値（即座） |

### 重要

- 反実仮想シミュレーションは**実行しない**
- evaluation_horizon回の先読み生成も**しない**
- 即座にペナルティを返してstep()を終了

---

## Case B: 介入する場合

### 処理フロー（step(N)内で完結）

```
step(N):
  1. スナップショットを保存（反実仮想用）

  2. ロボット発話を会話ログに追加

  3. 並行処理を開始（async/await）:
     ├─ 実世界: evaluation_horizon回の人間発話を生成
     └─ 反実仮想: 介入しなかった場合をシミュレーション（evaluation_horizon回）

  4. 両方の完了を待つ（await）

  5. 関係性を評価:
     ├─ actual_u_flip: 実世界のevaluation_horizon発話後の不安定度
     └─ counterfactual_u_flip: 反実仮想世界の不安定度

  6. 実世界の状態で分岐:
     ├─ 不安定のまま → Case B-1
     └─ 安定になった → Case B-2
```

---

### Case B-1: evaluation_horizon後も不安定

```
step(N):
  ... (上記1-5まで同じ)

  6. 報酬を計算:
     delta = counterfactual_u_flip - actual_u_flip
     reward = delta - intervention_cost - time_penalty

     内訳:
       - 関係性改善効果: delta
       - 介入コスト: -intervention_cost
       - タイムペナルティ: -time_penalty

  7. step(N)の返り値として即座に返す:
     return (observation, reward, done, info)
```

### 報酬

| 報酬要素 | 計算式 | 説明 |
|---------|--------|------|
| 関係性改善効果 | counterfactual_u_flip - actual_u_flip | 介入効果（正なら改善） |
| 介入コスト | -intervention_cost | 介入のコスト |
| タイムペナルティ | -time_penalty | 不安定状態継続のペナルティ |
| **合計** | delta - cost - penalty | step(N)で即座に返す |

---

### Case B-2: evaluation_horizon後に安定

```
step(N):
  ... (上記1-5まで同じ)

  6. 実世界がevaluation_horizon後に安定 → さらに先読み

  7. terminal_bonus_duration回の人間発話を追加生成:
     for i in range(terminal_bonus_duration):
       - 人間発話を1回生成
       - 関係性を評価
       - 不安定に戻った？
         └─ Yes → Case B-2-a（ループを抜ける）
         └─ No → 継続

     terminal_bonus_duration回全て安定 → Case B-2-b
```

---

### Case B-2-a: 途中で不安定に戻った

```
step(N):
  ... (上記1-7で途中まで生成)

  8. 報酬を計算:
     delta = counterfactual_u_flip - actual_u_flip
     reward = delta - intervention_cost - time_penalty

     内訳:
       - 関係性改善効果: delta
       - 介入コスト: -intervention_cost
       - タイムペナルティ: -time_penalty
       - 安定ボーナス: なし（途中で不安定に戻ったため）

  9. step(N)の返り値として即座に返す:
     return (observation, reward, done, info)
```

### 報酬

| 報酬要素 | 計算式 | 説明 |
|---------|--------|------|
| 関係性改善効果 | counterfactual_u_flip - actual_u_flip | 介入効果 |
| 介入コスト | -intervention_cost | 介入のコスト |
| タイムペナルティ | -time_penalty | 不安定状態継続のペナルティ |
| **合計** | delta - cost - penalty | step(N)で即座に返す |

---

### Case B-2-b: 安定が持続

```
step(N):
  ... (上記1-7で全て生成完了)

  8. 報酬を計算:
     delta = counterfactual_u_flip - actual_u_flip
     reward = delta - intervention_cost - time_penalty + terminal_bonus

     内訳:
       - 関係性改善効果: delta
       - 介入コスト: -intervention_cost
       - タイムペナルティ: -time_penalty
       - 安定ボーナス: +terminal_bonus

  9. step(N)の返り値として即座に返す:
     return (observation, reward, done, info)
```

### 報酬

| 報酬要素 | 計算式 | 説明 |
|---------|--------|------|
| 関係性改善効果 | counterfactual_u_flip - actual_u_flip | 介入効果 |
| 介入コスト | -intervention_cost | 介入のコスト |
| タイムペナルティ | -time_penalty | 不安定状態継続のペナルティ |
| 安定ボーナス | +terminal_bonus | 安定持続のボーナス |
| **合計** | delta - cost - penalty + bonus | step(N)で即座に返す |

---

## 並行処理の実装

### 実世界と反実仮想世界の並行シミュレーション

介入する場合（Case B）では、以下の2つの処理を並行実行します:

```python
async def step_with_intervention():
    # 同時に開始
    actual_task = generate_actual_conversation(evaluation_horizon)
    counterfactual_task = simulate_counterfactual(evaluation_horizon)

    # 両方の完了を待つ
    actual_result = await actual_task
    counterfactual_result = await counterfactual_task

    # 差分報酬を計算
    delta = counterfactual_result.u_flip - actual_result.u_flip
    reward = delta - intervention_cost - time_penalty

    # 安定チェック
    if actual_result.is_stable:
        # terminal_bonus_durationチェック
        reward += check_stability_bonus()

    return reward
```

### メリット

- 実世界の人間発話生成（LLM呼び出し）と反実仮想シミュレーション（別のLLM呼び出し）を並列実行
- 全体の待ち時間を短縮（直列実行の約半分）

---

## まとめ

| ケース | 報酬付与タイミング | 報酬内容 |
|-------|-----------------|---------|
| 安定 | なし | 報酬なし |
| 不安定 → 介入しない | step(N)で即座 | -time_penalty |
| 不安定 → 介入 → 不安定のまま | step(N)で即座 | delta - cost - penalty |
| 不安定 → 介入 → 安定 → 途中で不安定 | step(N)で即座 | delta - cost - penalty |
| 不安定 → 介入 → 安定 → 持続 | step(N)で即座 | delta - cost - penalty + bonus |

### 重要な特徴

1. **遅延なし**: 全ての報酬はstep(N)の返り値として即座に返される
2. **先読み評価**: step(N)内部でevaluation_horizon（+terminal_bonus_duration）回の発話を生成
3. **PPO互換**: 標準的なPPOアルゴリズムで学習可能
4. **並行処理**: 実世界と反実仮想世界を並列シミュレーションで高速化

この設計により、ロボットは以下を学習します:

1. **介入のタイミング**: 不安定な時に介入する
2. **介入の効果**: 関係性を改善する介入内容を選ぶ
3. **安定の維持**: 安定状態を持続させる介入をする
4. **効率性**: 無駄な介入を避ける（介入コストがかかるため）

---

## 実装上の注意点

### 1. 非同期処理の導入

- `step()`メソッドを`async def step()`に変更
- 人間発話生成とシミュレーションを`asyncio.gather()`で並行実行

### 2. evaluation_horizon中の会話管理

- step(N)内部で生成されたevaluation_horizon回の発話は、環境の状態に反映される
- 次のstep(N+1)では、これらの発話を含む状態から開始

### 3. terminal_bonus_durationチェック

- evaluation_horizon後に安定になった場合のみ、追加でterminal_bonus_duration回生成
- 毎回関係性を評価し、不安定に戻ったらループを抜ける

### 4. 関係性評価の最小発話数

- 3人全員の発話が揃うまで（最初の3発話）は関係性評価をスキップ
- この間は報酬も付与しない
