# Swallow-8Bモデルの使用方法

## 概要

`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5` は日本語に強化されたLlama-3.1ベースのモデルです。このプロジェクトでは **PyTorch版** を使用してPPO強化学習を実施します。

## ⚠️ 重要：MLX vs PyTorch

### MLX版（`mlx-community/...`）について

- **Apple Silicon専用**（M1/M2/M3 Macのみ）
- **PPO学習には使用不可**（TRLがPyTorchベースのため）
- 推論専用として使いたい場合は `app/run_mlx_inference.py` を参照

### ✅ PyTorch版（推奨）

```yaml
# config.local.yaml
ppo:
  model_name_or_path: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
```

## セットアップ手順

### 1. 依存関係の確認

```bash
# 必要なライブラリは既にインストール済み（requirements.txt参照）
pip install transformers torch trl peft bitsandbytes accelerate
```

### 2. 設定ファイルの更新

`app/config.local.yaml` で以下を変更：

```yaml
ppo:
  model_name_or_path: "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5"
  
  # Swallow推奨パラメータ
  temperature: 0.7  # 日本語タスクで適度な多様性
  top_p: 0.9  # Swallowの標準設定
```

### 3. 学習の実行

```bash
# 通常の学習コマンド（変更なし）
python app/ppo_train.py
```

## モデルの特徴

### Swallow-8B の強み

1. **日本語性能**: 東工大が日本語コーパスで追加学習
2. **Llama-3.1互換**: Llama-3.1と同じアーキテクチャ
3. **会話タスク最適化**: Instruct版は対話に特化
4. **メモリ効率**: 4bit量子化で約5-6GB VRAM

### 推奨ハイパーパラメータ

```yaml
# Swallowに最適化された設定
temperature: 0.6-0.7    # 日本語会話で自然な多様性
top_p: 0.9              # 公式推奨値
max_new_tokens: 512     # 会話応答に十分な長さ
```

## unslothは必要か？

### 結論：**不要**

現在のコードは標準のTransformers APIを使用しているため、unslothなしで動作します：

```python
# ppo_train.py の実装（unsloth非依存）
from transformers import AutoModelForCausalLM
from peft import LoraConfig
from bitsandbytes import BitsAndBytesConfig

# 任意のHugging Faceモデルで動作
policy = AutoModelForCausalLMWithValueHead.from_pretrained(
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5",
    quantization_config=quant,  # BitsAndBytes 4bit
    peft_config=lora,           # LoRA
)
```

### unslothのメリット（オプション）

- **高速化**: 学習速度が約2倍（特にQLoRA使用時）
- **メモリ効率**: VRAM使用量が若干削減
- **便利機能**: 事前量子化モデルなど

ただし、**unslothがなくても正常に動作します**。

## トラブルシューティング

### VRAM不足エラー

```yaml
# config.local.yaml で調整
ppo:
  per_device_train_batch_size: 4  # 8 → 4に削減
  grad_accum_steps: 8              # 4 → 8に増加（実効バッチサイズ維持）
  max_memory_per_device_gib: 40   # GPU VRAMに応じて調整
```

### 生成品質の問題

```yaml
# より決定論的な出力が必要な場合
temperature: 0.3  # 0.7 → 0.3（探索的 → 決定論的）
top_p: 0.95       # より広い語彙から選択
```

### モデルダウンロードが遅い

```bash
# Hugging Face Hubから事前ダウンロード
huggingface-cli download tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5

# またはPythonで
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5")
```

## MLX推論（オプション・Mac専用）

学習はPyTorchで行い、**推論のみMLXを使いたい場合**：

```bash
# MLXのインストール（Apple Siliconのみ）
pip install mlx-lm

# 推論スクリプトの実行
python app/run_mlx_inference.py \
  --model mlx-community/Llama-3.1-Swallow-8B-Instruct-v0.5-4bit \
  --mode intervention
```

**注意**: MLX版は学習には使用できません（推論専用）。

## 比較：モデル選択のガイドライン

| モデル | 日本語性能 | 英語性能 | メモリ | 推奨用途 |
|--------|-----------|---------|--------|----------|
| **Swallow-8B** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 6GB | 日本語会話タスク |
| Llama-3.1-8B | ⭐⭐ | ⭐⭐⭐⭐⭐ | 6GB | 多言語タスク |
| Qwen3-8B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 6GB | バランス型 |
| Qwen3-14B | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 10GB | 高性能が必要 |

## 参考リンク

- [Swallow公式リポジトリ](https://github.com/swallow-llm/swallow-llm)
- [Hugging Face - Swallow-8B](https://huggingface.co/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5)
- [MLX版（Apple Silicon専用）](https://huggingface.co/mlx-community/Llama-3.1-Swallow-8B-Instruct-v0.5-4bit)
