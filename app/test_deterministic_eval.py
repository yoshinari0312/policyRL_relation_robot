#!/usr/bin/env python3
"""
決定的評価のテストスクリプト
学習前のモデルで評価を実行して動作確認
"""
from transformers import AutoTokenizer
from eval_deterministic import run_deterministic_eval, print_eval_results, FIXED_SCENARIOS
from config import get_config


def build_dummy_messages(logs, persona_list, env=None):
    """ダミーのメッセージ構築関数（テスト用）"""
    return [
        {"role": "system", "content": "You are a robot assistant."},
        {"role": "user", "content": "Conversation history and task..."}
    ]


def main():
    cfg = get_config()
    ppo_cfg = cfg.ppo

    model_id = getattr(ppo_cfg, "model_name_or_path", "unsloth/Qwen3-14B-unsloth-bnb-4bit")

    print("=" * 80)
    print("決定的評価のテスト")
    print("=" * 80)
    print(f"モデル: {model_id}")
    print(f"シナリオ数: {len(FIXED_SCENARIOS)}")
    print()

    # トークナイザーのロード
    print("トークナイザーをロード中...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # モデルのロード（簡易版：学習前のベースモデル）
    print("モデルをロード中...")
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )

    print("評価を実行中...")
    results = run_deterministic_eval(
        model=model,
        tokenizer=tokenizer,
        build_messages_fn=build_dummy_messages,
        temperature=0.0,
        seed=42,
        max_steps=10,
    )

    # 結果を表示
    print_eval_results(results)

    print("\n✓ テスト完了")


if __name__ == "__main__":
    main()
