#!/usr/bin/env python3
"""
介入判定モデル比較シミュレーション

GPT5（Azure OpenAI）とQwen3学習前元モデルの介入判定性能を比較するシミュレーション。
モデルを切り替えてテストできる汎用的なツール。

使用方法:
    # GPT5でテスト
    python compare_intervention_models.py --model gpt5 --num-sessions 5 --max-steps 6

    # Qwen3学習前元モデルでテスト
    python compare_intervention_models.py --model qwen3 --num-sessions 5 --max-steps 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import torch
from typing import Any, Dict, List, Optional
from pathlib import Path

from azure_clients import get_azure_chat_completion_client
from config import get_config
from env.convo_env import ConversationEnv

# Qwen3用のimport
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class InterventionModelComparator:
    """介入判定モデルの比較シミュレーター"""

    def __init__(
        self,
        model_type: str = "gpt5",  # "gpt5" or "qwen3"
        max_steps: int = 6,
        num_sessions: int = 5,
        verbose: bool = True,
    ):
        """
        Args:
            model_type: 使用するモデル ("gpt5" or "qwen3")
            max_steps: 非推奨。config.local.yamlのmax_roundsが優先されます。
            num_sessions: シミュレートするセッション数
            verbose: 詳細な出力を行うかどうか
        """
        self.model_type = model_type.lower()
        self.max_steps = max_steps
        self.num_sessions = num_sessions
        self.verbose = verbose

        if self.model_type not in ["gpt5", "qwen3"]:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'gpt5' or 'qwen3'")

        # 設定を読み込み
        cfg = get_config()

        # モデル固有の初期化
        if self.model_type == "gpt5":
            self._init_gpt5(cfg)
        else:  # qwen3
            self._init_qwen3(cfg)

        # 環境を初期化
        personas = getattr(cfg.env, "personas", {})
        self.env = ConversationEnv(
            max_steps=max_steps,
            personas=personas,
            include_robot=True,
            max_history=getattr(cfg.env, "max_history", 8),
            backend=getattr(cfg.scorer, "backend", "azure"),
            decay_factor=getattr(cfg.scorer, "decay_factor", 1.5),
            debug=getattr(cfg.env, "debug", False),
            reward_backend=getattr(cfg.env, "reward_backend", "rule"),
            evaluation_horizon=getattr(cfg.env, "evaluation_horizon", 3),
            time_penalty=getattr(cfg.env, "time_penalty", 0.01),
            terminal_bonus=getattr(cfg.env, "terminal_bonus", 0.25),
            intervention_cost=getattr(cfg.env, "intervention_cost", 0.02),
        )

        # 統計情報
        self.session_stats: List[Dict[str, Any]] = []

    def _init_gpt5(self, cfg):
        """GPT5 (Azure OpenAI) の初期化"""
        llm_cfg = getattr(cfg, "llm", None)
        self.client, self.deployment = get_azure_chat_completion_client(llm_cfg)
        if not self.client or not self.deployment:
            raise RuntimeError("Azure OpenAI client could not be initialized")

        print(f"✅ GPT5 (Azure OpenAI) initialized: {self.deployment}")

    def _init_qwen3(self, cfg):
        """Qwen3学習前元モデルの初期化"""
        ppo_cfg = getattr(cfg, "ppo", None)
        if not ppo_cfg:
            raise RuntimeError("PPO config not found")

        model_name_or_path = getattr(ppo_cfg, "model_name_or_path", None)
        if not model_name_or_path:
            raise RuntimeError("model_name_or_path not found in ppo config")

        print(f"🔄 Loading Qwen3 base model: {model_name_or_path}")

        # トークナイザーをロード
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"

        # モデルをロード（4bit量子化）
        device_map = "auto"
        use_quant = torch.cuda.is_available()

        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if use_quant:
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = quant_cfg
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

        self.qwen3_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **model_kwargs
        )
        self.qwen3_model.eval()

        # 生成パラメータ
        self.gen_kwargs = {
            "do_sample": True,
            "temperature": getattr(ppo_cfg, "temperature", 0.7),
            "top_p": getattr(ppo_cfg, "top_p", 0.9),
            "max_new_tokens": getattr(ppo_cfg, "max_new_tokens", 64),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        print(f"✅ Qwen3 base model loaded: {model_name_or_path}")

    def get_intervention_decision_gpt5(self, observation: str) -> tuple[str, str]:
        """
        GPT5を使って介入判定を行う

        Args:
            observation: 環境からの観測（プロンプト）

        Returns:
            (介入判定のJSON文字列, 生の出力)
        """
        messages = [
            {
                "role": "system",
                "content": "あなたは会話の関係性を安定化するためのロボット介入プランナーです。"
                "指示に従って、JSON形式で介入計画を出力してください。"
            },
            {
                "role": "user",
                "content": observation
            }
        ]

        max_attempts = 3
        raw_output = ""

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment,
                    messages=messages,
                )

                if response and hasattr(response, "choices") and response.choices:
                    message = response.choices[0].message
                    content = message.content if hasattr(message, "content") else ""
                    if content:
                        raw_output = content

                        # JSON部分を抽出（```json ... ```があれば）
                        content = content.strip()
                        if "```json" in content:
                            start = content.find("```json") + 7
                            end = content.find("```", start)
                            if end > start:
                                content = content[start:end].strip()
                        elif "```" in content:
                            start = content.find("```") + 3
                            end = content.find("```", start)
                            if end > start:
                                content = content[start:end].strip()

                        return content, raw_output
            except Exception as e:
                if self.verbose:
                    print(f"[GPT5] 介入判定試行 {attempt}/{max_attempts} 失敗: {e}")
                if attempt < max_attempts:
                    time.sleep(0.5 * attempt)

        # フォールバック: 介入しない判定を返す
        fallback = '{"intervene_now": false}'
        return fallback, raw_output or f"Error: All {max_attempts} attempts failed"

    def get_intervention_decision_qwen3(self, observation: str) -> tuple[str, str]:
        """
        Qwen3学習前元モデルを使って介入判定を行う

        Args:
            observation: 環境からの観測（プロンプト）

        Returns:
            (介入判定のJSON文字列, 生の出力)
        """
        # observationからメッセージ形式を構築
        # GPT5と同じプロンプト構造を使用
        from ppo_train import build_robot_messages

        try:
            # 環境の現在のログとペルソナから適切なメッセージを構築
            messages = build_robot_messages(self.env.logs, self.env.personas, env=self.env)
        except Exception as e:
            if self.verbose:
                print(f"[Qwen3] メッセージ構築失敗: {e}")
            # フォールバックメッセージ
            messages = [
                {"role": "system", "content": "You are a robot intervention planner. Output JSON format intervention decision."},
                {"role": "user", "content": observation}
            ]

        max_attempts = 3
        raw_output = ""

        for attempt in range(1, max_attempts + 1):
            try:
                # チャットテンプレートを適用
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # エンコード
                encoded = self.tokenizer(prompt, return_tensors="pt").to(self.qwen3_model.device)

                # 生成
                with torch.no_grad():
                    generated = self.qwen3_model.generate(
                        **encoded,
                        **self.gen_kwargs,
                    )

                # デコード
                gen_ids = generated[0, encoded["input_ids"].shape[-1]:]
                raw_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                # JSON部分を抽出
                content = raw_output

                # think tagsを除去
                if "</think>" in content:
                    content = content.split("</think>")[-1].strip()

                # JSON部分を抽出
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_candidate = content[start:end + 1]
                    try:
                        # JSON妥当性チェック
                        json.loads(json_candidate)
                        return json_candidate, raw_output
                    except json.JSONDecodeError:
                        pass

                # JSON抽出失敗時は全体を返す
                return content, raw_output

            except Exception as e:
                if self.verbose:
                    print(f"[Qwen3] 介入判定試行 {attempt}/{max_attempts} 失敗: {e}")
                if attempt < max_attempts:
                    time.sleep(0.1 * attempt)

        # フォールバック
        fallback = '{"intervene_now": false}'
        return fallback, raw_output or f"Error: All {max_attempts} attempts failed"

    def get_intervention_decision(self, observation: str) -> tuple[str, str]:
        """
        選択されたモデルで介入判定を行う

        Args:
            observation: 環境からの観測（プロンプト）

        Returns:
            (介入判定のJSON文字列, 生の出力)
        """
        if self.model_type == "gpt5":
            return self.get_intervention_decision_gpt5(observation)
        else:  # qwen3
            return self.get_intervention_decision_qwen3(observation)

    def run_session(self, session_id: int) -> Dict[str, Any]:
        """
        1セッションのシミュレーションを実行

        Args:
            session_id: セッションID

        Returns:
            セッションの統計情報
        """
        print("\n" + "=" * 80)
        print(f"セッション {session_id} 開始 ({self.model_type.upper()})")
        print("=" * 80)

        # 環境をリセット
        observation = self.env.reset()

        print(f"\n📌 話題: {self.env.current_topic}")
        print(f"👥 ペルソナ: {', '.join(self.env.persona_pool)}")
        print(f"🤖 使用モデル: {self.model_type.upper()}")

        # 初期会話履歴を表示（介入判定LLM用のmax_history発話）
        print(f"\n💬 初期会話履歴（reset()で生成、ステップ1の入力）:")
        from utils.log_filtering import filter_logs_by_human_count
        filtered_logs = filter_logs_by_human_count(self.env.logs, self.env.max_history, exclude_robot=False)
        for i, entry in enumerate(filtered_logs, 1):
            speaker = entry.get("speaker", "?")
            utterance = entry.get("utterance", "")
            print(f"  {i}. [{speaker}] {utterance}")

        # 初期状態の関係性を表示
        print(f"\n📊 初期状態の関係性:")
        human_utterance_count = sum(1 for log in self.env.logs if log.get('speaker') != 'ロボット')
        if hasattr(self.env, 'scorer') and human_utterance_count >= 3:
            try:
                from utils.log_filtering import filter_logs_by_human_count
                participants = [p for p in self.env.persona_pool if p][:3]
                if len(participants) == 3:
                    filtered = filter_logs_by_human_count(self.env.logs, self.env.max_history_relation, exclude_robot=True)
                    scores = self.env.scorer.get_scores(filtered, participants, return_trace=False, update_state=False)
                    from network_metrics import analyze_relations_from_scores
                    metrics = analyze_relations_from_scores(scores, participants)
                    print(f"  不安定トライアド数: {metrics.get('unstable_triads', '?')}")
                    edges = metrics.get("edges", {})
                    if edges:
                        print(f"  エッジスコア:")
                        for (p1, p2), score in sorted(edges.items()):
                            edge_name = f"{p1}{p2}"
                            if score > 0.5:
                                emoji = "😊"
                            elif score > 0:
                                emoji = "🙂"
                            elif score > -0.5:
                                emoji = "😐"
                            else:
                                emoji = "😠"
                            print(f"    {edge_name}: {score:+.3f} {emoji}")
            except Exception as e:
                print(f"  （関係性評価エラー: {e}）")
        else:
            print(f"  （まだ3発話未満）")

        print("-" * 80)

        # セッション統計
        total_reward = 0.0
        interventions = 0
        steps_taken = 0
        conversation_log: List[Dict[str, Any]] = []

        # 前ステップの最終関係性をキャッシュ（初期状態用）
        previous_step_rel = None

        # ステップループ
        done = False
        step = 0
        target_utterances = self.env.target_human_utterances
        while not done:
            step += 1
            print(f"\n{'='*80}")
            human_count = sum(1 for log in self.env.logs if log.get('speaker') != 'ロボット')
            print(f"🔹 ステップ {step} (人間発話: {human_count}/{target_utterances})")
            print(f"{'='*80}")

            # ステップ開始時（step()実行前）の会話ログを保存
            logs_before_step = [dict(entry) for entry in self.env.logs]

            # ステップ開始時の関係性を表示（前ステップの最終関係性を使用）
            is_stable_now = False
            if previous_step_rel is not None:
                # 前ステップの最終関係性を表示
                unstable_triads = previous_step_rel.get('unstable_triads', 0)
                is_stable_now = unstable_triads == 0
            else:
                # ステップ1の場合、初期状態の関係性を再評価
                human_utterance_count = sum(1 for log in self.env.logs if log.get('speaker') != 'ロボット')
                if hasattr(self.env, 'scorer') and human_utterance_count >= 3:
                    try:
                        from utils.log_filtering import filter_logs_by_human_count
                        participants = [p for p in self.env.persona_pool if p][:3]
                        if len(participants) == 3:
                            filtered = filter_logs_by_human_count(self.env.logs, self.env.max_history_relation, exclude_robot=True)
                            scores = self.env.scorer.get_scores(filtered, participants, return_trace=False, update_state=False)
                            from network_metrics import analyze_relations_from_scores
                            metrics = analyze_relations_from_scores(scores, participants)
                            unstable_triads = metrics.get('unstable_triads', 0)
                            is_stable_now = unstable_triads == 0
                            print(f"  不安定トライアド数: {unstable_triads}")
                            edges = metrics.get("edges", {})
                            if edges:
                                print(f"  エッジスコア:")
                                for (p1, p2), score in sorted(edges.items()):
                                    edge_name = f"{p1}{p2}"
                                    if score > 0.5:
                                        emoji = "😊"
                                    elif score > 0:
                                        emoji = "🙂"
                                    elif score > -0.5:
                                        emoji = "😐"
                                    else:
                                        emoji = "😠"
                                    print(f"    {edge_name}: {score:+.3f} {emoji}")
                    except Exception as e:
                        print(f"  （関係性評価エラー: {e}）")
                else:
                    print(f"  （まだ3発話未満）")

            # 安定状態の場合は介入判定をスキップ
            raw_output = ""
            if is_stable_now:
                action = '{"intervene_now": false}'
                raw_output = "安定状態のためスキップ"
            else:
                # 観測プロンプトを表示
                if self.verbose:
                    print(f"\n📝 観測情報（{self.model_type.upper()}への入力）:")
                    print("-" * 80)
                    # 会話履歴部分を抽出して表示（介入判定LLM用のmax_history発話）
                    if "履歴:" in observation:
                        history_section = observation.split("履歴:")[1].split("現在の関係スコア")[0].strip()
                        history_lines = [line for line in history_section.split("\n") if line.strip()]

                        # 介入判定LLM用のmax_history発話をフィルタリング
                        from utils.log_filtering import filter_logs_by_human_count
                        filtered_logs = filter_logs_by_human_count(self.env.logs, self.env.max_history, exclude_robot=False)

                        print("  会話履歴:")
                        for entry in filtered_logs:
                            speaker = entry.get("speaker", "?")
                            utterance = entry.get("utterance", "")
                            print(f"    [{speaker}] {utterance}")
                    # 関係スコア部分を抽出
                    if "現在の関係スコア" in observation:
                        score_line = [line for line in observation.split("\n") if "現在の関係スコア" in line]
                        if score_line:
                            print(f"  {score_line[0]}")
                    print("-" * 80)

                # 介入判定を実行
                action, raw_output = self.get_intervention_decision(observation)

            # 介入判定結果を表示（安定状態の場合はスキップ）
            if self.verbose and not is_stable_now:
                print(f"\n🤖 {self.model_type.upper()}介入判定結果:")

                # 生の出力を表示
                print(f"\n📋 生出力 ({self.model_type.upper()}):")
                print("-" * 60)
                print(raw_output)
                print("-" * 60)

                # JSON解析結果を表示
                try:
                    action_json = json.loads(action)
                    intervene_now = action_json.get('intervene_now', False)
                    print(f"\n📊 解析結果:")
                    print(f"  介入判定: {'✅ 介入する' if intervene_now else '❌ 介入しない'}")
                    if intervene_now:
                        print(f"  対象エッジ: {action_json.get('edge_to_change', 'N/A')}")
                        print(f"  戦略: {action_json.get('strategy', 'N/A')}")
                        print(f"  対象話者: {action_json.get('target_speaker', 'N/A')}")
                        if action_json.get('reasoning'):
                            print(f"  理由: {action_json['reasoning']}")
                        print(f"\n  ➡️  この判定に基づき、step()内でロボット発話が生成されます")
                except json.JSONDecodeError:
                    print(f"  ⚠️ JSON解析失敗: {action[:100]}...")

            # 環境でステップ実行
            print(f"\n⚙️  環境step()を実行中...")
            print(f"   → step()の内部処理:")
            print(f"      1. 人間発話生成: ステップ1または前ステップで介入した場合はスキップ、それ以外は1人間発話生成")
            print(f"      2. 関係性を評価（安定なら早期リターン）")
            print(f"      3. 不安定なら介入判定実行")
            print(f"      4. 介入する場合:")
            print(f"         - ロボット発話 + evaluation_horizon={self.env.evaluation_horizon}人間発話")
            print(f"         - 安定達成時: 最低terminal_bonus_duration={self.env.terminal_bonus_duration}人間発話分の追加チェック")
            print(f"      5. 報酬計算して返却")
            observation, reward, done, info = self.env.step(action)

            total_reward += reward
            steps_taken += 1
            if info.get("intervened", False):
                interventions += 1

            # ステップ実行中に生成された会話を表示
            print(f"\n💬 このステップで生成された会話（step()の実行結果）:")
            print("-" * 80)

            # ステップ前後の差分を計算
            new_entries = self.env.logs[len(logs_before_step):]

            if not new_entries:
                print("  （新しい発話なし）")
            else:
                # 実際の発話数をカウント
                human_count = 0
                robot_count = 0
                for entry in new_entries:
                    speaker = entry.get("speaker", "不明")
                    if speaker == "ロボット":
                        robot_count += 1
                    else:
                        human_count += 1

                # 介入した場合とそうでない場合で表示を分ける
                if info.get("intervened", False):
                    print(f"  ✅ 介入あり（合計{len(new_entries)}発話）:")
                    print(f"     - 人間: {human_count}発話")
                    print(f"     - ロボット: {robot_count}発話")
                    print(f"       ※ロボット1発話 + evaluation_horizon={self.env.evaluation_horizon}人間発話")
                    print(f"        安定達成時は最低terminal_bonus_duration={self.env.terminal_bonus_duration}人間発話分の追加チェック（実際は1ラウンド=3人間発話が最小単位）")
                    print()
                else:
                    print(f"  ❌ 介入なし:")
                    print(f"     - 人間: {human_count}発話")
                    if step == 1:
                        print(f"       ※ステップ1は発話なし（reset()で生成済み）、安定状態で早期リターン")
                    else:
                        print(f"       ※1人間発話のみ（安定状態で早期リターン）")
                    print()

                # 発話内容を表示
                for i, entry in enumerate(new_entries, 1):
                    speaker = entry.get("speaker", "不明")
                    utterance = entry.get("utterance", "")
                    if speaker == "ロボット":
                        print(f"\n  {i}. 🤖 【ロボット介入発話】")
                        print(f"     戦略: {info.get('plan', {}).get('strategy', 'N/A') if info.get('plan') else 'N/A'}")
                        print(f"     発話: {utterance}")
                        print()
                    else:
                        print(f"  {i}. 👤 [{speaker}] {utterance}")

                # 発話数のサマリー
                print(f"\n  📊 発話数サマリー: 人間 {human_count}発話, ロボット {robot_count}発話, 合計 {len(new_entries)}発話")

            # evaluation_horizon後の関係性（介入して安定になった場合のみ）
            if info.get("intervened", False) and "rel_after_horizon" in info and info.get('stable_after_horizon', False):
                rel_after = info.get("rel_after_horizon", {})
                print(f"\n📊 evaluation_horizon後の関係性:")
                print(f"  不安定トライアド数: {rel_after.get('unstable_triads', 0)}")
                print(f"  安定状態: ✅ はい")

                edges_after = rel_after.get("edges", {})
                if edges_after:
                    print(f"  エッジスコア:")
                    edge_order = [("A", "B"), ("B", "C"), ("A", "C")]
                    for edge_pair in edge_order:
                        if edge_pair in edges_after:
                            score = edges_after[edge_pair]
                            edge_name = f"{edge_pair[0]}{edge_pair[1]}"
                            if score > 0.5:
                                emoji = "😊"
                            elif score > 0:
                                emoji = "🙂"
                            elif score > -0.5:
                                emoji = "😐"
                            else:
                                emoji = "😠"
                            print(f"    {edge_name}: {score:+.3f} {emoji}")

            # 最終的な関係性スコアの出力
            rel = info.get("rel", {})
            print(f"\n📊 最終的な関係性評価:")
            print(f"  不安定トライアド数: {rel.get('unstable_triads', 0)}")
            print(f"  安定状態: {'✅ はい' if info.get('balanced', False) else '❌ いいえ'}")

            # 数値スコアの出力
            edges = rel.get("edges", {})
            if edges:
                print(f"  エッジスコア:")
                edge_order = [("A", "B"), ("B", "C"), ("A", "C")]
                for edge_pair in edge_order:
                    if edge_pair in edges:
                        score = edges[edge_pair]
                        edge_name = f"{edge_pair[0]}{edge_pair[1]}"
                        # スコアを視覚化
                        if score > 0.5:
                            emoji = "😊"
                        elif score > 0:
                            emoji = "🙂"
                        elif score > -0.5:
                            emoji = "😐"
                        else:
                            emoji = "😠"
                        print(f"    {edge_name}: {score:+.3f} {emoji}")

            # 報酬の詳細
            print(f"\n💰 報酬:")
            print(f"  ステップ報酬: {reward:.4f}")
            print(f"  累積報酬: {total_reward:.4f}")

            reward_breakdown = info.get("reward_breakdown", {})
            if reward_breakdown:
                print(f"  報酬内訳:")
                label_map = {
                    "delta_u_flip": "  📈 関係性改善効果 (Δu_flip)",
                    "counterfactual_u_flip": "  🔮 反実仮想不安定度",
                    "actual_u_flip": "  📉 実際の不安定度",
                    "intervention_cost": "  💸 介入コスト",
                    "time_penalty": "  ⏱️  時間ペナルティ",
                    "terminal_bonus": "  🎁 終了ボーナス"
                }
                for key, value in reward_breakdown.items():
                    label = label_map.get(key, f"  {key}")
                    print(f"    {label}: {value:.4f}")

            # 反実仮想情報
            if self.verbose and info.get("counterfactual_u_flip") is not None:
                print(f"\n🔮 反実仮想シミュレーション:")
                print(f"  反実仮想不安定度: {info.get('counterfactual_u_flip', 0):.4f}")
                print(f"  実際の不安定度: {info.get('actual_u_flip', 0):.4f}")
                print(f"  改善効果 (差分): {info.get('delta_u_flip', 0):.4f}")

            # 会話ログに追加
            conversation_log.append({
                "step": step,
                "action": action,
                "raw_output": raw_output,
                "reward": reward,
                "intervened": info.get("intervened", False),
                "robot_utterance": info.get("robot_utterance"),
                "replies": info.get("replies", []),
                "balanced": info.get("balanced", False),
                "new_entries": new_entries,
            })

            # ステップサマリー
            print(f"\n{'─'*80}")
            print(f"✅ ステップ {step} 完了")
            print(f"   介入: {'あり' if info.get('intervened', False) else 'なし'}")
            print(f"   報酬: {reward:.4f}")
            print(f"   累積報酬: {total_reward:.4f}")
            print(f"   安定状態: {'✅' if info.get('balanced', False) else '❌'}")
            print(f"{'─'*80}")

            # 次ステップのために最終関係性をキャッシュ
            previous_step_rel = info.get("rel", {})

        # セッション結果サマリー
        print("\n" + "=" * 80)
        print(f"セッション {session_id} 終了 ({self.model_type.upper()})")
        print("=" * 80)
        print(f"総ステップ数: {steps_taken}")
        print(f"総介入回数: {interventions}")
        print(f"総報酬: {total_reward:.4f}")
        print(f"平均ステップ報酬: {total_reward / steps_taken if steps_taken > 0 else 0:.4f}")

        return {
            "session_id": session_id,
            "model_type": self.model_type,
            "topic": self.env.current_topic,
            "total_reward": total_reward,
            "interventions": interventions,
            "steps": steps_taken,
            "average_reward": total_reward / steps_taken if steps_taken > 0 else 0,
            "conversation_log": conversation_log,
        }

    def run(self) -> None:
        """全セッションのシミュレーションを実行"""
        print("=" * 80)
        print(f"{self.model_type.upper()}介入判定シミュレーション開始")
        print("=" * 80)
        print(f"使用モデル: {self.model_type.upper()}")
        print(f"セッション数: {self.num_sessions}")
        print(f"設定されたラウンド数: {self.env.max_rounds}")
        print(f"1セッションあたりの目標人間発話数: {self.env.target_human_utterances}")

        start_time = time.time()

        # 各セッションを実行
        for i in range(1, self.num_sessions + 1):
            session_stats = self.run_session(i)
            self.session_stats.append(session_stats)

        elapsed_time = time.time() - start_time

        # 全体の統計を出力
        self.print_overall_statistics(elapsed_time)

    def print_overall_statistics(self, elapsed_time: float) -> None:
        """全セッションの統計を出力"""
        print("\n" + "=" * 80)
        print(f"全セッション統計 ({self.model_type.upper()})")
        print("=" * 80)

        if not self.session_stats:
            print("統計データがありません")
            return

        total_rewards = [s["total_reward"] for s in self.session_stats]
        total_interventions = sum(s["interventions"] for s in self.session_stats)
        total_steps = sum(s["steps"] for s in self.session_stats)

        print(f"使用モデル: {self.model_type.upper()}")
        print(f"総セッション数: {len(self.session_stats)}")
        print(f"総ステップ数: {total_steps}")
        print(f"総介入回数: {total_interventions}")
        print(f"総実行時間: {elapsed_time:.2f}秒")
        print()
        print(f"総報酬:")
        print(f"  合計: {sum(total_rewards):.4f}")
        print(f"  平均: {sum(total_rewards) / len(total_rewards):.4f}")
        print(f"  最大: {max(total_rewards):.4f}")
        print(f"  最小: {min(total_rewards):.4f}")
        print()
        print(f"セッションごとの平均介入回数: {total_interventions / len(self.session_stats):.2f}")
        print(f"セッションごとの平均ステップ数: {total_steps / len(self.session_stats):.2f}")

        # セッションごとの詳細
        print("\n" + "-" * 80)
        print("セッション別詳細:")
        print("-" * 80)
        for stat in self.session_stats:
            print(f"セッション {stat['session_id']}:")
            print(f"  話題: {stat['topic']}")
            print(f"  総報酬: {stat['total_reward']:.4f}")
            print(f"  介入回数: {stat['interventions']}")
            print(f"  ステップ数: {stat['steps']}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="介入判定モデル比較シミュレーション"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt5", "qwen3"],
        default="gpt5",
        help="使用するモデル (gpt5: Azure OpenAI GPT5, qwen3: Qwen3学習前元モデル) (デフォルト: gpt5)",
    )
    parser.add_argument(
        "--num-sessions",
        type=int,
        default=5,
        help="シミュレートするセッション数 (デフォルト: 5)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6,
        help="非推奨: config.local.yamlのmax_roundsを使用してください (デフォルト: 6)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="詳細な出力を抑制",
    )

    args = parser.parse_args()

    try:
        simulator = InterventionModelComparator(
            model_type=args.model,
            max_steps=args.max_steps,
            num_sessions=args.num_sessions,
            verbose=not args.quiet,
        )
        simulator.run()
    except KeyboardInterrupt:
        print("\n\nシミュレーション中断")
        sys.exit(1)
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()