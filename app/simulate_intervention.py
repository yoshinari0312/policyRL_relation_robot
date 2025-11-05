#!/usr/bin/env python3
"""
複数モデルを使った介入判定シミュレーション

GPT5（Azure OpenAI）またはローカルの学習前モデルで介入判定した時の
報酬や会話の様子を確認するシミュレーションスクリプト。

使用方法:
    # GPT5を使用
    python simulate_intervention.py --model gpt5 --num-sessions 5
    
    # Qwen3などのローカルモデルの学習前モデルを使用
    python simulate_intervention.py --model local --num-sessions 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

from azure_clients import get_azure_chat_completion_client
from config import get_config
from env.convo_env import ConversationEnv


class InterventionSimulator:
    """複数モデルを使って介入判定を行うシミュレーター"""

    def __init__(
        self,
        model_type: str = "gpt5",
        max_steps: int = 6,
        num_sessions: int = 5,
        verbose: bool = True,
    ):
        """
        Args:
            model_type: 使用するモデル ("gpt5" または "local")
            max_steps: 1エピソードあたりの最大ステップ数
            num_sessions: シミュレートするセッション数
            verbose: 詳細な出力を行うかどうか
        """
        self.model_type = model_type.lower()
        self.max_steps = max_steps
        self.num_sessions = num_sessions
        self.verbose = verbose

        # 設定を読み込み
        cfg = get_config()

        # モデルタイプに応じてクライアントを初期化
        if self.model_type == "gpt5":
            self._init_gpt5_client(cfg)
        elif self.model_type == "local":
            self._init_local_client(cfg)
        else:
            raise ValueError(f"サポートされていないモデルタイプ: {self.model_type}")

        # 環境を初期化
        personas = getattr(cfg.env, "personas", {})
        
        # max_stepsを明示的に設定
        # max_roundsは使用せず、max_stepsで終了条件を制御
        # max_historyは後方互換性のため渡すが、env内部で新しいパラメータを優先する
        self.env = ConversationEnv(
            max_steps=max_steps,
            personas=personas,
            include_robot=True,
            max_history=getattr(cfg.env, "max_history", 6),  # フォールバック値
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

    def _init_gpt5_client(self, cfg):
        """GPT5（Azure OpenAI）クライアントを初期化"""
        llm_cfg = getattr(cfg, "llm", None)
        self.client, self.deployment = get_azure_chat_completion_client(llm_cfg)
        if not self.client or not self.deployment:
            raise RuntimeError("Azure OpenAI client could not be initialized")
        print(f"✓ GPT5クライアント初期化完了 (deployment: {self.deployment})")

    def _init_local_client(self, cfg):
        """ローカル学習前モデルクライアントを初期化"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise RuntimeError("transformersライブラリが必要です: pip install transformers torch")

        # ローカルモデルのパスを取得（configから）
        ppo_cfg = getattr(cfg, "ppo", None)
        if not ppo_cfg:
            raise RuntimeError("config.ppo が見つかりません")
        
        self.model_name_or_path = getattr(ppo_cfg, "model_name_or_path", None)
        if not self.model_name_or_path:
            raise RuntimeError("config.ppo.model_name_or_path が設定されていません")

        print(f"🔄 モデルをロード中: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()
        print(f"✓ モデルロード完了: {self.model_name_or_path}")

    def get_intervention_decision(self, observation: str) -> str:
        """
        指定されたモデルを使って介入判定を行う

        Args:
            observation: 環境からの観測（プロンプト）

        Returns:
            介入判定の数字文字列（1-4）または後方互換性のためのJSON文字列
        """
        if self.model_type == "gpt5":
            return self._get_gpt5_decision(observation)
        elif self.model_type == "local":
            return self._get_local_decision(observation)
        else:
            return '4'  # デフォルトは介入しない

    def _get_gpt5_decision(self, observation: str) -> str:
        """GPT5で介入判定を行う"""
        # プロンプト最適化: 新しい数字形式
        system_content = """
あなたは、会話が不安定になっているときに、ロボットがどのように介入すれば関係を安定化できるかを判断するアシスタントです。
会話は三者（A, B, C）の間で行われており、一時的な対立や不調和が生じています。

入力として、
- 三者の会話文脈
- 各ペア（AB, BC, CA）の関係スコア（-1〜1）
- 今回の介入対象者（ターゲット）
- その対象関係ペア（エッジ）
が与えられます。

あなたの目的は、このターゲットが含まれる関係（エッジ）を安定化状態（+++、+--、-+-、--+）へ近づけるために、最も効果的な介入戦略を選択することです。

戦略の選択肢：
1. validate — 対象者の感情や意見を承認し、心理的安全性を構築する。
2. bridge — 対立する相手との共通点や協力の軸を見つけ、関係を再接続する。
3. plan — 対象者に次の行動や方針を示し、前向きな関係改善を促す。
4. no_intervention — まだ介入のタイミングではなく、自然な回復を見守る。

出力形式：
- 数字1桁のみを出力してください（1, 2, 3, または 4）
- 説明や補足は一切不要です。
- 与えられた会話文脈・関係スコア・ターゲット情報に基づいて選択してください。
"""

        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": observation
            }
        ]

        max_attempts = 3
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
                        # 生の出力をターミナルに表示
                        if self.verbose:
                            print(f"\n🤖 GPT5生出力:")
                            print("-" * 80)
                            print(content)
                            print("-" * 80)
                        
                        # 数字を抽出（1-4のいずれか）
                        content = content.strip()
                        
                        # 数字のみを探す
                        for char in content:
                            if char in '1234':
                                return char

                        return content
            except Exception as e:
                if self.verbose:
                    print(f"[GPT5] 介入判定試行 {attempt}/{max_attempts} 失敗: {e}")
                if attempt < max_attempts:
                    time.sleep(0.5 * attempt)

        # フォールバック: 介入しない判定を返す
        return '4'

    def _get_local_decision(self, observation: str) -> str:
        """ローカルモデル（Qwen3など）で介入判定を行う"""
        import torch
        
        # プロンプト最適化: 新しい数字形式
        system_content = """
あなたは、会話が一時的に不安定になっている場面で、ロボットがどのような介入を行えば、または行わないことで、最も良い結果（関係の安定化）を導けるかを判断します。
会話には感情のズレや対立が含まれていますが、状況によっては人間同士が自然に回復することもあります。
したがって、必ずしもロボットが発言する必要はありません。
会話は三者（A, B, C）の間で行われており、一時的な対立や不調和が生じています。

入力として、
- 三者の会話文脈
- 各ペア（AB, BC, CA）の関係スコア（-1〜1）
- 今回の介入対象者（ターゲット）
- その対象関係ペア（エッジ）
が与えられます。

あなたの目的は、このターゲットが含まれる関係（エッジ）をより安定(+)に近づけるために、「今この瞬間にどの戦略を選ぶことが最も効果的か」を判断することです。

戦略の選択肢：
1. validate — 対象者の感情や意見を承認し、心理的安全性を構築する。
2. bridge — 対立する相手との共通点や協力の軸を見つけ、関係を再接続する。
3. plan — 対象者に次の行動や方針を示し、前向きな関係改善を促す。
4. no_intervention — 対立が軽度で、介入が逆効果になりそうなときや自然回復が見込める時に選ぶ。

出力形式：
- 数字1桁と理由のみを出力してください（1, 2, 3, 4）
- 理由は文脈に沿ったものとしてください
- 説明や補足は一切不要です。
- 与えられた入力情報に基づいて選択してください。

出力例
1,理由：〜
2,理由：〜
3,理由：〜
4,理由：〜
"""
        
        # Qwen3モデルの場合のみ、\no_thinkを追加
        if hasattr(self, 'model_name_or_path') and self.model_name_or_path and "Qwen3" in self.model_name_or_path:
            system_content = "\\no_think\n" + system_content
        
        # モデルのチャットテンプレートを使用
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": observation}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        try:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                )
            
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # 生の出力をターミナルに表示
            if self.verbose:
                print(f"\n🤖 モデル生出力:")
                print("-" * 80)
                print(response)
                print("-" * 80)
            
            # 数字を抽出（1-4のいずれか）
            response = response.strip()
            
            # 数字のみを探す
            for char in response:
                if char in '1234':
                    return char
            
            return response
            
        except Exception as e:
            if self.verbose:
                print(f"[ローカル] 介入判定失敗: {e}")
            return '4'

    def run_session(self, session_id: int) -> Dict[str, Any]:
        """
        1セッションのシミュレーションを実行

        Args:
            session_id: セッションID

        Returns:
            セッションの統計情報
        """
        print("\n" + "=" * 80)
        print(f"セッション {session_id} 開始 (モデル: {self.model_type.upper()})")
        print("=" * 80)

        # 環境をリセット
        observation = self.env.reset()

        print(f"\n📌 話題: {self.env.current_topic}")
        print(f"👥 ペルソナ: {', '.join(self.env.persona_pool)}")

        # 初期会話履歴を表示（介入判定LLM用のintervention_max_history発話）
        print(f"\n💬 初期会話履歴（reset()で生成、ステップ1の入力）:")
        from utils.log_filtering import filter_logs_by_human_count
        filtered_logs = filter_logs_by_human_count(self.env.logs, self.env.intervention_max_history, exclude_robot=False)
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
        
        # 介入戦略の統計
        strategy_counts = {
            'validate': 0,
            'bridge': 0,
            'plan': 0,
            'no_intervention': 0
        }
        llm_inference_count = 0  # LLMで推論した回数（安定状態を除く）
        stable_skip_count = 0     # 安定状態でスキップした回数

        # 前ステップの最終関係性をキャッシュ（初期状態用）
        previous_step_rel = None

        # ステップループ
        done = False
        step = 0
        while not done:
            step += 1
            print(f"\n{'='*80}")
            print(f"🔹 ステップ {step}/{self.env.max_steps}")
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
            if is_stable_now:
                stable_skip_count += 1
                if self.verbose:
                    print(f"\n⏭️  安定状態のため介入判定をスキップ")
            else:
                # LLMで推論を実行
                llm_inference_count += 1
                
                # 観測プロンプトを表示
                if self.verbose:
                    print(f"\n📝 観測情報（{self.model_type.upper()}への入力）:")
                    print("-" * 80)
                    # 会話履歴部分を抽出して表示（介入判定LLM用のintervention_max_history発話）
                    if "履歴:" in observation:
                        history_section = observation.split("履歴:")[1].split("現在の関係スコア")[0].strip()
                        history_lines = [line for line in history_section.split("\n") if line.strip()]

                        # 介入判定LLM用のintervention_max_history発話をフィルタリング
                        from utils.log_filtering import filter_logs_by_human_count
                        filtered_logs = filter_logs_by_human_count(self.env.logs, self.env.intervention_max_history, exclude_robot=False)

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
                    if "エッジ" in observation:
                        edge_line = [line for line in observation.split("\n") if "エッジ" in line]
                        if edge_line:
                            print(f"  {edge_line[0]}")
                    if "ターゲット" in observation:
                        target_line = [line for line in observation.split("\n") if "ターゲット" in line]
                        if target_line:
                            print(f"  {target_line[0]}")
                    print("-" * 80)

                # モデルで介入判定
                action = self.get_intervention_decision(observation)
                
                # 戦略をカウント
                strategy_map = {
                    '1': 'validate',
                    '2': 'bridge',
                    '3': 'plan',
                    '4': 'no_intervention'
                }
                strategy = strategy_map.get(action.strip(), 'no_intervention')
                if strategy in strategy_counts:
                    strategy_counts[strategy] += 1

            # 介入判定結果を表示（安定状態の場合はスキップ）
            if self.verbose and not is_stable_now:
                print(f"\n🤖 {self.model_type.upper()}介入判定結果:")
                # 数字形式（1-4）で出力される
                strategy_map = {
                    '1': 'validate',
                    '2': 'bridge',
                    '3': 'plan',
                    '4': 'no_intervention'
                }
                strategy = strategy_map.get(action.strip(), 'unknown')
                
                if strategy == 'no_intervention':
                    print(f"  介入判定: ❌ 介入しない (4)")
                elif strategy != 'unknown':
                    print(f"  介入判定: ✅ 介入する")
                    print(f"  戦略: {strategy} ({action.strip()})")
                    print(f"\n  ➡️  この判定に基づき、step()内でロボット発話が生成されます")
                    print(f"      edge_to_change と target_speaker は環境側で自動決定されます")
                else:
                    print(f"  ⚠️ 不明な出力: {action}")

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
        print(f"セッション {session_id} 終了")
        print("=" * 80)
        print(f"総ステップ数: {steps_taken}")
        print(f"総介入回数: {llm_inference_count} (安定状態スキップ: {stable_skip_count}回)")
        print(f"総報酬: {total_reward:.4f}")
        print(f"平均ステップ報酬: {total_reward / steps_taken if steps_taken > 0 else 0:.4f}")
        
        # 戦略使用統計
        print(f"\n介入戦略の使用回数:")
        print(f"  validate: {strategy_counts['validate']}回")
        print(f"  bridge: {strategy_counts['bridge']}回")
        print(f"  plan: {strategy_counts['plan']}回")
        print(f"  no_intervention: {strategy_counts['no_intervention']}回")

        return {
            "session_id": session_id,
            "topic": self.env.current_topic,
            "total_reward": total_reward,
            "interventions": interventions,
            "steps": steps_taken,
            "average_reward": total_reward / steps_taken if steps_taken > 0 else 0,
            "conversation_log": conversation_log,
            "llm_inference_count": llm_inference_count,
            "stable_skip_count": stable_skip_count,
            "strategy_counts": strategy_counts,
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
        print(f"max_steps: {self.env.max_steps}")

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
        print("全セッション統計")
        print("=" * 80)

        if not self.session_stats:
            print("統計データがありません")
            return

        total_rewards = [s["total_reward"] for s in self.session_stats]
        total_interventions = sum(s["interventions"] for s in self.session_stats)
        total_steps = sum(s["steps"] for s in self.session_stats)
        total_llm_inferences = sum(s["llm_inference_count"] for s in self.session_stats)
        total_stable_skips = sum(s["stable_skip_count"] for s in self.session_stats)
        
        # 戦略統計を集計
        total_strategy_counts = {
            'validate': 0,
            'bridge': 0,
            'plan': 0,
            'no_intervention': 0
        }
        for stat in self.session_stats:
            for strategy, count in stat["strategy_counts"].items():
                total_strategy_counts[strategy] += count

        print(f"使用モデル: {self.model_type.upper()}")
        print(f"総セッション数: {len(self.session_stats)}")
        print(f"総ステップ数: {total_steps}")
        print(f"総介入回数: {total_llm_inferences} (安定状態でスキップ: {total_stable_skips}回)")
        print(f"  ※安定状態の場合は介入判定LLMを呼び出さずにスキップします")
        print(f"総実行時間: {elapsed_time:.2f}秒")
        print()
        print(f"総報酬:")
        print(f"  合計: {sum(total_rewards):.4f}")
        print(f"  平均: {sum(total_rewards) / len(total_rewards):.4f}")
        print(f"  最大: {max(total_rewards):.4f}")
        print(f"  最小: {min(total_rewards):.4f}")
        print()
        print(f"介入戦略の使用統計:")
        print(f"  validate: {total_strategy_counts['validate']}回 ({total_strategy_counts['validate']/max(total_llm_inferences,1)*100:.1f}%)")
        print(f"  bridge: {total_strategy_counts['bridge']}回 ({total_strategy_counts['bridge']/max(total_llm_inferences,1)*100:.1f}%)")
        print(f"  plan: {total_strategy_counts['plan']}回 ({total_strategy_counts['plan']/max(total_llm_inferences,1)*100:.1f}%)")
        print(f"  no_intervention: {total_strategy_counts['no_intervention']}回 ({total_strategy_counts['no_intervention']/max(total_llm_inferences,1)*100:.1f}%)")
        print()
        print(f"セッションごとの平均介入回数: {total_interventions / len(self.session_stats):.2f}")
        print(f"セッションごとの平均ステップ数: {total_steps / len(self.session_stats):.2f}")
        print(f"セッションごとの平均介入回数: {total_llm_inferences / len(self.session_stats):.2f}")

        # セッションごとの詳細
        print("\n" + "-" * 80)
        print("セッション別詳細:")
        print("-" * 80)
        for stat in self.session_stats:
            print(f"セッション {stat['session_id']}:")
            print(f"  話題: {stat['topic']}")
            print(f"  総報酬: {stat['total_reward']:.4f}")
            print(f"  ステップ数: {stat['steps']}")
            print(f"  介入回数: {stat['llm_inference_count']} (安定スキップ: {stat['stable_skip_count']}回)")
            
            # 戦略使用統計
            strategy_counts = stat['strategy_counts']
            total_inferences = stat['llm_inference_count']
            if total_inferences > 0:
                print(f"  戦略使用:")
                print(f"    validate: {strategy_counts['validate']}回 ({strategy_counts['validate']/total_inferences*100:.1f}%)")
                print(f"    bridge: {strategy_counts['bridge']}回 ({strategy_counts['bridge']/total_inferences*100:.1f}%)")
                print(f"    plan: {strategy_counts['plan']}回 ({strategy_counts['plan']/total_inferences*100:.1f}%)")
                print(f"    no_intervention: {strategy_counts['no_intervention']}回 ({strategy_counts['no_intervention']/total_inferences*100:.1f}%)")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="複数モデルを使った介入判定シミュレーション"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt5", "local"],
        default="gpt5",
        help="使用するモデル (デフォルト: gpt5)",
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
        default=8,
        help="1エピソードあたりの最大ステップ数 (デフォルト: 8)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="詳細な出力を抑制",
    )

    args = parser.parse_args()

    try:
        simulator = InterventionSimulator(
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
