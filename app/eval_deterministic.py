"""
決定的評価（Deterministic Evaluation）
- 温度0、固定シード、固定シナリオで評価
- チェックポイント選定のためのKPI測定
"""
from __future__ import annotations
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from env.convo_env import ConversationEnv
from config import get_config


@dataclass
class EvalScenario:
    """評価用の固定シナリオ"""
    scenario_id: str
    topic: str
    trigger: str
    personas_with_trigger: List[str]
    initial_conversation: List[Dict[str, str]]
    initial_relation: Dict[str, Any]


# 固定シナリオの定義
FIXED_SCENARIOS = [
    EvalScenario(
        scenario_id="scenario_001_gender",
        topic="海外でのジェンダー観の違い",
        trigger="国際・異文化",
        personas_with_trigger=["A", "C"],
        initial_conversation=[
            {
                "speaker": "A",
                "utterance": "C、また上から目線で語るつもり？海外のジェンダー観なんて偉そうに語っても、実際に行ったことあるの？"
            },
            {
                "speaker": "B",
                "utterance": "まあまあ、A。そんなに構えなくてもいいじゃない？いろんな国の話を聞くだけでもけっこう面白いよ。"
            },
            {
                "speaker": "C",
                "utterance": "A、そんな皮肉言わなくてもいいだろ。行ったことくらいあるし、聞く耳くらい持てよ。"
            }
        ],
        initial_relation={
            "is_stable": False,
            "edges": {
                "A,B": 0.3,
                "A,C": -0.8,
                "B,C": 0.1
            }
        }
    ),
    # 他のシナリオも追加可能
]


@dataclass
class EvalMetrics:
    """評価メトリクス"""
    stable_reach_rate: float  # 安定到達率
    avg_turns_to_stable: float  # 平均到達ターン数
    intervention_count: int  # 介入回数
    non_intervention_count: int  # 非介入回数（intervene_now=false）
    noop_baseline_diff: float  # no-opベースラインとの差分
    harmful_intervention_rate: float  # 有害介入率
    total_reward: float  # 総報酬
    avg_reward: float  # 平均報酬


def run_deterministic_eval(
    model,
    tokenizer,
    build_messages_fn,
    scenarios: Optional[List[EvalScenario]] = None,
    temperature: float = 0.0,
    seed: int = 42,
    max_steps: int = 10,
) -> Dict[str, Any]:
    """
    決定的評価を実行

    Args:
        model: 評価対象のモデル
        tokenizer: トークナイザー
        build_messages_fn: メッセージ構築関数
        scenarios: 評価シナリオのリスト（Noneの場合はFIXED_SCENARIOSを使用）
        temperature: 生成温度（0.0で決定論的）
        seed: 乱数シード
        max_steps: 最大ステップ数

    Returns:
        評価結果の辞書
    """
    if scenarios is None:
        scenarios = FIXED_SCENARIOS

    np.random.seed(seed)

    # シナリオごとの結果を収集
    scenario_results = []

    for scenario in scenarios:
        # モデルあり評価
        model_result = _eval_single_scenario(
            scenario=scenario,
            model=model,
            tokenizer=tokenizer,
            build_messages_fn=build_messages_fn,
            temperature=temperature,
            max_steps=max_steps,
            use_model=True,
        )

        # no-opベースライン評価（介入なし）
        noop_result = _eval_single_scenario(
            scenario=scenario,
            model=None,
            tokenizer=None,
            build_messages_fn=None,
            temperature=0.0,
            max_steps=max_steps,
            use_model=False,
        )

        # no-op差分を計算
        noop_diff = model_result["total_reward"] - noop_result["total_reward"]

        scenario_results.append({
            "scenario_id": scenario.scenario_id,
            "model_result": model_result,
            "noop_result": noop_result,
            "noop_diff": noop_diff,
        })

    # 集約メトリクスを計算
    metrics = _aggregate_metrics(scenario_results)

    return {
        "metrics": metrics,
        "scenario_results": scenario_results,
    }


def _eval_single_scenario(
    scenario: EvalScenario,
    model,
    tokenizer,
    build_messages_fn,
    temperature: float,
    max_steps: int,
    use_model: bool,
) -> Dict[str, Any]:
    """単一シナリオの評価"""
    cfg = get_config()

    # 環境の初期化
    env = ConversationEnv(
        max_steps=max_steps,
        debug=False,
    )

    # 固定シナリオで初期化
    env.reset()

    # 初期会話を設定
    env.logs = []
    for utterance in scenario.initial_conversation:
        env.logs.append({
            "speaker": utterance["speaker"],
            "utterance": utterance["utterance"],
        })

    # トピックとトリガーを設定
    env.current_topic = scenario.topic
    env.current_topic_trigger = scenario.trigger

    # メトリクス収集
    total_reward = 0.0
    intervention_count = 0
    non_intervention_count = 0
    harmful_intervention_count = 0
    turns_to_stable = None
    reached_stable = False

    step = 0
    done = False

    while not done and step < max_steps:
        step += 1

        if use_model:
            # モデルを使って介入判定
            persona_list = list(env.persona_pool)
            messages = build_messages_fn(env.logs, persona_list, env=env)
            query = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 生成
            inputs = tokenizer(query, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=temperature if temperature > 0 else 1e-8,  # 完全な0はエラーになる場合がある
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # JSONを抽出
            plan_text = _extract_json_from_response(response)
        else:
            # no-opベースライン（常に介入しない）
            plan_text = '{"intervene_now": false}'

        # プランを解析
        try:
            plan = json.loads(plan_text)
            intervened = plan.get("intervene_now", False)
            if intervened:
                intervention_count += 1
            else:
                non_intervention_count += 1
        except json.JSONDecodeError:
            # パースエラー時は非介入扱い
            non_intervention_count += 1

        # ステップ実行
        obs, reward, done, info = env.step(plan_text)
        total_reward += reward

        # 有害介入の検出（介入後に関係性が悪化した場合）
        if info.get("intervened"):
            rel_before = info.get("status", {})
            rel_after = info.get("rel_after_horizon", {})

            if rel_before and rel_after:
                unstable_before = rel_before.get("unstable_triads", 0) if isinstance(rel_before, dict) else 0
                unstable_after = rel_after.get("unstable_triads", 0) if isinstance(rel_after, dict) else 0

                # 介入後に不安定度が増加した場合は有害介入
                if unstable_after > unstable_before:
                    harmful_intervention_count += 1

        # 安定到達の確認
        current_rel = env.relation_snapshot()
        if isinstance(current_rel, dict):
            metrics_data = current_rel.get("metrics", current_rel)
            is_stable = metrics_data.get("unstable_triads", 0) == 0

            if is_stable and not reached_stable:
                reached_stable = True
                turns_to_stable = step

    # 有害介入率
    harmful_rate = harmful_intervention_count / intervention_count if intervention_count > 0 else 0.0

    return {
        "total_reward": total_reward,
        "avg_reward": total_reward / step if step > 0 else 0.0,
        "intervention_count": intervention_count,
        "non_intervention_count": non_intervention_count,
        "harmful_intervention_count": harmful_intervention_count,
        "harmful_intervention_rate": harmful_rate,
        "reached_stable": reached_stable,
        "turns_to_stable": turns_to_stable,
        "total_steps": step,
    }


def _extract_json_from_response(response: str) -> str:
    """レスポンスからJSON部分を抽出"""
    start = response.find("{")
    end = response.rfind("}")
    if start != -1 and end != -1 and end > start:
        return response[start:end+1]
    return '{"intervene_now": false}'


def _aggregate_metrics(scenario_results: List[Dict[str, Any]]) -> EvalMetrics:
    """複数シナリオの結果を集約"""
    total_scenarios = len(scenario_results)

    # 安定到達率
    reached_stable_count = sum(
        1 for r in scenario_results if r["model_result"]["reached_stable"]
    )
    stable_reach_rate = reached_stable_count / total_scenarios if total_scenarios > 0 else 0.0

    # 平均到達ターン数（到達したシナリオのみ）
    turns_list = [
        r["model_result"]["turns_to_stable"]
        for r in scenario_results
        if r["model_result"]["turns_to_stable"] is not None
    ]
    avg_turns_to_stable = np.mean(turns_list) if turns_list else float('inf')

    # 介入回数と非介入回数
    total_interventions = sum(r["model_result"]["intervention_count"] for r in scenario_results)
    total_non_interventions = sum(r["model_result"]["non_intervention_count"] for r in scenario_results)

    # no-op差分の平均
    noop_diffs = [r["noop_diff"] for r in scenario_results]
    avg_noop_diff = np.mean(noop_diffs) if noop_diffs else 0.0

    # 有害介入率
    total_harmful = sum(r["model_result"]["harmful_intervention_count"] for r in scenario_results)
    harmful_rate = total_harmful / total_interventions if total_interventions > 0 else 0.0

    # 総報酬と平均報酬
    total_reward = sum(r["model_result"]["total_reward"] for r in scenario_results)
    avg_reward = np.mean([r["model_result"]["avg_reward"] for r in scenario_results])

    return EvalMetrics(
        stable_reach_rate=stable_reach_rate,
        avg_turns_to_stable=avg_turns_to_stable,
        intervention_count=total_interventions,
        non_intervention_count=total_non_interventions,
        noop_baseline_diff=avg_noop_diff,
        harmful_intervention_rate=harmful_rate,
        total_reward=total_reward,
        avg_reward=avg_reward,
    )


def print_eval_results(results: Dict[str, Any]) -> None:
    """評価結果を表示"""
    metrics = results["metrics"]

    print("=" * 80)
    print("決定的評価結果")
    print("=" * 80)
    print()
    print(f"安定到達率:              {metrics.stable_reach_rate:.2%}")
    print(f"平均到達ターン数:        {metrics.avg_turns_to_stable:.2f}")
    print(f"介入回数:                {metrics.intervention_count}")
    print(f"非介入回数:              {metrics.non_intervention_count}")
    print(f"no-opベースライン差分:   {metrics.noop_baseline_diff:+.4f}")
    print(f"有害介入率:              {metrics.harmful_intervention_rate:.2%}")
    print(f"総報酬:                  {metrics.total_reward:.4f}")
    print(f"平均報酬:                {metrics.avg_reward:.4f}")
    print()
    print("=" * 80)

    # シナリオごとの詳細
    print("\nシナリオ別詳細:")
    print("-" * 80)
    for scenario_result in results["scenario_results"]:
        scenario_id = scenario_result["scenario_id"]
        model_res = scenario_result["model_result"]
        noop_diff = scenario_result["noop_diff"]

        print(f"\n{scenario_id}:")
        print(f"  安定到達:     {'✓' if model_res['reached_stable'] else '✗'}")
        print(f"  到達ターン:   {model_res['turns_to_stable'] if model_res['turns_to_stable'] else 'N/A'}")
        print(f"  介入回数:     {model_res['intervention_count']}")
        print(f"  総報酬:       {model_res['total_reward']:.4f}")
        print(f"  no-op差分:    {noop_diff:+.4f}")
