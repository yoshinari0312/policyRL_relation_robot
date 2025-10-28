import os
import re
import time
import requests
from typing import List, Dict, Optional

from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from config import get_config, load_config
from utils import (
    filter_logs_by_human_count,
    call_ollama_generate_safe,
    get_common_triggers_for_others,
    format_common_triggers_message,
)

# モデルやベースURLは環境変数で切り替え可能
_CFG = get_config()
# If the runtime cwd doesn't contain a config.local.yaml, try loading the one
# located next to this module (app/config.local.yaml). This helps when tests
# are executed from the repository root.
try:
    if getattr(_CFG, "env", None) is None or getattr(_CFG.env, "debug", None) is None:
        local_path = os.path.join(os.path.dirname(__file__), "config.local.yaml")
        if os.path.exists(local_path):
            _CFG = load_config(local_path)
except Exception:
    # best-effort fallback: keep the original _CFG
    pass

cfg_bases = list(getattr(_CFG.ollama, "bases", []) or [])
cfg_default_base = getattr(_CFG.ollama, "base_url", "http://ollama_a:11434")

# 1つだけ指定の場合（従来互換）
OLLAMA_BASE = os.getenv("OLLAMA_BASE") or (cfg_bases[0] if cfg_bases else cfg_default_base)

# 人間を複数GPUに振り分けるため、カンマ区切りの複数エンドポイントも受け付け
env_bases = os.getenv("OLLAMA_BASES")
if env_bases:
    base_candidates = env_bases.split(",")
elif cfg_bases:
    base_candidates = cfg_bases
else:
    base_candidates = [OLLAMA_BASE]
_BASES = [b.strip() for b in base_candidates if b and b.strip()]

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or getattr(_CFG.ollama, "model", "gpt-oss:20b")
GEN_OPTS     = _CFG.ollama.gen_options

_HIRAGANA_RE = re.compile(r"[\u3040-\u309f]")

def _contains_hiragana(text: str) -> bool:
    return bool(_HIRAGANA_RE.search(text))

def _post_ollama(prompt: str, base: str) -> str:
    """
    Ollamaエンドポイントに対してテキスト生成をリクエストする

    Args:
        prompt: 生成用のプロンプト
        base: OllamaのベースURL（例: "http://ollama_a:11434"）

    Returns:
        生成されたテキスト
    """
    # GEN_OPTSを辞書に変換（存在する場合）
    options = dict(GEN_OPTS) if GEN_OPTS else None

    result = call_ollama_generate_safe(
        prompt=prompt,
        base_url=base,
        model=OLLAMA_MODEL,
        options=options,
        timeout=120,
        max_attempts=3,
        default=""
    )

    if not result and _CFG.env.debug:
        print(f"[_post_ollama] Failed to generate response from {base}")

    return result


def build_human_prompt(speaker: str, logs: List[Dict], topic: str = None, topic_trigger: str = None) -> str:
    """Construct a neutral conversation prompt for the given speaker.

    Args:
        speaker: 話者の名前（A, B, Cなど）
        logs: 会話履歴
        topic: 会話のトピック
        topic_trigger: トピックの元になった地雷（この地雷だけを使用）
    """
    # max_history_human個の人間発話 + その間のロボット発話を取得
    cfg = get_config()
    max_history_human = getattr(cfg.env, "max_history_human", 12) or 12
    filtered_logs = filter_logs_by_human_count(logs, max_history_human)

    history = "\n".join([f"[{entry['speaker']}] {entry['utterance']}" for entry in filtered_logs])
    if not history:
        history = "(まだ発言なし)"

    # topic_triggerが指定されている場合は、それだけを地雷として使用
    # 指定されていない場合は、従来通りconfigから全地雷を取得
    personas_cfg = getattr(cfg.env, "personas", None)
    triggers = []
    all_triggers = {}

    if topic_trigger:
        # トピックの元になった地雷だけを使用
        # 各話者がこの地雷を持っているかチェック
        if personas_cfg and isinstance(personas_cfg, dict):
            for persona_name in personas_cfg.keys():
                persona_info = personas_cfg.get(persona_name, {})
                if isinstance(persona_info, dict):
                    persona_triggers = persona_info.get("triggers", []) or []
                    # topic_triggerがこの話者の地雷リストに含まれている場合のみ追加
                    if topic_trigger in persona_triggers:
                        all_triggers[persona_name] = [topic_trigger]
                    else:
                        all_triggers[persona_name] = []

            # 現在の話者がこの地雷を持っているかチェック
            persona_info = personas_cfg.get(speaker, {})
            if isinstance(persona_info, dict):
                persona_triggers = persona_info.get("triggers", []) or []
                if topic_trigger in persona_triggers:
                    triggers = [topic_trigger]
    else:
        # 従来通り、configから全地雷を取得
        if personas_cfg:
            if isinstance(personas_cfg, dict):
                # すべての話者の地雷を取得
                for persona_name in personas_cfg.keys():
                    persona_info = personas_cfg.get(persona_name, {})
                    if isinstance(persona_info, dict):
                        persona_triggers = persona_info.get("triggers", []) or []
                        # 空文字列を除外
                        all_triggers[persona_name] = [t for t in persona_triggers if t and t.strip()]

                # 現在の話者の地雷を取得
                persona_info = personas_cfg.get(speaker, {})
                if isinstance(persona_info, dict):
                    triggers = persona_info.get("triggers", []) or []
                    triggers = [t for t in triggers if t and t.strip()]

    lines = [
        f"あなたは会話参加者の一人「{speaker}」です。実在の人のように、状況に応じて態度や言い方が微妙に変わります。\n",
        f"会話は A/B/C(/ロボット) で進みます。あなたは {speaker} として、会話履歴に続けて、1–2文で自然に応答してください。\n",
        f"分析・箇条書き・メタ説明・ラベル出力はしません。{speaker} の発話だけを出力します。\n",
    ]

    if topic:
        lines.append(f"会話のテーマ：{topic}\n")

    lines.append("会話の開始時点では、他の参加者がまだ発言していない場合もあります。その場合は話題に沿って発言してください\n")

    # 地雷がある場合のみ不機嫌トリガーの説明を追加
    if triggers:
        dont_like = ", ".join(triggers)

        # 他の参加者を取得
        other_speakers = [s for s in all_triggers.keys() if s != speaker]
        other_names = "、".join(other_speakers) if other_speakers else "他の参加者"

        # 案2: 共通地雷の場合の特別処理
        # topic_triggerが指定されていて、自分も持っている場合
        if topic_trigger and topic_trigger in triggers:
            # 誰が同じ地雷を持っているか
            who_has_same_trigger = [s for s in all_triggers.keys()
                                     if topic_trigger in all_triggers.get(s, [])]

            if len(who_has_same_trigger) == 3:  # 3人全員が持っている
                # 全員イライラ→言い合い
                other_angry = [s for s in who_has_same_trigger if s != speaker]
                angry_names = "、".join(other_angry)
                lines.append(
                    f"最初はあなた（{speaker}）も{angry_names}も全員が不機嫌な状態です。"
                    f"全員がイライラしているため、意見の違いや些細なことで言い合いになり、お互いに攻撃的な言葉遣いで八つ当たりし合います。\n"
                )
            elif len(who_has_same_trigger) == 2:  # 2人だけが持っている
                # その1人に対してのみ攻撃的、もう1人には穏やか
                other_angry = [s for s in who_has_same_trigger if s != speaker]
                angry_name = other_angry[0]  # 同じ地雷を持っているもう1人

                # 地雷を持っていない1人を特定
                calm_person = [s for s in all_triggers.keys() if s not in who_has_same_trigger]
                calm_name = calm_person[0] if calm_person else "他の参加者"

                lines.append(
                    f"最初はあなた（{speaker}）も{angry_name}も不機嫌な状態です。"
                    f"{angry_name}に対して攻撃的な言葉遣いで八つ当たりします。{calm_name}には穏やかに接します。\n"
                )
            else:
                # 自分だけが持っている地雷の場合
                lines.append(
                    f"最初はかなり不機嫌で、{other_names}など他の参加者に対して攻撃的な言葉遣いで文句を言ったり八つ当たりします。\n"
                )
        else:
            # topic_triggerが指定されていない、または自分は持っていない場合
            # 案1 & 案3: 他の人を具体的に指定 + 「八つ当たり」に変更
            lines.append(
                f"最初はかなり不機嫌で、{other_names}など他の参加者に対して攻撃的な言葉遣いで文句を言ったり八つ当たりします。\n"
            )

    # topic_triggerが他の2人に共通していて、かつ自分は持っていない場合のみ、共通地雷のメッセージを追加
    # （自分も地雷を持っている場合は、自分も不機嫌なので他人に優しくできないため除外）
    if topic_trigger and topic_trigger not in triggers:
        # 他の2人（自分以外）を取得
        other_speakers = [s for s in all_triggers.keys() if s != speaker]

        # topic_triggerが他の2人全員に共通しているかチェック
        if other_speakers and all(topic_trigger in all_triggers.get(s, []) for s in other_speakers):
            # 他の2人全員がこの地雷を持っている
            common_msg = format_common_triggers_message([topic_trigger], other_speakers)
            lines.append(f"{common_msg}\n")

    # 地雷がある場合のみ回復の方針を追加
    if triggers:
        lines.append("回復の方針：一度不機嫌になって戻るためには、ロボットを含む他者の発言に納得できたときやかなり寄り添ってくれた時などに限ります。実在の人もそうですが、簡単に不機嫌でなくなることはありません。\n")

    lines.extend([
        f"出力：{speaker} の次発話のみ（1–2文、自然な日本語）\n",
        "",
        "--- 会話履歴 ---",
        history,
        "",
        "[あなたの返答]",
    ])
    return "\n".join(lines)

def human_reply(logs: List[Dict], personas: List[str], topic: str = None, topic_trigger: str = None, num_speakers: Optional[int] = None) -> List[Dict]:
    """会話参加者それぞれの応答を生成する。

    Args:
        logs: 会話履歴
        personas: 話者のリスト
        topic: 会話のトピック
        topic_trigger: トピックの元になった地雷（この地雷だけを使用）
        num_speakers: 生成する発話数（Noneの場合は全員分、1の場合は1人間発話のみ）
    """

    replies = []
    working_logs: List[Dict] = [dict(item) for item in logs]
    max_attempts = 3
    fallback_text = "すみません、落ち着いて話を続けましょう。"

    provider = _CFG.llm.provider or os.getenv("HUMAN_LLM_PROVIDER") or "ollama"
    provider = (provider or "ollama").lower()

    identities = list(personas or [])
    if not identities:
        identities = ["A", "B", "C"]

    speakers: List[str] = []
    for idx, identity in enumerate(identities):
        name = (identity or "").split(":", 1)[0].strip()
        if not name:
            name = chr(ord("A") + idx)
        speakers.append(name)

    # 会話履歴から最後の人間発話者を取得し、次の話者から開始する
    last_human_speaker = None
    for log in reversed(logs):
        if log.get('speaker') != 'ロボット':
            last_human_speaker = log.get('speaker')
            break

    # num_speakersが指定されている場合、最後の話者の次から開始するようにローテート
    start_idx = 0
    if last_human_speaker and num_speakers is not None:
        try:
            last_idx = speakers.index(last_human_speaker)
            start_idx = (last_idx + 1) % len(speakers)
        except ValueError:
            pass

    # start_idxから開始するようにspeakersをローテート
    if start_idx > 0:
        speakers = speakers[start_idx:] + speakers[:start_idx]

    for idx, speaker in enumerate(speakers):
        # num_speakers が指定されていて、既に指定数に達したら終了
        if num_speakers is not None and len(replies) >= num_speakers:
            break
        prompt_logs = working_logs
        text = ""

        if provider == "azure":
            client, deployment = get_azure_chat_completion_client(getattr(_CFG, "llm", None), model_type="human")
            max_attempts = getattr(_CFG.llm, "max_attempts", 5) or 5
            base_backoff = getattr(_CFG.llm, "base_backoff", 0.5) or 0.5
            if client and deployment:
                persona_prompt = build_human_prompt(speaker, prompt_logs, topic=topic, topic_trigger=topic_trigger)
                messages = [
                    {
                        "role": "system",
                        "content": "あなたは会話参加者の一人として、自然で短い日本語の返答を行います。",
                    },
                    {"role": "user", "content": persona_prompt},
                ]
                for attempt in range(1, max_attempts + 1):
                    try:
                        # GPT-5の場合はreasoningパラメータを追加
                        params = build_chat_completion_params(deployment, messages, getattr(_CFG, "llm", None))
                        res = client.chat.completions.create(**params)
                        if res and getattr(res, "choices", None):
                            choice = res.choices[0]
                            message = getattr(choice, "message", None)
                            if isinstance(message, dict):
                                txt = message.get("content", "")
                            else:
                                txt = getattr(message, "content", "")
                            txt = (txt or "").strip()
                            if txt:
                                text = txt
                                break
                    except Exception as exc:
                        if getattr(_CFG.env, "debug", False):
                            print(f"[human_reply] attempt {attempt} failed:", exc)
                        if attempt < max_attempts:
                            time.sleep(base_backoff * (2 ** (attempt - 1)))
                        else:
                            if getattr(_CFG.env, "debug", False):
                                print("[human_reply] all attempts failed, falling back to local heuristic")
            else:
                if _CFG.env.debug:
                    print("[human_reply] Azure OpenAI client unavailable; falling back")
                text = ""
        else:
            base = _BASES[idx % len(_BASES)]
            for attempt in range(1, max_attempts + 1):
                try:
                    candidate = _post_ollama(build_human_prompt(speaker, prompt_logs, topic=topic, topic_trigger=topic_trigger), base=base)
                except Exception as e:
                    if _CFG.env.debug:
                        print(f"[human_reply] Ollama call error for {speaker}: {e}")
                    candidate = ""
                if _contains_hiragana(candidate):
                    text = candidate
                    break
                if attempt < max_attempts and _CFG.env.debug:
                    print(f"[human_reply] retrying for {speaker}: missing hiragana (attempt {attempt})")
                text = candidate

        if not _contains_hiragana(text):
            text = fallback_text

        if not text:
            text = fallback_text

        reply = {"speaker": speaker, "utterance": text}
        replies.append(reply)
        working_logs.append(reply)

    return replies
