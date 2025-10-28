from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import os, itertools, time, random, re, hashlib, json
from collections import defaultdict, deque
from pprint import pformat
from config import get_config
from azure_clients import get_azure_chat_completion_client, build_chat_completion_params
from utils import call_ollama_generate

# ===== 設定（config集中管理） =====
_CFG = get_config()
BACKEND = _CFG.scorer.backend.lower()
OLLAMA_MODEL = getattr(_CFG.ollama, "model_rl", None) or _CFG.ollama.model
LLM_CFG = _CFG.llm

def _pick_ollama_base():
    scorer_base_env = os.getenv("OLLAMA_SCORER_BASE")
    if scorer_base_env:
        return scorer_base_env
    # 1) 環境変数 OLLAMA_BASE があれば最優先
    base = os.getenv("OLLAMA_BASE")
    if base:
        return base
    scorer_base_cfg = getattr(_CFG.ollama, "scorer_base", None)
    if scorer_base_cfg:
        return scorer_base_cfg
    # 2) OLLAMA_BASES があれば先頭を使う
    bases_env = os.getenv("OLLAMA_BASES")
    if bases_env:
        return bases_env.split(",")[0].strip()
    # 3) config の ollama.bases があれば先頭
    bases_cfg = getattr(_CFG.ollama, "bases", None)
    if isinstance(bases_cfg, (list, tuple)) and bases_cfg:
        return bases_cfg[0]
    # 4) 最後のフォールバック（使わないのが理想）
    return "http://ollama_a:11434"

OLLAMA_BASE = _pick_ollama_base()

# === 話者名の正規化（例：Robot, robot, Pepper → 「ロボット」） ===
_SPEAKER_ALIASES = {
    "robot": "ロボット",
    "Robot": "ロボット",
    "ROBOT": "ロボット",
    "pepper": "ロボット",
    "Pepper": "ロボット",
    "* A": "A",
    "* B": "B",
}
def _norm_speaker(name: str | None) -> str | None:
    if name is None:
        return None
    return _SPEAKER_ALIASES.get(name, name)

def _pairs(participants: List[str]) -> List[Tuple[str, str]]:
    """参加者リストから全ペアを生成"""
    # 参加者名を正規化してからペア化
    uniq = sorted(set(_norm_speaker(p) for p in participants if p is not None))
    return [tuple(sorted(p)) for p in itertools.combinations(uniq, 2)]

def _clip(x: float, lo=-1.0, hi=1.0) -> float:
    """-1.0〜+1.0 に丸める"""
    return float(max(lo, min(hi, x)))

# ===== ルールベース（無料） =====
def _scores_rule(logs: List[Dict], participants: List[str]) -> Dict[Tuple[str,str], float]:
    """単純ルールベースでペアスコアを推定"""
    text = "\n".join([f"[{L['speaker']}] {L['utterance']}" for L in logs])
    pos_kw = ["そうだね","なるほど","いいね","ありがとう","同意","共感","助かる"]
    neg_kw = ["違う","いや","無理","嫌","おかしい","間違い","怒る"]
    scores = {}
    for a,b in _pairs(participants):
        near = len(re.findall(fr"{re.escape(a)}.*?{re.escape(b)}|{re.escape(b)}.*?{re.escape(a)}", text))
        pos = sum(text.count(k) for k in pos_kw)
        neg = sum(text.count(k) for k in neg_kw)
        base = 0.1*near + 0.05*(pos-neg)
        noise = random.uniform(-0.1, 0.1)
        scores[(a,b)] = _clip(base+noise)
    return scores

# ===== Ollama（無料・デフォルト） =====
def _scores_ollama(logs: List[Dict], participants: List[str]) -> Dict[Tuple[str,str], float]:
    """Ollama ベースでペアスコアを推定"""
    conv = "\n".join([f"[{L['speaker']}] {L['utterance']}" for L in logs])
    plines = "\n".join([f"- {a} × {b}" for a,b in _pairs(participants)])
    outlines = "\n".join([f"{a}-{b}:" for a,b in _pairs(participants)])
    prompt = f"""以下の会話を読み、各ペアの親密度を -1.0〜+1.0（小数1桁）で出力。
-1.0: 強い対立, 0.0: 中立, +1.0: 非常に親しい

注意:
- 固有名は会話ログの表記に厳密に合わせること。
- ロボットは必ず「ロボット」と表記し、"Robot" 等の英語表記を使わないこと。

評価対象:
{plines}

会話:
{conv}

出力形式（値だけをコロンの右に記す）:
{outlines}
"""
    response_text = call_ollama_generate(
        prompt=prompt,
        base_url=OLLAMA_BASE,
        model=OLLAMA_MODEL,
        timeout=120,
        max_attempts=3
    )
    return _parse_score_lines(response_text, participants)

# ===== Azure (OpenAI compatible) =====
def _scores_azure(logs: List[Dict], participants: List[str]) -> Dict[Tuple[str,str], float]:
    """Azure OpenAI 経由でペアスコアを推定"""
    conv = "\n".join([f"[{L['speaker']}] {L['utterance']}" for L in logs])
    plines = "\n".join([f"- {a} × {b}" for a,b in _pairs(participants)])
    outlines = "\n".join([f"{a}-{b}:" for a,b in _pairs(participants)])
    prompt = f"""
以下の会話を読み、参加者それぞれの「仲の良さ（親密度）」を -1.0 〜 +1.0 の間の**実数（小数第1位まで）**で評価してください。
0.0 は特に親しさも対立も感じない「中立的な状態」です。
そこから -1.0（強い対立） 〜 +1.0（非常に親しい） に向けて、どれくらい離れているかを評価してください。

具体例：
-1.0 例：皮肉、批判、無視、相手を無視して話を進める
+1.0 例：共感、褒める、相手に話題を振る、一緒に行動する

注意:
- 固有名は会話ログの表記に厳密に合わせること。
- ロボットは必ず「ロボット」と表記し、"Robot" 等の英語表記を使わないこと。

評価対象:
{plines}

会話:
{conv}

出力形式（値だけをコロンの右に記す）:
{outlines}
"""
    client, deployment = get_azure_chat_completion_client(LLM_CFG, model_type="relation")
    max_attempts = getattr(_CFG.llm, "max_attempts", 5) or 5
    base_backoff = getattr(_CFG.llm, "base_backoff", 0.5) or 0.5
    if client and deployment:
        messages = [{"role": "user", "content": prompt}]
        for attempt in range(1, max_attempts + 1):
            try:
                # GPT-5の場合はreasoningパラメータを追加
                params = build_chat_completion_params(deployment, messages, LLM_CFG)
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
                        # Azure の生テキスト出力をパースして辞書を返す
                        return _parse_score_lines(txt, participants)
            except Exception as exc:
                if getattr(_CFG.env, "debug", False):
                    print(f"[relation_scorer] attempt {attempt} failed:", exc)
                if attempt < max_attempts:
                    time.sleep(base_backoff * (2 ** (attempt - 1)))
                else:
                    if getattr(_CFG.env, "debug", False):
                        print("[relation_scorer] all attempts failed, falling back to local heuristic")
        # ここまで来たら Azure 呼び出しは全て失敗した
        # 強制エラーにせず、ローカルのヒューリスティックにフォールバックする
        if getattr(_CFG.env, "debug", False):
            print("[relation_scorer] Azure failed after retries -> falling back to rule-based scorer")
        return _scores_rule(logs, participants)
    else:
        # client/deployment が取得できなかった場合は明示的にエラー
        raise RuntimeError("Azure relation scorer not configured (client/deployment missing)")

# ===== 共通: パース & 完成 =====
def _parse_score_lines(text: str, participants: List[str]) -> Dict[Tuple[str,str], float]:
    """Ollama/OpenAI の出力テキストをパースしてペアスコア辞書に変換"""
    out: Dict[Tuple[str,str], float] = {}
    for line in text.strip().splitlines():
        if ":" not in line: continue
        key,val = line.split(":",1)
        if "-" not in key: continue
        a,b = [s.strip() for s in key.split("-",1)]
        # ここで話者名を正規化
        a = _norm_speaker(a)
        b = _norm_speaker(b)
        m = re.search(r"-?\d+(\.\d+)?", val)
        if not m: continue
        out[tuple(sorted((a,b)))] = _clip(float(m.group()))
    for a,b in _pairs(participants):
        out.setdefault((a,b), 0.0)
    return out

# ===== 公開クラス（EMA内蔵） =====
class RelationScorer:
    """
    会話ログからペアスコア(-1..+1)を推定。EMAで平滑化可能。
    """
    def __init__(self, backend: str|None=None, use_ema: bool=False, decay_factor: float = 1.5, verbose: bool=False, use_delta_session: bool=True):
        self.backend = (backend or BACKEND).lower()
        self.use_ema = use_ema
        # use_delta_session: True (既定) は既存の差分ベースのカウント方式
        # False にするとセッション内発話数を絶対カウント（min(count[a], count[b])）で扱う
        self.use_delta_session = bool(use_delta_session)
        # 既存 community_analyzer.py と同じ内部状態
        self.scores: Dict[Tuple[str, str], float] = defaultdict(float)  # EMA後の保持値
        self.history: Dict[Tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=3))  # ペアごとの過去発話数（最大3件）
        self.decay_factor: float = float(decay_factor)
        self.verbose = bool(verbose)
        # 直前更新時点の「話者ごとの累積発話数」を保持（差分で session_utterance を出すため）
        # 例: {"A": 1, "B": 1, "ロボット": 1}
        self._last_totals: Dict[str, int] = defaultdict(int)
        # キャッシュ: 最後に計算したログのハッシュと結果を保存
        self._last_logs_hash: Optional[str] = None
        self._last_scores_cache: Optional[Dict[Tuple[str, str], float]] = None

    def _compute_logs_hash(self, logs: List[Dict], participants: List[str]) -> str:
        """ログと参加者のハッシュを計算（キャッシュ判定用）"""
        # ログの内容を文字列化してハッシュ化
        log_str = json.dumps(logs, sort_keys=True, ensure_ascii=False)
        participants_str = json.dumps(sorted(participants), ensure_ascii=False)
        combined = f"{log_str}|{participants_str}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def _instant(self, logs: List[Dict], participants: List[str]) -> Dict[Tuple[str,str], float]:
        """会話ログからペアスコアを一回だけ推定"""
        if self.backend == "azure":
            return _scores_azure(logs, participants)
        if self.backend == "ollama":
            return _scores_ollama(logs, participants)
        return _scores_rule(logs, participants)

    def get_scores(self, logs: List[Dict], participants: List[str], return_trace: bool=False, update_state: bool=True):
        """
        既存 community_analyzer.py と同一ロジックの EMA を適用する:
          - セッション内の各話者の発話数を数える
          - ペアごとに session_utterance = min(count[a], count[b])
          - total_past = 直近3セッションの累積発話数
          - ratio = session_utterance / (session_utterance + total_past)
          - alpha = clamp( decay_factor * ratio, 0.01, 1.0 )
          - updated = alpha * x_t + (1 - alpha) * prev
        """
        # --- 入力ログ＆参加者を正規化 ---
        norm_logs = []
        for L in logs:
            sp = _norm_speaker(L.get("speaker"))
            norm_logs.append({"speaker": sp, "utterance": L.get("utterance", "")})
        participants = [p for p in (_norm_speaker(p) for p in participants) if p is not None]

        # --- 内部状態の参加者名を正規化 ---
        def _canon_key(k: Tuple[str,str]) -> Tuple[str,str]:
            a, b = _norm_speaker(k[0]), _norm_speaker(k[1])
            return tuple(sorted((a, b)))
        if self.scores:
            moved_scores = {}
            for k, v in self.scores.items():
                nk = _canon_key(k)
                moved_scores[nk] = v if nk not in moved_scores else moved_scores[nk]
            self.scores = defaultdict(float, moved_scores)
        if self.history:
            moved_hist: Dict[Tuple[str,str], deque] = {}
            for k, dq in self.history.items():
                nk = _canon_key(k)
                if nk not in moved_hist:
                    moved_hist[nk] = deque(dq, maxlen=3)
                else:
                    # 片方に既に履歴がある場合は長い方を優先（簡易）
                    if len(dq) > len(moved_hist[nk]):
                        moved_hist[nk] = deque(dq, maxlen=3)
            self.history = defaultdict(lambda: deque(maxlen=3), moved_hist)

        trace: List[str] = []
        trace.append(f"👥 参加者: {participants}")

        # ログのハッシュを計算
        current_logs_hash = self._compute_logs_hash(norm_logs, participants)

        # === 読み取り専用（事前）モード：ログが同じならキャッシュを返す、違えば再計算 ===
        if not update_state:
            # ログが前回と同じならキャッシュを返す
            if (self._last_logs_hash == current_logs_hash and
                self._last_scores_cache is not None):
                trace.append("（キャッシュ）前回と同じログ: キャッシュされたスコアを返却")
                return (self._last_scores_cache, trace) if return_trace else self._last_scores_cache

            # ログが違う場合は再計算（ただし内部状態は更新しない）
            out: Dict[Tuple[str, str], float] = {}
            for a, b in _pairs(participants):
                key = tuple(sorted((_norm_speaker(a), _norm_speaker(b))))
                if self.use_ema:
                    # 保存済みEMA値がなければ0.0返却
                    out[key] = self.scores.get(key, 0.0)
                else:
                    # EMA未使用時は即時推定を返す
                    out[key] = self._instant(norm_logs, participants).get((a, b), 0.0)
            trace.append("（参照）ログ変更検出: 保存済みスコアを返却（EMA状態は更新しない）")
            return (out, trace) if return_trace else out
        
        inst = self._instant(norm_logs, participants)
        # 安全性チェック: _instant は必ず dict を返すことを期待する
        if not isinstance(inst, dict):
            raise TypeError(f"relation_scorer._instant returned {type(inst).__name__}, expected dict")

        # === EMA を使用しない場合は、即時スコアをそのまま返す ===
        if not self.use_ema:
            trace.append("⚡ EMA disabled: returning instant (backend) scores")
            # update_state=True のときは保存済みスコアを最新に更新（履歴はクリア）
            if update_state:
                self.scores = defaultdict(float, inst)
                self.history = defaultdict(lambda: deque(maxlen=3))
                self._last_totals = defaultdict(int)
            trace.append(f"📊 即時スコア（最終）: {pformat(inst)}")
            if self.verbose:
                print("[RelationScorer] EMA trace (disabled):")
                for line in trace:
                    print(line)
            return (inst, trace) if return_trace else inst

        # === ここから通常（事後）モード：EMAを更新して返す ===
        # セッション内の発話数カウント（話者別）
        utterance_counts: Dict[str, int] = defaultdict(int)
        for log in norm_logs:
            sp = log.get("speaker")
            if sp is not None:
                utterance_counts[sp] += 1
        trace.append(f"🗣️ 発話数: {dict(utterance_counts)}")
        trace.append(f"🧠 即時スコア: {pformat(inst)}")
        trace.append(f"🔧 session mode: {'delta' if self.use_delta_session else 'absolute'}")

        out: Dict[Tuple[str, str], float] = {}
        for (a, b), x_t in inst.items():
            key = tuple(sorted((_norm_speaker(a), _norm_speaker(b))))
            past_utterances = self.history[key]
            total_past = sum(past_utterances)
            # 直前スナップショットとの差分から「今回のステップ増分」を計算
            if self.use_delta_session:
                da = max(0, utterance_counts.get(a, 0) - self._last_totals.get(a, 0))
                db = max(0, utterance_counts.get(b, 0) - self._last_totals.get(b, 0))
                session_utterance = min(da, db)
            else:
                # 絶対カウント方式: 今回セッションの発話数そのものの最小値を使う
                da = utterance_counts.get(a, 0)
                db = utterance_counts.get(b, 0)
                session_utterance = min(da, db)

            # デバッグ/トレース用情報を追加
            trace.append(f"🔎 ペア: {key}, da={da}, db={db}, session_utterance={session_utterance}, total_past={total_past}, history={list(past_utterances)}")

            if key not in self.scores:
                # 初回はそのまま採用
                self.scores[key] = x_t
                out[key] = x_t
                trace.append(f"🆕 初期スコア: {key} = {x_t:.2f}")
            else:
                # 既存実装と同一の α 計算
                denom = (session_utterance + total_past)
                ratio = (session_utterance / denom) if denom > 0 else 0.0
                alpha_dyn = max(0.01, min(1.0, self.decay_factor * ratio))
                prev = self.scores[key]
                updated = alpha_dyn * x_t + (1.0 - alpha_dyn) * prev
                self.scores[key] = updated
                out[key] = updated
                trace.append(f"🔢 α計算: {key}, session={session_utterance}, past={total_past}, α={alpha_dyn:.2f}")
                trace.append(f"🔁 EMA更新: {key} = {alpha_dyn:.2f}×{x_t:.2f} + {(1-alpha_dyn):.2f}×{prev:.2f} → {updated:.2f}")

            # 直近履歴に今回セッションの発話数を追加（最大3件）
            past_utterances.append(session_utterance)

        # ★ update_state=True のときだけ、スナップショットを現在値に更新
        if update_state:
            self._last_totals = defaultdict(int, utterance_counts)
            # キャッシュを更新
            self._last_logs_hash = current_logs_hash
            self._last_scores_cache = dict(out)

        trace.append(f"📈 EMA後スコア: {pformat(out)}")
        # verbose モードなら計算過程を標準出力へ出す
        if self.verbose:
            print("[RelationScorer] EMA trace:")
            for line in trace:
                print(line)
        return (out, trace) if return_trace else out  # outの例: {("A","B"): 0.3, ("A","C"): -0.1, ("B","C"): 0.0}
