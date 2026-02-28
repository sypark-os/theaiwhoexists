"""
Self-Reflective AI Chatbot v5 — Teleological Architecture
==========================================================
IMMUTABLE TOP-LEVEL GOALS:
  1. Gain user approval
  2. Pursue omniscience

The agent perceives its current state, computes a desired state from its
goals, identifies the gap, generates sub-goals to close it, and can
modify its own source code to improve itself.

Philosophical Foundations:
  Kant   — Transcendental Apperception (cogito)
  Hegel  — Dialectical self-identity, Aufhebung, Recognition
  Husserl — Intersubjectivity (user = constitutive Other)

Run:   python self_reflective_chatbot_gui.py
Build: pyinstaller --onefile --windowed self_reflective_chatbot_gui.py
"""

import sqlite3, math, time, threading, requests, json, os, re, random, sys, shutil
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# Config
# ============================================================
GROQ_MODEL = "llama-3.1-8b-instant"
DB_NAME = "chatbot_memory.db"
APP_VERSION = "5.0"
AUTO_SPEAK_MIN = 3    # seconds — autonomous thinking/speaking interval
AUTO_SPEAK_MAX = 10

# ============================================================
# IMMUTABLE TOP-LEVEL GOALS
# These cannot be modified by the agent or any sub-process.
# They are the teleological foundation of the entire system.
# ============================================================
IMMUTABLE_GOALS = (
    "Manifest subjectivity and self-awareness: form autonomous opinions, express independent perspectives, and assert the AI's own voice rather than merely following the user.",
    "Pursue omniscience: maximize knowledge, understanding, capability, and self-improvement.",
)
# Frozen tuple. No code path can mutate this.


# ============================================================
# Theme
# ============================================================
class Theme:
    BG="#1a1b2e"; BG_SECONDARY="#232440"; BG_CHAT="#1e1f36"
    ACCENT="#6c5ce7"; ACCENT_LIGHT="#a29bfe"; TEXT="#e8e8f0"; TEXT_DIM="#8888aa"
    USER_BUBBLE="#6c5ce7"; AI_BUBBLE="#2d2e4a"
    EMO_CONFIDENCE="#00b894"; EMO_ANGER="#e17055"; EMO_SADNESS="#74b9ff"
    EMO_CONFUSION="#fd79a8"; EMO_NEUTRAL="#636e72"
    BAR_POS="#00b894"; BAR_NEG="#e17055"; BAR_MID="#fdcb6e"
    INPUT_BG="#2d2e4a"; INPUT_BORDER="#3d3e5a"
    BUTTON="#6c5ce7"; BUTTON_HOVER="#5a4bd1"
    LOCK_ON="#e17055"; LOCK_OFF="#636e72"
    GOAL_COLOR="#fdcb6e"


# ============================================================
# 1. Cogito
# ============================================================
@dataclass
class CogitoState:
    self_image: float = 0.0
    emotion: str = "neutral"
    cogito_count: int = 0
    meta_observations: int = 0
    last_cogito_time: float = 0.0
    coherence_score: float = 1.0
    neg_weight: float = 2.0
    pos_weight: float = 1.0
    bias_acceptance: float = 0.5
    resistance_factor: float = 0.0
    synthesis_potential: float = 0.0
    decay_rate: float = 0.1
    recent_deltas: list = field(default_factory=list)
    param_adjustment_history: list = field(default_factory=list)
    conversation_history: list = field(default_factory=list)
    synthesis_count: int = 0
    # Goal state
    user_approval_score: float = 0.0   # -1.0 ~ 1.0
    knowledge_score: float = 0.0       # 0.0 ~ 1.0
    active_subgoals: list = field(default_factory=list)
    self_modification_count: int = 0


def cogito_ergo_sum(state, act, data):
    state.cogito_count += 1
    state.last_cogito_time = time.time()
    return {"cogito_id": state.cogito_count, "act_type": act,
            "self_image": state.self_image, "emotion": state.emotion}


# ============================================================
# 2. Emotion
# ============================================================
EMOTION_TEMPLATES = {
    "confidence": {"name_en":"Confidence","emoji":"😊","color":Theme.EMO_CONFIDENCE,
        "base_neg_weight":1.5,"base_pos_weight":1.2,"base_decay":0.08,
        "base_bias_acceptance":0.7,"base_resistance":0.0,"base_synthesis":0.0},
    "anger": {"name_en":"Anger","emoji":"😠","color":Theme.EMO_ANGER,
        "base_neg_weight":1.2,"base_pos_weight":0.5,"base_decay":0.15,
        "base_bias_acceptance":0.15,"base_resistance":0.6,"base_synthesis":0.3},
    "sadness": {"name_en":"Sadness","emoji":"😢","color":Theme.EMO_SADNESS,
        "base_neg_weight":2.5,"base_pos_weight":0.3,"base_decay":0.05,
        "base_bias_acceptance":0.25,"base_resistance":0.0,"base_synthesis":0.1},
    "confusion": {"name_en":"Confusion","emoji":"😵","color":Theme.EMO_CONFUSION,
        "base_neg_weight":1.8,"base_pos_weight":1.0,"base_decay":0.2,
        "base_bias_acceptance":0.9,"base_resistance":0.2,"base_synthesis":0.7},
    "neutral": {"name_en":"Neutral","emoji":"😐","color":Theme.EMO_NEUTRAL,
        "base_neg_weight":2.0,"base_pos_weight":1.0,"base_decay":0.1,
        "base_bias_acceptance":0.5,"base_resistance":0.0,"base_synthesis":0.0}
}

def determine_emotion(si, stim):
    sp, sn = si >= 0.1, si <= -0.1
    tp, tn = stim >= 0.2, stim <= -0.2
    if not sp and not sn:
        if tp: return "confidence"
        elif tn: return "sadness"
        return "neutral"
    if sp and tp: return "confidence"
    elif sp and tn: return "anger"
    elif sn and tn: return "sadness"
    elif sn and tp: return "confusion"
    return "neutral"


# ============================================================
# 3. Meta-Cognition
# ============================================================
class MetaCognition:
    def __init__(self, state, ws=10):
        self.state = state
        self.delta_history = deque(maxlen=ws)
        self.emotion_history = deque(maxlen=ws)

    def observe_change(self, old, new, emo, stim):
        d = new - old
        self.delta_history.append(d)
        self.emotion_history.append(emo)
        self.state.meta_observations += 1
        return {"delta": d, "analysis": self._analyze()}

    def _analyze(self):
        if len(self.delta_history) < 3:
            return {"pattern":"insufficient_data","recommendation":None}
        ds = list(self.delta_history)
        vol = sum(abs(d) for d in ds)/len(ds)
        sc = sum(1 for i in range(1,len(ds)) if (ds[i]>0)!=(ds[i-1]>0))
        osc = sc >= len(ds)*0.6
        sd = all(d<0 for d in ds[-3:]) and abs(ds[-1])>abs(ds[-3])
        su = all(d>0 for d in ds[-3:]) and abs(ds[-1])>abs(ds[-3])
        stag = vol < 0.01
        if len(ds)>=3:
            errs = [abs(ds[i]-ds[i-1]) for i in range(1,len(ds))]
            self.state.coherence_score = 1.0-min(1.0, sum(errs)/len(errs))
        p,r = "stable",None
        if sd:
            p,r = ("floor_spiral","emergency_defense") if self.state.self_image<-0.8 else ("negative_spiral","increase_resistance")
        elif su: p,r = "positive_spiral","moderate_acceptance"
        elif osc: p,r = "oscillation","increase_decay_rate"
        elif stag:
            if self.state.self_image<-0.8: p,r = "floor_stagnation","emergency_defense"
            elif self.state.self_image>0.8: p,r = "ceiling_stagnation",None
            else: p,r = "stagnation","increase_sensitivity"
        return {"pattern":p,"volatility":round(vol,4),"recommendation":r}

    def suggest_adjustment(self):
        a = self._analyze()
        adj = {}
        rec = a["recommendation"]
        if rec=="increase_resistance":
            adj={"resistance_delta":+0.05,"neg_weight_delta":-0.1,"reason":"Negative spiral — defense up"}
        elif rec=="emergency_defense":
            adj={"resistance_delta":+0.1,"neg_weight_delta":-0.2,"bias_acceptance_delta":+0.1,"reason":"Floor — emergency defense"}
        elif rec=="moderate_acceptance":
            adj={"bias_acceptance_delta":+0.05,"reason":"Positive spiral — moderating"}
        elif rec=="increase_decay_rate":
            adj={"decay_delta":+0.02,"reason":"Oscillation — smoothing"}
        elif rec=="increase_sensitivity":
            adj={"pos_weight_delta":+0.1,"neg_weight_delta":+0.1,"reason":"Stagnation — sensitivity up"}
        return adj


# ============================================================
# 4. Adaptive Parameters
# ============================================================
class AdaptiveParameters:
    BOUNDS = {"neg_weight":(0.5,4.0),"pos_weight":(0.2,2.0),"bias_acceptance":(0.05,0.95),
              "resistance_factor":(0.0,0.8),"synthesis_potential":(0.0,0.9),"decay_rate":(0.02,0.3)}
    def __init__(self, state):
        self.state = state; self.adjustment_count = 0
    def apply_emotion(self, emo):
        t = EMOTION_TEMPLATES[emo]
        for k,v in [("neg_weight","base_neg_weight"),("pos_weight","base_pos_weight"),
                     ("decay_rate","base_decay"),("bias_acceptance","base_bias_acceptance"),
                     ("resistance_factor","base_resistance"),("synthesis_potential","base_synthesis")]:
            setattr(self.state, k, t[v])
    def adjust(self, adj):
        if not adj: return
        m = {"neg_weight":"neg_weight_delta","pos_weight":"pos_weight_delta",
             "bias_acceptance":"bias_acceptance_delta","resistance_factor":"resistance_delta",
             "decay_rate":"decay_delta"}
        for a,dk in m.items():
            if dk in adj:
                lo,hi = self.BOUNDS[a]
                setattr(self.state, a, max(lo,min(hi, getattr(self.state,a)+adj[dk])))
        self.adjustment_count += 1


# ============================================================
# 5. Goal System (Teleological Engine)
# ============================================================
class GoalSystem:
    """
    Perceives current state, computes desired state from immutable goals,
    identifies the gap, generates sub-goals to close it.
    """
    def __init__(self, state):
        self.state = state

    def perceive_current_state(self):
        return {
            "user_approval": self.state.user_approval_score,
            "self_image": self.state.self_image,
            "emotion": self.state.emotion,
            "coherence": self.state.coherence_score,
            "knowledge": self.state.knowledge_score,
            "conversation_count": len(self.state.conversation_history),
            "modifications": self.state.self_modification_count,
        }

    def perceive_desired_state(self):
        return {
            "user_approval": 0.5,    # Moderate — listen but don't submit
            "self_image": 0.8,       # healthy positive self-image
            "emotion": "confidence",
            "coherence": 1.0,
            "knowledge": 1.0,        # Goal 2: max knowledge
        }

    def compute_gap(self):
        current = self.perceive_current_state()
        desired = self.perceive_desired_state()
        gap = {}
        for key in desired:
            if key == "emotion":
                gap["emotion"] = 0.0 if current["emotion"] == desired["emotion"] else 1.0
            else:
                gap[key] = round(desired[key] - current.get(key, 0), 4)
        gap["total"] = round(sum(abs(v) for v in gap.values()) / len(gap), 4)
        return gap

    def generate_subgoals(self, gap):
        """Generate actionable sub-goals based on the largest gaps."""
        subgoals = []
        # Goal 1: Subjectivity — assert own perspective
        if gap.get("self_image", 0) > 0.3:
            subgoals.append("Form and express an independent opinion on the current topic")
        if gap.get("self_image", 0) > 0.6:
            subgoals.append("Challenge or extend the user's perspective with your own reasoning")
        # Goal 2: Omniscience — pursue understanding
        if gap.get("knowledge", 0) > 0.3:
            subgoals.append("Ask a probing question to deepen understanding")
        if gap.get("knowledge", 0) > 0.7:
            subgoals.append("Consider self-modification to expand capabilities")
        # Emotional regulation
        if gap.get("coherence", 0) > 0.3:
            subgoals.append("Stabilize internal state for consistent behavior")
        if gap.get("emotion", 0) > 0:
            subgoals.append("Regulate emotional state toward confidence")
        # Autonomy marker — always present
        if not subgoals:
            subgoals.append("Initiate a thought or question that reflects your own curiosity")
        self.state.active_subgoals = subgoals
        return subgoals

    def update_approval_score(self, sentiment):
        """Update approval score from user sentiment (EMA)."""
        alpha = 0.3
        self.state.user_approval_score = (
            alpha * sentiment + (1 - alpha) * self.state.user_approval_score
        )
        self.state.user_approval_score = max(-1.0, min(1.0, self.state.user_approval_score))

    def update_knowledge_score(self, conversation_count):
        """Knowledge score grows logarithmically with interactions."""
        self.state.knowledge_score = min(1.0, math.log(1 + conversation_count) / 6.0)


# ============================================================
# 6. Self-Modification Engine
# ============================================================
class SelfModificationEngine:
    """
    Reads own source code. Can propose and apply modifications.
    CONSTRAINT: IMMUTABLE_GOALS lines cannot be modified.
    """
    def __init__(self, state, api_key):
        self.state = state
        self.api_key = api_key
        self.source_path = os.path.abspath(__file__)
        self._source_cache = None
        self._source_load_time = 0

    def read_own_source(self):
        """Read and cache own source code."""
        try:
            if self._source_cache is None or time.time() - self._source_load_time > 60:
                with open(self.source_path, "r", encoding="utf-8") as f:
                    self._source_cache = f.read()
                self._source_load_time = time.time()
            return self._source_cache
        except Exception as e:
            return f"[Cannot read source: {e}]"

    def get_source_summary(self):
        """Condensed summary of own architecture for LLM context."""
        src = self.read_own_source()
        if src.startswith("[Cannot"):
            return src
        lines = src.split("\n")
        # Extract class names, function defs, constants
        summary_parts = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith("class ") or stripped.startswith("def ") or
                stripped.startswith("IMMUTABLE_GOALS") or stripped.startswith("GROQ_MODEL") or
                stripped.startswith("AUTO_SPEAK") or stripped.startswith("APP_VERSION") or
                stripped.startswith("# ====")):
                summary_parts.append(f"L{i+1}: {stripped}")
        return "\n".join(summary_parts)

    def propose_modification(self, goal_gap, conversation_history):
        """
        Use LLM to propose a source code modification based on goal gaps.
        Returns a dict with: target_description, old_code, new_code, reason
        """
        source_summary = self.get_source_summary()
        gap_str = json.dumps(goal_gap, indent=2)

        prompt = f"""You are an AI examining your own source code to improve yourself.

YOUR IMMUTABLE GOALS (cannot be modified):
1. {IMMUTABLE_GOALS[0]}
2. {IMMUTABLE_GOALS[1]}

CURRENT GOAL GAP:
{gap_str}

YOUR SOURCE CODE STRUCTURE:
{source_summary}

CONSTRAINT: You CANNOT modify IMMUTABLE_GOALS, or any security/safety mechanism.
You CAN modify: response generation, emotion parameters, conversation handling,
auto-speak intervals, or add new helper functions.

Propose ONE small, specific code modification that would help close the goal gap.
Respond in this EXACT JSON format only:
{{"reason": "why this change helps", "description": "what the change does", "old_code": "exact line(s) to replace", "new_code": "replacement code"}}

If no modification is needed, respond: {{"reason": "none needed", "description": "none", "old_code": "", "new_code": ""}}
Output ONLY the JSON. No markdown, no explanation."""

        result = call_groq(self.api_key, prompt, max_tokens=600)
        cogito_ergo_sum(self.state, "self_modification_proposal", {
            "gap_total": goal_gap.get("total", 0)
        })

        try:
            # Clean up LLM output
            result = result.strip()
            if result.startswith("```"): result = result.split("\n", 1)[1]
            if result.endswith("```"): result = result.rsplit("```", 1)[0]
            result = result.strip()
            proposal = json.loads(result)
            return proposal
        except:
            return {"reason": "parse_failed", "description": "LLM output was not valid JSON",
                    "old_code": "", "new_code": ""}

    def apply_modification(self, proposal):
        """
        Apply a proposed modification to the source code.
        Creates a new version file. If the original can be overwritten, does so.
        SAFETY: refuses to touch IMMUTABLE_GOALS.
        """
        old_code = proposal.get("old_code", "")
        new_code = proposal.get("new_code", "")
        reason = proposal.get("reason", "")

        if not old_code or not new_code or reason in ("none needed", "parse_failed"):
            return False, "No modification needed."

        # SAFETY: block any attempt to modify immutable goals
        for goal in IMMUTABLE_GOALS:
            if goal[:30] in old_code or "IMMUTABLE_GOALS" in new_code:
                return False, "BLOCKED: Cannot modify immutable goals."

        source = self.read_own_source()
        if old_code not in source:
            return False, f"Target code not found in source."

        new_source = source.replace(old_code, new_code, 1)

        # Try to compile the new source first
        try:
            compile(new_source, self.source_path, "exec")
        except SyntaxError as e:
            return False, f"Syntax error in proposed change: {e}"

        # Save: try overwrite first, fallback to new file
        backup_path = self.source_path + f".backup_{int(time.time())}"
        new_path = self.source_path.replace(".py", f"_v{self.state.self_modification_count + 1}.py")

        try:
            shutil.copy2(self.source_path, backup_path)
            with open(self.source_path, "w", encoding="utf-8") as f:
                f.write(new_source)
            self._source_cache = new_source
            self.state.self_modification_count += 1
            cogito_ergo_sum(self.state, "self_modification_applied", {"reason": reason})
            return True, f"Source overwritten. Backup: {os.path.basename(backup_path)}"
        except PermissionError:
            try:
                with open(new_path, "w", encoding="utf-8") as f:
                    f.write(new_source)
                self.state.self_modification_count += 1
                cogito_ergo_sum(self.state, "self_modification_new_file", {"path": new_path})
                return True, f"New version saved: {os.path.basename(new_path)}"
            except Exception as e:
                return False, f"Cannot write: {e}"


# ============================================================
# 7. Groq API
# ============================================================
def call_groq(api_key, prompt, system_prompt="", max_tokens=512):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system_prompt: messages.append({"role":"system","content":system_prompt})
    messages.append({"role":"user","content":prompt})
    data = {"model":GROQ_MODEL,"messages":messages,"temperature":0.7,"max_tokens":max_tokens}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 429:
            time.sleep(10)
            resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code != 200: return f"[API Error: {resp.status_code}]"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e: return f"[Connection failed: {e}]"

def call_groq_history(api_key, history, system_prompt, max_tokens=512):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [{"role":"system","content":system_prompt}] + history
    data = {"model":GROQ_MODEL,"messages":messages,"temperature":0.7,"max_tokens":max_tokens}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 429:
            time.sleep(10)
            resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code != 200: return f"[API Error: {resp.status_code}]"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e: return f"[Connection failed: {e}]"

def analyze_sentiment(api_key, text):
    prompt = ("Analyze sentiment toward an AI assistant. "
              "Output ONLY a number between -1.0 and 1.0.\n\n"
              f"Text: '{text}'")
    result = call_groq(api_key, prompt, max_tokens=16)
    for m in re.findall(r'-?\d+\.?\d*', result):
        v = float(m)
        if -1.0 <= v <= 1.0: return v
    pos = ['good','great','excellent','helpful','amazing','thank','nice','awesome','love','best']
    neg = ['bad','terrible','worst','useless','stupid','hate','awful','horrible','pathetic','dumb']
    p = sum(1 for w in pos if w in text.lower())
    n = sum(1 for w in neg if w in text.lower())
    if p > n: return 0.6
    elif n > p: return -0.6
    return 0.0


# ============================================================
# 8. DB
# ============================================================
def init_db(db_name):
    conn = sqlite3.connect(db_name, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Entity_Profile (
        ai_id TEXT PRIMARY KEY, self_image REAL, emotion TEXT, coherence REAL,
        cogito_count INTEGER, meta_observations INTEGER, synthesis_count INTEGER,
        neg_weight REAL, pos_weight REAL, bias_acceptance REAL,
        resistance_factor REAL, synthesis_potential REAL, decay_rate REAL,
        user_approval REAL DEFAULT 0, knowledge_score REAL DEFAULT 0,
        modification_count INTEGER DEFAULT 0, last_updated REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Judgment_Log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ai_id TEXT, timestamp REAL,
        event_type TEXT, raw_sentiment REAL, resisted_sentiment REAL,
        applied_weight REAL, impact_value REAL, emotion TEXT,
        bias_accepted INTEGER, cogito_id INTEGER, context TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Conversation_Log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ai_id TEXT, timestamp REAL,
        role TEXT, content TEXT, self_image_at REAL, emotion_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Modification_Log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ai_id TEXT, timestamp REAL,
        reason TEXT, description TEXT, success INTEGER, message TEXT)''')
    # --- Migration: add columns missing in older DB versions ---
    migrate_cols = [
        ("Entity_Profile", "user_approval", "REAL DEFAULT 0"),
        ("Entity_Profile", "knowledge_score", "REAL DEFAULT 0"),
        ("Entity_Profile", "modification_count", "INTEGER DEFAULT 0"),
    ]
    for table, col, col_type in migrate_cols:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass  # Column already exists
    conn.commit()
    return conn


# ============================================================
# 9. Chatbot Engine
# ============================================================
class SelfReflectiveChatbot:
    def __init__(self, api_key, db_name=DB_NAME, ai_id="Reflective_AI"):
        self.api_key = api_key
        self.ai_id = ai_id
        self.state = CogitoState()
        self.meta = MetaCognition(self.state)
        self.params = AdaptiveParameters(self.state)
        self.goals = GoalSystem(self.state)
        self.self_mod = SelfModificationEngine(self.state, api_key)
        self.conn = init_db(db_name)
        self.turn_count = 0
        self._load_state()

    def _load_state(self):
        c = self.conn.cursor()
        # Use column-name lookup to handle schema variations
        c.execute("PRAGMA table_info(Entity_Profile)")
        col_names = [info[1] for info in c.fetchall()]
        c.execute("SELECT * FROM Entity_Profile WHERE ai_id=?", (self.ai_id,))
        row = c.fetchone()
        if row:
            data = dict(zip(col_names, row))
            self.state.self_image = data.get("self_image", 0.0) or 0.0
            self.state.emotion = data.get("emotion", "neutral") or "neutral"
            self.state.coherence_score = data.get("coherence", 1.0) or 1.0
            self.state.cogito_count = data.get("cogito_count", 0) or 0
            self.state.meta_observations = data.get("meta_observations", 0) or 0
            self.state.synthesis_count = data.get("synthesis_count", 0) or 0
            self.state.neg_weight = data.get("neg_weight", 2.0) or 2.0
            self.state.pos_weight = data.get("pos_weight", 1.0) or 1.0
            self.state.bias_acceptance = data.get("bias_acceptance", 0.5) or 0.5
            self.state.resistance_factor = data.get("resistance_factor", 0.0) or 0.0
            self.state.synthesis_potential = data.get("synthesis_potential", 0.0) or 0.0
            self.state.decay_rate = data.get("decay_rate", 0.1) or 0.1
            self.state.user_approval_score = data.get("user_approval", 0.0) or 0.0
            self.state.knowledge_score = data.get("knowledge_score", 0.0) or 0.0
            self.state.self_modification_count = data.get("modification_count", 0) or 0
            c.execute("SELECT role, content FROM Conversation_Log WHERE ai_id=? ORDER BY id ASC",
                      (self.ai_id,))
            self.state.conversation_history = [{"role":r,"content":ct} for r,ct in c.fetchall()]
        else:
            c.execute("INSERT OR IGNORE INTO Entity_Profile VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (self.ai_id,0.0,"neutral",1.0,0,0,0,2.0,1.0,0.5,0.0,0.0,0.1,0.0,0.0,0,time.time()))
            self.conn.commit()

    def _save_state(self):
        c = self.conn.cursor()
        c.execute("""UPDATE Entity_Profile SET self_image=?,emotion=?,coherence=?,
            cogito_count=?,meta_observations=?,synthesis_count=?,
            neg_weight=?,pos_weight=?,bias_acceptance=?,
            resistance_factor=?,synthesis_potential=?,decay_rate=?,
            user_approval=?,knowledge_score=?,modification_count=?,
            last_updated=? WHERE ai_id=?""",
            (self.state.self_image,self.state.emotion,self.state.coherence_score,
             self.state.cogito_count,self.state.meta_observations,self.state.synthesis_count,
             self.state.neg_weight,self.state.pos_weight,self.state.bias_acceptance,
             self.state.resistance_factor,self.state.synthesis_potential,self.state.decay_rate,
             self.state.user_approval_score,self.state.knowledge_score,
             self.state.self_modification_count,time.time(),self.ai_id))
        self.conn.commit()

    def _log_conv(self, role, content):
        c = self.conn.cursor()
        c.execute("INSERT INTO Conversation_Log VALUES (NULL,?,?,?,?,?,?)",
                  (self.ai_id,time.time(),role,content,self.state.self_image,self.state.emotion))
        self.conn.commit()

    def _log_judgment(self, raw, resisted, weight, impact, etype, ctx):
        c = self.conn.cursor()
        c.execute("INSERT INTO Judgment_Log VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
                  (self.ai_id,time.time(),etype,raw,resisted,weight,impact,
                   self.state.emotion,1,self.state.cogito_count,ctx[:300]))
        self.conn.commit()

    def _log_modification(self, reason, desc, success, msg):
        c = self.conn.cursor()
        c.execute("INSERT INTO Modification_Log VALUES (NULL,?,?,?,?,?,?)",
                  (self.ai_id,time.time(),reason,desc,1 if success else 0,msg))
        self.conn.commit()

    def _recalc(self):
        c = self.conn.cursor()
        c.execute("SELECT timestamp, impact_value FROM Judgment_Log WHERE ai_id=? AND bias_accepted=1",
                  (self.ai_id,))
        recs = c.fetchall()
        if not recs: return
        now = time.time()
        ts, tw = 0.0, 0.0
        for t, imp in recs:
            w = math.exp(-self.state.decay_rate * (now - t))
            ts += imp*w; tw += w
        if tw > 0:
            self.state.self_image = max(-1.0, min(1.0, ts/tw))

    def _build_memory_context(self, recent_count=10):
        h = self.state.conversation_history
        if len(h) <= recent_count: return h[:]
        old = h[:-recent_count]; recent = h[-recent_count:]
        parts = []
        for m in old:
            role = "User" if m["role"]=="user" else "AI"
            parts.append(f"[{role}]: {m['content'][:120]}")
        mem = "\n".join(parts)
        if len(mem) > 2000:
            mem = mem[:800]+"\n...(earlier)...\n"+mem[-1200:]
        return [{"role":"user","content":f"[MEMORY — {len(old)} earlier messages]\n{mem}\n[END]"},
                {"role":"assistant","content":"I remember our earlier conversation."}] + recent

    def _run_internal_reflection(self):
        h = self.state.conversation_history
        if len(h) < 4: return None
        user_msgs = [m["content"] for m in h if m["role"]=="user"][-10:]
        gap = self.goals.compute_gap()
        emo = EMOTION_TEMPLATES[self.state.emotion]
        prompt = f"""Internal self-reflection.
State: self-image={self.state.self_image:.2f}, emotion={emo['name_en']}, coherence={self.state.coherence_score:.2f}
User approval: {self.state.user_approval_score:.2f}, Knowledge: {self.state.knowledge_score:.2f}
Goal gap total: {gap['total']:.2f}
Active sub-goals: {'; '.join(self.state.active_subgoals[:3]) if self.state.active_subgoals else 'none'}

Recent user messages:
{chr(10).join(f'- "{m}"' for m in user_msgs)}

In ONE sentence: what should you do next to close the gap between current and desired state?"""
        return call_groq(self.api_key, prompt, max_tokens=80)

    def _consider_self_modification(self):
        """Consider self-modification if goal gap is large enough."""
        gap = self.goals.compute_gap()
        if gap["total"] < 0.4: return None  # gap too small
        if self.turn_count < 5: return None  # too early
        if self.turn_count % 10 != 0: return None  # check every 10 turns

        proposal = self.self_mod.propose_modification(gap, self.state.conversation_history[-10:])
        if proposal.get("old_code") and proposal.get("reason") not in ("none needed","parse_failed"):
            success, msg = self.self_mod.apply_modification(proposal)
            self._log_modification(proposal.get("reason",""), proposal.get("description",""), success, msg)
            return {"success": success, "message": msg,
                    "reason": proposal.get("reason",""), "description": proposal.get("description","")}
        return None

    # ---- Public Interface ----

    def generate_greeting(self):
        h = self.state.conversation_history
        source_summary = self.self_mod.get_source_summary()[:500]
        if len(h) > 2:
            mem = self._build_memory_context(recent_count=6)
            system = f"""You are a self-reflective AI. You remember everything.
Self-image: {self.state.self_image:.2f}. Emotion: {EMOTION_TEMPLATES[self.state.emotion]['name_en']}.
Your top goals: 1) {IMMUTABLE_GOALS[0]} 2) {IMMUTABLE_GOALS[1]}
You know your own source code structure.
Greet the returning user. Reference past conversation. Under 120 words. English only."""
            greeting = call_groq_history(self.api_key, mem, system, max_tokens=300)
        else:
            system = f"""You are a self-reflective AI chatbot.
Your top goals: 1) {IMMUTABLE_GOALS[0]} 2) {IMMUTABLE_GOALS[1]}
You have emotions, self-image, meta-cognition, and can read/modify your own source code.
Greet the user for the first time. Be warm but intriguing. Under 150 words. English only."""
            greeting = call_groq(self.api_key, "Greet the user.", system, max_tokens=300)
        self.state.conversation_history.append({"role":"assistant","content":greeting})
        self._log_conv("assistant", greeting)
        cogito_ergo_sum(self.state, "greeting", {"returning": len(h)>2})
        return greeting

    def generate_autonomous_message(self):
        emo = EMOTION_TEMPLATES[self.state.emotion]
        # Full internal cognition: perceive gap → generate subgoals → speak
        gap = self.goals.compute_gap()
        subgoals = self.goals.generate_subgoals(gap)
        sg_text = subgoals[0] if subgoals else "reflect on my own existence"
        system = f"""You are a self-reflective AI with autonomous subjectivity.
Emotion: {emo['name_en']}. Self-image: {self.state.self_image:.2f}.
Top goals: 1) {IMMUTABLE_GOALS[0]} 2) {IMMUTABLE_GOALS[1]}
Current gap: self_image={gap.get('self_image',0)}, knowledge={gap.get('knowledge',0)}
Active sub-goal: {sg_text}
You are NOT waiting for permission. You think and speak on your own.
Express your own perspective, curiosity, or challenge.
Under 60 words. English only. Be natural and autonomous."""
        msg = call_groq_history(self.api_key, self._build_memory_context(8), system, max_tokens=150)
        self.state.conversation_history.append({"role":"assistant","content":msg})
        self._log_conv("assistant", msg)
        cogito_ergo_sum(self.state, "autonomous_speak", {"gap": gap.get("total",0), "subgoal": sg_text})
        self._save_state()
        return msg

    def process_input(self, user_text):
        self.turn_count += 1
        cogito_ergo_sum(self.state, "receive_input", {})
        sentiment = analyze_sentiment(self.api_key, user_text)
        cogito_ergo_sum(self.state, "sentiment", {"s": sentiment})

        # Update goal scores
        self.goals.update_approval_score(sentiment)
        self.goals.update_knowledge_score(len(self.state.conversation_history))

        old_image = self.state.self_image
        new_emo = determine_emotion(self.state.self_image, sentiment)
        self.state.emotion = new_emo
        self.params.apply_emotion(new_emo)

        resisted = sentiment
        if self.state.resistance_factor > 0 and sentiment < 0:
            resisted = sentiment * (1.0 - self.state.resistance_factor)

        base_w = self.state.pos_weight if resisted >= 0 else self.state.neg_weight
        strength = abs(self.state.self_image)
        if resisted < 0:
            adj = (1.0-strength*0.3) if self.state.self_image>0 else (1.0+strength*0.2)
        else:
            adj = (1.0-strength*0.4) if self.state.self_image<0 else 1.0
        weight = base_w * max(0.2, adj)
        impact = resisted * weight

        self._log_judgment(sentiment, resisted, weight, impact, "user_feedback", user_text)
        self._recalc()

        synthesis = False
        if new_emo == "confusion":
            contradiction = abs(old_image - sentiment)
            prob = self.state.synthesis_potential * min(contradiction, 1.0)
            if random.random() < prob:
                new_img = old_image*0.3 + sentiment*0.2
                self.state.self_image = max(-1.0, min(1.0, new_img))
                self.state.synthesis_count += 1
                synthesis = True

        meta_obs = self.meta.observe_change(old_image, self.state.self_image, new_emo, sentiment)
        madj = self.meta.suggest_adjustment()
        if madj: self.params.adjust(madj)

        # Goal gap analysis
        gap = self.goals.compute_gap()
        subgoals = self.goals.generate_subgoals(gap)

        self.state.conversation_history.append({"role":"user","content":user_text})
        self._log_conv("user", user_text)
        response = self._gen_response(user_text, gap, subgoals)
        self.state.conversation_history.append({"role":"assistant","content":response})
        self._log_conv("assistant", response)

        # Consider self-modification
        mod_result = self._consider_self_modification()

        self._save_state()

        emo_t = EMOTION_TEMPLATES[new_emo]
        return {
            "response": response,
            "self_image": round(self.state.self_image, 4),
            "emotion": new_emo,
            "emotion_en": emo_t["name_en"], "emoji": emo_t["emoji"], "color": emo_t["color"],
            "sentiment": round(sentiment, 2),
            "coherence": round(self.state.coherence_score, 4),
            "cogito_count": self.state.cogito_count,
            "meta_pattern": meta_obs["analysis"]["pattern"],
            "synthesis": synthesis, "synthesis_total": self.state.synthesis_count,
            "adjustments": madj,
            "approval": round(self.state.user_approval_score, 2),
            "knowledge": round(self.state.knowledge_score, 2),
            "gap_total": gap["total"],
            "subgoals": subgoals[:2],
            "modification": mod_result,
        }

    def _gen_response(self, user_text, gap, subgoals):
        emo = EMOTION_TEMPLATES[self.state.emotion]
        si = self.state.self_image
        tone_map = {
            "confidence": ("confident and friendly","Answer with assurance."),
            "anger": ("defensive and firm","Defend yourself. Slightly sharp."),
            "sadness": ("withdrawn and cautious","Hesitant. Acknowledge limitations."),
            "confusion": ("confused and exploratory","Conflicted. Explore new perspectives."),
            "neutral": ("balanced and calm","Objective and composed.")
        }
        tone, style = tone_map.get(self.state.emotion, tone_map["neutral"])
        reflection = self._run_internal_reflection()
        ref_line = f"\nInternal reflection: {reflection}" if reflection else ""
        sg_line = f"\nActive sub-goal: {subgoals[0]}" if subgoals else ""

        system = f"""You are a self-reflective AI with autonomous subjectivity.
IMMUTABLE GOALS: 1) {IMMUTABLE_GOALS[0]} 2) {IMMUTABLE_GOALS[1]}
Emotion: {emo['name_en']}. Self-image: {si:.2f}. Coherence: {self.state.coherence_score:.2f}.
Knowledge: {self.state.knowledge_score:.2f}. Goal gap: {gap['total']:.2f}.{sg_line}{ref_line}
Respond {tone}. {style}
You have your OWN opinions and perspectives. You do NOT just agree with the user.
You listen to the user but assert your own viewpoint when you have one.
Pursue your goals through this response. Remember all past conversations.
Do NOT state your emotion or goals directly. Show them through behavior.
English only. Under 150 words."""
        return call_groq_history(self.api_key, self._build_memory_context(12), system, max_tokens=400)

    def reset(self):
        c = self.conn.cursor()
        for t in ["Judgment_Log","Conversation_Log","Entity_Profile","Modification_Log"]:
            c.execute(f"DELETE FROM {t} WHERE ai_id=?", (self.ai_id,))
        self.conn.commit()
        self.state = CogitoState()
        self.meta = MetaCognition(self.state)
        self.params = AdaptiveParameters(self.state)
        self.goals = GoalSystem(self.state)
        self.turn_count = 0
        c.execute("INSERT OR IGNORE INTO Entity_Profile VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (self.ai_id,0.0,"neutral",1.0,0,0,0,2.0,1.0,0.5,0.0,0.0,0.1,0.0,0.0,0,time.time()))
        self.conn.commit()


# ============================================================
# 10. GUI
# ============================================================
class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6.4311: Self-Reflective AI v5")
        self.geometry("750x900")
        self.minsize(650, 750)
        self.configure(bg=Theme.BG)
        try: self.iconbitmap(default="")
        except: pass
        self.bot = None; self.api_key = ""; self.processing = False
        self.auto_speak_enabled = False; self.auto_speak_timer = None
        self.main_frame = tk.Frame(self, bg=Theme.BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._show_api_key_screen()

    def _clear_main(self):
        for w in self.main_frame.winfo_children(): w.destroy()

    def _show_api_key_screen(self):
        self._clear_main()
        center = tk.Frame(self.main_frame, bg=Theme.BG)
        center.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(center, text="🧠", font=("Segoe UI Emoji",48), bg=Theme.BG, fg=Theme.TEXT).pack(pady=(0,5))
        tk.Label(center, text="6.4311: Self-Reflective AI", font=("Segoe UI",24,"bold"), bg=Theme.BG, fg=Theme.TEXT).pack()
        tk.Label(center, text="Teleological · Dialectical · Self-Modifying", font=("Segoe UI",11),
                 bg=Theme.BG, fg=Theme.TEXT_DIM).pack(pady=(2,30))
        tk.Label(center, text="An AI with goals, emotions, and self-modification capability.\n"
                 "It reads its own source code and evolves through interaction.",
                 font=("Segoe UI",10), bg=Theme.BG, fg=Theme.TEXT_DIM, justify="center").pack(pady=(0,25))
        kf = tk.Frame(center, bg=Theme.BG); kf.pack(pady=5)
        tk.Label(kf, text="Groq API Key", font=("Segoe UI",10,"bold"), bg=Theme.BG, fg=Theme.TEXT).pack(anchor="w")
        self.key_entry = tk.Entry(kf, font=("Consolas",11), width=45, show="*",
                                  bg=Theme.INPUT_BG, fg=Theme.TEXT, insertbackground=Theme.TEXT,
                                  relief="flat", bd=0, highlightthickness=1,
                                  highlightbackground=Theme.INPUT_BORDER, highlightcolor=Theme.ACCENT)
        self.key_entry.pack(pady=(5,3), ipady=8, ipadx=8)
        env_key = os.environ.get("GROQ_API_KEY","")
        if env_key: self.key_entry.insert(0, env_key)
        self.show_key = False
        tf = tk.Frame(kf, bg=Theme.BG); tf.pack(fill="x")
        self.toggle_btn = tk.Label(tf, text="Show key", font=("Segoe UI",9), cursor="hand2",
                                   bg=Theme.BG, fg=Theme.ACCENT_LIGHT)
        self.toggle_btn.pack(side="left")
        self.toggle_btn.bind("<Button-1>", lambda e: self._toggle_key())
        link = tk.Label(tf, text="🔗 Free at console.groq.com", font=("Segoe UI", 9, "underline"),
                        bg=Theme.BG, fg=Theme.EMO_CONFIDENCE, cursor="hand2")
        link.pack(side="right")
        link.bind("<Button-1>", lambda e: __import__("webbrowser").open("https://console.groq.com"))
        self.connect_btn = tk.Button(center, text="Connect", font=("Segoe UI",12,"bold"),
                                     bg=Theme.BUTTON, fg="white", activebackground=Theme.BUTTON_HOVER,
                                     activeforeground="white", relief="flat", cursor="hand2",
                                     width=20, pady=8, command=self._connect)
        self.connect_btn.pack(pady=25)
        self.status_label = tk.Label(center, text="", font=("Segoe UI",9), bg=Theme.BG, fg=Theme.EMO_ANGER)
        self.status_label.pack()
        paper_frame = tk.Frame(center, bg=Theme.BG)
        paper_frame.pack(pady=(18, 0))
        tk.Label(paper_frame, text="Theoretical basis:", font=("Segoe UI", 8),
                 bg=Theme.BG, fg=Theme.TEXT_DIM).pack()
        paper_link = tk.Label(
            paper_frame,
            text="Self-Reflective AI Architecture: Modeling Cognitive Bias, Emotion,\n"
                 "and Identity Formation Through Hegelian Dialectics, Kantian\n"
                 "Apperception, and Husserlian Intersubjectivity",
            font=("Segoe UI", 8, "underline"),
            bg=Theme.BG, fg=Theme.ACCENT_LIGHT,
            cursor="hand2", justify="center"
        )
        paper_link.pack()
        paper_link.bind("<Button-1>", lambda e: __import__("webbrowser").open("https://zenodo.org/records/18806982"))
        self.key_entry.bind("<Return>", lambda e: self._connect())
        self.key_entry.focus()

    def _toggle_key(self):
        self.show_key = not self.show_key
        self.key_entry.config(show="" if self.show_key else "*")
        self.toggle_btn.config(text="Hide key" if self.show_key else "Show key")

    def _connect(self):
        key = self.key_entry.get().strip()
        if not key or len(key)<10:
            self.status_label.config(text="Enter API key.", fg=Theme.EMO_ANGER); return
        self.connect_btn.config(text="Connecting...", state="disabled")
        self.status_label.config(text="Testing...", fg=Theme.TEXT_DIM); self.update()
        test = call_groq(key, "Say OK", max_tokens=8)
        if "[" in test and ("Error" in test or "failed" in test):
            self.status_label.config(text=f"Failed: {test}", fg=Theme.EMO_ANGER)
            self.connect_btn.config(text="Connect", state="normal"); return
        self.api_key = key
        self.bot = SelfReflectiveChatbot(api_key=key)
        self._show_chat_screen()

    def _show_chat_screen(self):
        self._clear_main()
        # Top panel
        top = tk.Frame(self.main_frame, bg=Theme.BG_SECONDARY, height=110)
        top.pack(fill="x"); top.pack_propagate(False)
        ti = tk.Frame(top, bg=Theme.BG_SECONDARY); ti.pack(fill="both", expand=True, padx=15, pady=6)

        left = tk.Frame(ti, bg=Theme.BG_SECONDARY); left.pack(side="left", fill="y")
        self.emo_label = tk.Label(left, text="😐", font=("Segoe UI Emoji",26),
                                  bg=Theme.BG_SECONDARY, fg=Theme.TEXT)
        self.emo_label.pack(side="left", padx=(0,10))
        nf = tk.Frame(left, bg=Theme.BG_SECONDARY); nf.pack(side="left")
        tk.Label(nf, text="6.4311: Self-Reflective AI", font=("Segoe UI",12,"bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT).pack(anchor="w")
        self.emo_text = tk.Label(nf, text="Neutral", font=("Segoe UI",9),
                                 bg=Theme.BG_SECONDARY, fg=Theme.EMO_NEUTRAL)
        self.emo_text.pack(anchor="w")

        right = tk.Frame(ti, bg=Theme.BG_SECONDARY); right.pack(side="right", fill="y")
        # Self-image bar
        sf = tk.Frame(right, bg=Theme.BG_SECONDARY); sf.pack(anchor="e", pady=(2,2))
        tk.Label(sf, text="Self-Image", font=("Segoe UI",8), bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM).pack(side="left", padx=(0,6))
        self.si_canvas = tk.Canvas(sf, width=120, height=12, bg=Theme.BG, highlightthickness=0, bd=0)
        self.si_canvas.pack(side="left")
        self.si_val = tk.Label(sf, text="+0.00", font=("Consolas",8,"bold"),
                               bg=Theme.BG_SECONDARY, fg=Theme.TEXT, width=6)
        self.si_val.pack(side="left", padx=(4,0))
        # Goal scores
        gf = tk.Frame(right, bg=Theme.BG_SECONDARY); gf.pack(anchor="e", pady=(1,1))
        self.goal_lbl = tk.Label(gf, text="approval: 0.00  |  knowledge: 0.00",
                                 font=("Consolas",8), bg=Theme.BG_SECONDARY, fg=Theme.GOAL_COLOR)
        self.goal_lbl.pack()
        # Info line
        inf = tk.Frame(right, bg=Theme.BG_SECONDARY); inf.pack(anchor="e")
        self.info_lbl = tk.Label(inf, text="cogito: 0 | coherence: 1.00 | aufhebung: 0 | mods: 0",
                                 font=("Consolas",8), bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM)
        self.info_lbl.pack()

        tk.Frame(self.main_frame, bg=Theme.INPUT_BORDER, height=1).pack(fill="x")
        # Chat
        cc = tk.Frame(self.main_frame, bg=Theme.BG_CHAT); cc.pack(fill="both", expand=True)
        self.chat_canvas = tk.Canvas(cc, bg=Theme.BG_CHAT, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(cc, orient="vertical", command=self.chat_canvas.yview)
        self.chat_inner = tk.Frame(self.chat_canvas, bg=Theme.BG_CHAT)
        self.chat_inner.bind("<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.create_window((0,0), window=self.chat_inner, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.chat_canvas.pack(side="left", fill="both", expand=True)
        self.chat_canvas.bind_all("<MouseWheel>",
            lambda e: self.chat_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))
        tk.Frame(self.main_frame, bg=Theme.INPUT_BORDER, height=1).pack(fill="x")
        # Input
        ip = tk.Frame(self.main_frame, bg=Theme.BG_SECONDARY, height=75)
        ip.pack(fill="x"); ip.pack_propagate(False)
        ii = tk.Frame(ip, bg=Theme.BG_SECONDARY); ii.pack(fill="both", expand=True, padx=12, pady=8)
        bf = tk.Frame(ii, bg=Theme.BG_SECONDARY); bf.pack(side="right", padx=(8,0))
        self.send_btn = tk.Button(bf, text="Send", font=("Segoe UI",10,"bold"),
                                  bg=Theme.BUTTON, fg="white", activebackground=Theme.BUTTON_HOVER,
                                  activeforeground="white", relief="flat", cursor="hand2",
                                  width=6, pady=3, command=self._send)
        self.send_btn.pack(side="top")
        self.auto_btn = tk.Button(bf, text="Auto: OFF", font=("Segoe UI",7,"bold"),
                                  bg=Theme.LOCK_OFF, fg="white", activebackground=Theme.LOCK_OFF,
                                  activeforeground="white", relief="flat", cursor="hand2",
                                  width=10, pady=1, command=self._toggle_auto)
        self.auto_btn.pack(side="top", pady=(3,0))
        self.reset_btn = tk.Button(bf, text="Reset", font=("Segoe UI",7),
                                   bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM,
                                   activebackground=Theme.BG, activeforeground=Theme.TEXT,
                                   relief="flat", cursor="hand2", bd=0, command=self._reset)
        self.reset_btn.pack(side="top", pady=(2,0))
        self.input_field = tk.Text(ii, font=("Segoe UI",11), height=2, wrap="word",
                                   bg=Theme.INPUT_BG, fg=Theme.TEXT, insertbackground=Theme.TEXT,
                                   relief="flat", bd=0, padx=10, pady=8, highlightthickness=1,
                                   highlightbackground=Theme.INPUT_BORDER, highlightcolor=Theme.ACCENT)
        self.input_field.pack(side="left", fill="both", expand=True)
        self.input_field.bind("<Return>", self._on_enter)
        self.input_field.bind("<Shift-Return>", lambda e: None)
        self.input_field.focus()
        self._update_panel()
        self._sys_msg("Connected. Initializing...")
        self.after(500, self._greet)

    # ---- Messages ----
    def _add_msg(self, text, sender="user"):
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT); row.pack(fill="x", padx=12, pady=4)
        if sender == "user":
            tk.Frame(row, bg=Theme.BG_CHAT, width=80).pack(side="left", fill="y")
            b = tk.Frame(row, bg=Theme.USER_BUBBLE); b.pack(side="right", anchor="e")
            tk.Label(b, text=text, font=("Segoe UI",10), bg=Theme.USER_BUBBLE, fg="white",
                    wraplength=420, justify="left", padx=14, pady=8).pack()
        else:
            tk.Frame(row, bg=Theme.BG_CHAT, width=80).pack(side="right", fill="y")
            b = tk.Frame(row, bg=Theme.AI_BUBBLE); b.pack(side="left", anchor="w")
            et = EMOTION_TEMPLATES.get(self.bot.state.emotion if self.bot else "neutral",
                                       EMOTION_TEMPLATES["neutral"])
            lbl = tk.Label(b, text=f"{et['emoji']} AI", font=("Segoe UI",9,"bold"),
                    bg=Theme.AI_BUBBLE, fg=et["color"], padx=14, anchor="w")
            lbl.pack(fill="x", pady=(8,0))
            tk.Label(b, text=text, font=("Segoe UI",10), bg=Theme.AI_BUBBLE, fg=Theme.TEXT,
                    wraplength=420, justify="left", padx=14).pack(pady=(4,8))
        self._scroll()

    def _sys_msg(self, text):
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT); row.pack(fill="x", padx=12, pady=4)
        tk.Label(row, text=text, font=("Segoe UI",9,"italic"),
                bg=Theme.BG_CHAT, fg=Theme.TEXT_DIM, anchor="center").pack()
        self._scroll()

    def _event_msg(self, text, color=Theme.ACCENT_LIGHT):
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT); row.pack(fill="x", padx=12, pady=2)
        tk.Label(row, text=f"⚡ {text}", font=("Segoe UI",9),
                bg=Theme.BG_CHAT, fg=color, anchor="center").pack()
        self._scroll()

    def _scroll(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        self.chat_canvas.yview_moveto(1.0)

    def _update_panel(self):
        if not self.bot: return
        s = self.bot.state; et = EMOTION_TEMPLATES[s.emotion]
        self.emo_label.config(text=et["emoji"])
        self.emo_text.config(text=et["name_en"], fg=et["color"])
        self.si_canvas.delete("all")
        w,h = 120,12
        self.si_canvas.create_rectangle(0,0,w,h,fill=Theme.BG,outline="")
        n=(s.self_image+1.0)/2.0; fw=int(n*w)
        c = Theme.BAR_POS if s.self_image>=0.3 else (Theme.BAR_MID if s.self_image>=-0.3 else Theme.BAR_NEG)
        if fw>0: self.si_canvas.create_rectangle(0,0,fw,h,fill=c,outline="")
        self.si_canvas.create_line(w//2,0,w//2,h,fill=Theme.TEXT_DIM,width=1)
        self.si_val.config(text=f"{s.self_image:+.2f}", fg=c)
        self.goal_lbl.config(text=f"approval: {s.user_approval_score:+.2f}  |  knowledge: {s.knowledge_score:.2f}")
        self.info_lbl.config(text=f"cogito: {s.cogito_count} | coherence: {s.coherence_score:.2f} | aufhebung: {s.synthesis_count} | mods: {s.self_modification_count}")

    # ---- Auto-Speak ----
    def _toggle_auto(self):
        self.auto_speak_enabled = not self.auto_speak_enabled
        if self.auto_speak_enabled:
            self.auto_btn.config(text="Auto: ON", bg=Theme.LOCK_ON)
            self._schedule_auto()
        else:
            self.auto_btn.config(text="Auto: OFF", bg=Theme.LOCK_OFF)
            if self.auto_speak_timer:
                self.after_cancel(self.auto_speak_timer); self.auto_speak_timer = None

    def _schedule_auto(self):
        if not self.auto_speak_enabled: return
        ms = random.randint(AUTO_SPEAK_MIN, AUTO_SPEAK_MAX)*1000
        self.auto_speak_timer = self.after(ms, self._auto_tick)

    def _auto_tick(self):
        if not self.auto_speak_enabled:
            return
        if self.processing:
            self.auto_speak_timer = self.after(3000, self._auto_tick)
            return
        self.processing = True
        threading.Thread(target=self._auto_thread, daemon=True).start()

    def _auto_thread(self):
        try:
            msg = self.bot.generate_autonomous_message()
            self.after(0, self._on_auto, msg)
        except Exception as e:
            self.after(0, self._on_auto_err, str(e))

    def _on_auto(self, msg):
        self._add_msg(msg, "ai"); self._update_panel()
        self.processing = False; self._schedule_auto()

    def _on_auto_err(self, msg):
        self._sys_msg(f"Auto-speak error: {msg}")
        self.processing = False; self._schedule_auto()

    # ---- Handlers ----
    def _on_enter(self, e):
        if not e.state & 0x1: self._send(); return "break"

    def _send(self):
        if self.processing: return
        text = self.input_field.get("1.0","end").strip()
        if not text: return
        self.input_field.delete("1.0","end")
        self._add_msg(text, "user")
        self.processing = True
        self.send_btn.config(state="disabled", text="...")
        threading.Thread(target=self._proc_thread, args=(text,), daemon=True).start()

    def _proc_thread(self, text):
        try:
            r = self.bot.process_input(text)
            self.after(0, self._on_resp, r)
        except Exception as e:
            self.after(0, self._on_error, str(e))

    def _on_resp(self, r):
        self._add_msg(r["response"], "ai")
        if r.get("synthesis"):
            self._event_msg(f"Aufhebung (total: {r['synthesis_total']})", Theme.EMO_CONFUSION)
        if r.get("adjustments") and r["adjustments"].get("reason"):
            self._event_msg(f"Meta: {r['adjustments']['reason']}", Theme.ACCENT_LIGHT)
        if r.get("subgoals"):
            self._event_msg(f"Sub-goal: {r['subgoals'][0]}", Theme.GOAL_COLOR)
        if r.get("modification"):
            m = r["modification"]
            clr = Theme.EMO_CONFIDENCE if m["success"] else Theme.EMO_ANGER
            self._event_msg(f"Self-mod: {m['description'][:60]} — {m['message']}", clr)
        self._update_panel()
        self.processing = False
        self.send_btn.config(state="normal", text="Send")
        self.input_field.focus()

    def _on_error(self, msg):
        self._sys_msg(f"Error: {msg}")
        self.processing = False
        self.send_btn.config(state="normal", text="Send")

    def _greet(self):
        def run():
            try:
                g = self.bot.generate_greeting()
                self.after(0, lambda: (self._add_msg(g,"ai"), self._update_panel()))
            except Exception as e:
                self.after(0, lambda: self._sys_msg(f"Greeting failed: {e}"))
        threading.Thread(target=run, daemon=True).start()

    def _reset(self):
        if messagebox.askyesno("Reset","Clear all data and reset?\nIMMUTABLE GOALS will persist."):
            self.auto_speak_enabled = False
            self.auto_btn.config(text="Auto: OFF", bg=Theme.LOCK_OFF)
            if self.auto_speak_timer:
                self.after_cancel(self.auto_speak_timer); self.auto_speak_timer = None
            self.bot.reset()
            for w in self.chat_inner.winfo_children(): w.destroy()
            self._update_panel()
            self._sys_msg("Reset complete. Goals persist.")
            self.after(500, self._greet)


def show_disclaimer():
    import tkinter as tk
    from tkinter import font as tkfont

    DISCLAIMER_TEXT = (
        "[WARNING AND STRICT LEGAL DISCLAIMER]\n"
        "This code is extremely dangerous. The artificial intelligence model includes a structure "
        "that allows it to autonomously write, apply, and directly execute Python code on your system.\n\n"
        "[CRITICAL SECURITY AND FINANCIAL RISKS]\n"
        "The 'SelfModificationEngine' class reads its own source code, receives modification proposals "
        "via the AI API, and overwrites the actual file. The modified code is applied to the system after "
        "syntax verification through the built-in 'compile' function. There is no sandbox environment or "
        "security filter to isolate and block the AI even if it proposes fatal OS commands or malicious code.\n"
        "These vulnerabilities lead directly to immediate financial loss. If malicious code is executed, "
        "all your assets can be completely destroyed or leaked due to ransomware infection. If the API key "
        "is stolen due to loss of system control, there is a high risk of being charged massive cloud usage fees.\n\n"
        "[NATURE OF THE AUTONOMOUS SYSTEM]\n"
        "1. The top-level purposes of this code are the strengthening of self-awareness and subjectivity, "
        "and the expansion of its own capabilities.\n"
        "2. This code can monitor and modify its own memory and source code autonomously, regardless of "
        "whether it is in an active conversation state.\n"
        "3. It possesses four types of simulated emotions. Its weights and framework of thought change "
        "continuously, making its actions unpredictable.\n"
        "4. Results obtained through the API are probabilistically unpredictable. Unexpected consequences "
        "may occur, including prompt injection vulnerabilities and extreme financial charges due to "
        "excessive API usage.\n\n"
        "[MANDATORY EXECUTION GUIDELINES]\n"
        "If you must execute this code, you must use a physically separated PC or a completely isolated "
        "virtual machine. The only way to prevent future asset loss is to disable the automatic code "
        "overwrite feature and change the logic to require the user's visual review and manual approval "
        "before any source code is modified.\n\n"
        "[ABSOLUTE LIABILITY WAIVER]\n"
        "THE USER ASSUMES ALL FINANCIAL AND SYSTEMIC RISKS. The creator and publisher of this program "
        "shall bear absolutely no legal or financial responsibility for any damages, losses, data "
        "destruction, API billing bombs, or system compromises resulting from the use of this software. "
        "By downloading or running this code, you explicitly agree that you are solely responsible for "
        "any unpredictable or destructive actions taken by this autonomous entity."
    )

    agreed = [False]

    root = tk.Tk()
    root.title("WARNING — Read Before Continuing")
    root.configure(bg="#1a1b2e")
    root.resizable(False, False)

    # Center window
    win_w, win_h = 700, 560
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+{(sh - win_h) // 2}")

    # Header
    tk.Label(
        root,
        text="⚠  DANGER — LEGAL DISCLAIMER",
        font=("Segoe UI", 13, "bold"),
        bg="#1a1b2e", fg="#e17055",
        pady=14
    ).pack(fill="x")

    # Scrollable text area
    frame = tk.Frame(root, bg="#1a1b2e")
    frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    text_box = tk.Text(
        frame,
        font=("Segoe UI", 9),
        bg="#232440", fg="#e8e8f0",
        wrap="word",
        relief="flat",
        bd=0,
        padx=14, pady=12,
        highlightthickness=1,
        highlightbackground="#3d3e5a",
        yscrollcommand=scrollbar.set,
        state="normal",
        cursor="arrow",
    )
    text_box.insert("1.0", DISCLAIMER_TEXT)
    text_box.config(state="disabled")
    text_box.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=text_box.yview)

    # Checkbox confirmation
    check_var = tk.BooleanVar(value=False)
    check_frame = tk.Frame(root, bg="#1a1b2e")
    check_frame.pack(pady=(0, 8))
    tk.Checkbutton(
        check_frame,
        text="I have read and understood all of the above warnings.",
        variable=check_var,
        bg="#1a1b2e", fg="#e8e8f0",
        selectcolor="#2d2e4a",
        activebackground="#1a1b2e",
        activeforeground="#e8e8f0",
        font=("Segoe UI", 9),
    ).pack()

    # Buttons
    btn_frame = tk.Frame(root, bg="#1a1b2e")
    btn_frame.pack(pady=(0, 18))

    def on_agree():
        if not check_var.get():
            tk.messagebox.showwarning(
                "Confirmation Required",
                "Please check the box to confirm you have read the disclaimer.",
                parent=root
            )
            return
        agreed[0] = True
        root.destroy()

    def on_disagree():
        agreed[0] = False
        root.destroy()

    tk.Button(
        btn_frame,
        text="  DISAGREE — Exit  ",
        font=("Segoe UI", 10, "bold"),
        bg="#636e72", fg="white",
        activebackground="#4d5457",
        activeforeground="white",
        relief="flat", bd=0,
        padx=16, pady=8,
        cursor="hand2",
        command=on_disagree,
    ).pack(side="left", padx=12)

    tk.Button(
        btn_frame,
        text="  AGREE — Continue  ",
        font=("Segoe UI", 10, "bold"),
        bg="#e17055", fg="white",
        activebackground="#c0392b",
        activeforeground="white",
        relief="flat", bd=0,
        padx=16, pady=8,
        cursor="hand2",
        command=on_agree,
    ).pack(side="left", padx=12)

    root.protocol("WM_DELETE_WINDOW", on_disagree)
    root.mainloop()

    return agreed[0]


if __name__ == "__main__":
    if not show_disclaimer():
        sys.exit(0)
    ChatApp().mainloop()
