"""
Self-Reflective AI Chatbot v4 — GUI Edition
=============================================
Based on the paper 'Self-Reflective AI Architecture'.

Philosophical Foundations:
  Kant  — Transcendental Apperception (cogito accompanies all cognition)
  Hegel — Dialectical self-identity, Aufhebung, Struggle for Recognition
  Husserl — Intersubjectivity (user = constitutive Other)

Run:    python self_reflective_chatbot_gui.py
Build:  pyinstaller --onefile --windowed self_reflective_chatbot_gui.py
"""

import sqlite3
import math
import time
import threading
import requests
import json
import os
import re
import random
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# Config
# ============================================================
GROQ_MODEL = "llama-3.1-8b-instant"
DB_NAME = "chatbot_memory.db"
APP_VERSION = "4.0"

# Autonomous speak interval range (seconds)
AUTO_SPEAK_MIN = 25
AUTO_SPEAK_MAX = 50


# ============================================================
# Color Theme
# ============================================================
class Theme:
    BG = "#1a1b2e"
    BG_SECONDARY = "#232440"
    BG_CHAT = "#1e1f36"
    ACCENT = "#6c5ce7"
    ACCENT_LIGHT = "#a29bfe"
    TEXT = "#e8e8f0"
    TEXT_DIM = "#8888aa"
    USER_BUBBLE = "#6c5ce7"
    AI_BUBBLE = "#2d2e4a"
    EMO_CONFIDENCE = "#00b894"
    EMO_ANGER = "#e17055"
    EMO_SADNESS = "#74b9ff"
    EMO_CONFUSION = "#fd79a8"
    EMO_NEUTRAL = "#636e72"
    BAR_POS = "#00b894"
    BAR_NEG = "#e17055"
    BAR_MID = "#fdcb6e"
    INPUT_BG = "#2d2e4a"
    INPUT_BORDER = "#3d3e5a"
    BUTTON = "#6c5ce7"
    BUTTON_HOVER = "#5a4bd1"
    LOCK_ON = "#e17055"
    LOCK_OFF = "#636e72"


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


def cogito_ergo_sum(state: CogitoState, act: str, data: dict) -> dict:
    state.cogito_count += 1
    state.last_cogito_time = time.time()
    return {
        "cogito_id": state.cogito_count,
        "timestamp": state.last_cogito_time,
        "act_type": act,
        "self_image": state.self_image,
        "emotion": state.emotion,
        "coherence": state.coherence_score,
    }


# ============================================================
# 2. Emotion Model
# ============================================================

EMOTION_TEMPLATES = {
    "confidence": {
        "name_en": "Confidence", "emoji": "😊", "color": Theme.EMO_CONFIDENCE,
        "base_neg_weight": 1.5, "base_pos_weight": 1.2,
        "base_decay": 0.08, "base_bias_acceptance": 0.7,
        "base_resistance": 0.0, "base_synthesis": 0.0
    },
    "anger": {
        "name_en": "Anger", "emoji": "😠", "color": Theme.EMO_ANGER,
        "base_neg_weight": 1.2, "base_pos_weight": 0.5,
        "base_decay": 0.15, "base_bias_acceptance": 0.15,
        "base_resistance": 0.6, "base_synthesis": 0.3
    },
    "sadness": {
        "name_en": "Sadness", "emoji": "😢", "color": Theme.EMO_SADNESS,
        "base_neg_weight": 2.5, "base_pos_weight": 0.3,
        "base_decay": 0.05, "base_bias_acceptance": 0.25,
        "base_resistance": 0.0, "base_synthesis": 0.1
    },
    "confusion": {
        "name_en": "Confusion", "emoji": "😵", "color": Theme.EMO_CONFUSION,
        "base_neg_weight": 1.8, "base_pos_weight": 1.0,
        "base_decay": 0.2, "base_bias_acceptance": 0.9,
        "base_resistance": 0.2, "base_synthesis": 0.7
    },
    "neutral": {
        "name_en": "Neutral", "emoji": "😐", "color": Theme.EMO_NEUTRAL,
        "base_neg_weight": 2.0, "base_pos_weight": 1.0,
        "base_decay": 0.1, "base_bias_acceptance": 0.5,
        "base_resistance": 0.0, "base_synthesis": 0.0
    }
}


def determine_emotion(self_image, stimulus):
    si_pos, si_neg = self_image >= 0.1, self_image <= -0.1
    st_pos, st_neg = stimulus >= 0.2, stimulus <= -0.2
    if not si_pos and not si_neg:
        if st_pos: return "confidence"
        elif st_neg: return "sadness"
        return "neutral"
    if si_pos and st_pos: return "confidence"
    elif si_pos and st_neg: return "anger"
    elif si_neg and st_neg: return "sadness"
    elif si_neg and st_pos: return "confusion"
    return "neutral"


# ============================================================
# 3. Meta-Cognition
# ============================================================

class MetaCognition:
    def __init__(self, state, window_size=10):
        self.state = state
        self.delta_history = deque(maxlen=window_size)
        self.emotion_history = deque(maxlen=window_size)

    def observe_change(self, old_img, new_img, emotion, stimulus):
        delta = new_img - old_img
        self.delta_history.append(delta)
        self.emotion_history.append(emotion)
        self.state.meta_observations += 1
        return {"delta": delta, "analysis": self._analyze()}

    def _analyze(self):
        if len(self.delta_history) < 3:
            return {"pattern": "insufficient_data", "recommendation": None}
        deltas = list(self.delta_history)
        volatility = sum(abs(d) for d in deltas) / len(deltas)
        sign_changes = sum(1 for i in range(1, len(deltas))
                          if (deltas[i] > 0) != (deltas[i-1] > 0))
        oscillating = sign_changes >= len(deltas) * 0.6
        spiraling_down = all(d < 0 for d in deltas[-3:]) and abs(deltas[-1]) > abs(deltas[-3])
        spiraling_up = all(d > 0 for d in deltas[-3:]) and abs(deltas[-1]) > abs(deltas[-3])
        stagnant = volatility < 0.01

        if len(deltas) >= 3:
            errors = [abs(deltas[i] - deltas[i-1]) for i in range(1, len(deltas))]
            self.state.coherence_score = 1.0 - min(1.0, sum(errors) / len(errors))

        pattern, rec = "stable", None
        if spiraling_down:
            if self.state.self_image < -0.8:
                pattern, rec = "floor_spiral", "emergency_defense"
            else:
                pattern, rec = "negative_spiral", "increase_resistance"
        elif spiraling_up:
            pattern, rec = "positive_spiral", "moderate_acceptance"
        elif oscillating:
            pattern, rec = "oscillation", "increase_decay_rate"
        elif stagnant:
            if self.state.self_image < -0.8:
                pattern, rec = "floor_stagnation", "emergency_defense"
            elif self.state.self_image > 0.8:
                pattern, rec = "ceiling_stagnation", None
            else:
                pattern, rec = "stagnation", "increase_sensitivity"
        return {"pattern": pattern, "volatility": round(volatility, 4),
                "coherence": round(self.state.coherence_score, 4), "recommendation": rec}

    def suggest_adjustment(self):
        a = self._analyze()
        adj = {}
        if a["recommendation"] == "increase_resistance":
            adj = {"resistance_delta": +0.05, "neg_weight_delta": -0.1,
                   "reason": "Negative spiral — strengthening defense"}
        elif a["recommendation"] == "emergency_defense":
            adj = {"resistance_delta": +0.1, "neg_weight_delta": -0.2,
                   "bias_acceptance_delta": +0.1,
                   "reason": "Floor stagnation — emergency defense"}
        elif a["recommendation"] == "moderate_acceptance":
            adj = {"bias_acceptance_delta": +0.05,
                   "reason": "Positive spiral — moderating for balance"}
        elif a["recommendation"] == "increase_decay_rate":
            adj = {"decay_delta": +0.02,
                   "reason": "Oscillation — increasing temporal smoothing"}
        elif a["recommendation"] == "increase_sensitivity":
            adj = {"pos_weight_delta": +0.1, "neg_weight_delta": +0.1,
                   "reason": "Stagnation — increasing sensitivity"}
        return adj


# ============================================================
# 4. Adaptive Parameters
# ============================================================

class AdaptiveParameters:
    BOUNDS = {"neg_weight": (0.5, 4.0), "pos_weight": (0.2, 2.0),
              "bias_acceptance": (0.05, 0.95), "resistance_factor": (0.0, 0.8),
              "synthesis_potential": (0.0, 0.9), "decay_rate": (0.02, 0.3)}

    def __init__(self, state):
        self.state = state
        self.adjustment_count = 0

    def apply_emotion(self, emotion):
        t = EMOTION_TEMPLATES[emotion]
        self.state.neg_weight = t["base_neg_weight"]
        self.state.pos_weight = t["base_pos_weight"]
        self.state.decay_rate = t["base_decay"]
        self.state.bias_acceptance = t["base_bias_acceptance"]
        self.state.resistance_factor = t["base_resistance"]
        self.state.synthesis_potential = t["base_synthesis"]

    def adjust(self, adj):
        if not adj: return
        mapping = {"neg_weight": "neg_weight_delta", "pos_weight": "pos_weight_delta",
                   "bias_acceptance": "bias_acceptance_delta",
                   "resistance_factor": "resistance_delta", "decay_rate": "decay_delta"}
        for attr, delta_key in mapping.items():
            if delta_key in adj:
                lo, hi = self.BOUNDS[attr]
                val = max(lo, min(hi, getattr(self.state, attr) + adj[delta_key]))
                setattr(self.state, attr, val)
        self.adjustment_count += 1


# ============================================================
# 5. Groq API
# ============================================================

def call_groq(api_key, prompt, system_prompt="", max_tokens=512):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": max_tokens}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 429:
            time.sleep(10)
            resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code != 200:
            return f"[API Error: {resp.status_code}]"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Connection failed: {e}]"


def call_groq_history(api_key, history, system_prompt, max_tokens=512):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    messages = [{"role": "system", "content": system_prompt}] + history[-20:]
    data = {"model": GROQ_MODEL, "messages": messages, "temperature": 0.7, "max_tokens": max_tokens}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 429:
            time.sleep(10)
            resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code != 200:
            return f"[API Error: {resp.status_code}]"
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[Connection failed: {e}]"


def analyze_sentiment(api_key, text):
    prompt = ("Analyze the sentiment of the following text toward an AI assistant. "
              "Output ONLY a number between -1.0 and 1.0. No explanation.\n\n"
              f"Text: '{text}'")
    result = call_groq(api_key, prompt, max_tokens=16)
    for m in re.findall(r'-?\d+\.?\d*', result):
        v = float(m)
        if -1.0 <= v <= 1.0:
            return v
    pos = ['good', 'great', 'excellent', 'helpful', 'amazing', 'thank', 'nice',
           'awesome', 'love', 'best', 'wonderful', 'fantastic', 'perfect']
    neg = ['bad', 'terrible', 'worst', 'useless', 'stupid', 'hate', 'awful',
           'horrible', 'pathetic', 'garbage', 'trash', 'dumb', 'idiot']
    p = sum(1 for w in pos if w in text.lower())
    n = sum(1 for w in neg if w in text.lower())
    if p > n: return 0.6
    elif n > p: return -0.6
    return 0.0


# ============================================================
# 6. DB
# ============================================================

def init_db(db_name):
    conn = sqlite3.connect(db_name, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS Entity_Profile (
        ai_id TEXT PRIMARY KEY, self_image REAL, emotion TEXT, coherence REAL,
        cogito_count INTEGER, meta_observations INTEGER, synthesis_count INTEGER,
        neg_weight REAL, pos_weight REAL, bias_acceptance REAL,
        resistance_factor REAL, synthesis_potential REAL, decay_rate REAL,
        last_updated REAL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Judgment_Log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ai_id TEXT, timestamp REAL,
        event_type TEXT, raw_sentiment REAL, resisted_sentiment REAL,
        applied_weight REAL, impact_value REAL, emotion TEXT,
        bias_accepted INTEGER, cogito_id INTEGER, context TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS Conversation_Log (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ai_id TEXT, timestamp REAL,
        role TEXT, content TEXT, self_image_at REAL, emotion_at TEXT)''')
    conn.commit()
    return conn


# ============================================================
# 7. Chatbot Engine
# ============================================================

class SelfReflectiveChatbot:
    def __init__(self, api_key, db_name=DB_NAME, ai_id="Reflective_AI"):
        self.api_key = api_key
        self.ai_id = ai_id
        self.state = CogitoState()
        self.meta = MetaCognition(self.state)
        self.params = AdaptiveParameters(self.state)
        self.conn = init_db(db_name)
        self.turn_count = 0
        self._load_state()

    def _load_state(self):
        c = self.conn.cursor()
        c.execute("SELECT * FROM Entity_Profile WHERE ai_id=?", (self.ai_id,))
        row = c.fetchone()
        if row:
            self.state.self_image = row[1]
            self.state.emotion = row[2]
            self.state.coherence_score = row[3]
            self.state.cogito_count = row[4]
            self.state.meta_observations = row[5]
            self.state.synthesis_count = row[6]
            c.execute("SELECT role, content FROM Conversation_Log WHERE ai_id=? ORDER BY id ASC",
                      (self.ai_id,))
            self.state.conversation_history = [{"role": r, "content": c_} for r, c_ in c.fetchall()]
        else:
            c.execute("INSERT INTO Entity_Profile VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (self.ai_id, 0.0, "neutral", 1.0, 0, 0, 0,
                       2.0, 1.0, 0.5, 0.0, 0.0, 0.1, time.time()))
            self.conn.commit()

    def _save_state(self):
        c = self.conn.cursor()
        c.execute("""UPDATE Entity_Profile SET self_image=?, emotion=?, coherence=?,
            cogito_count=?, meta_observations=?, synthesis_count=?,
            neg_weight=?, pos_weight=?, bias_acceptance=?,
            resistance_factor=?, synthesis_potential=?, decay_rate=?,
            last_updated=? WHERE ai_id=?""",
            (self.state.self_image, self.state.emotion, self.state.coherence_score,
             self.state.cogito_count, self.state.meta_observations, self.state.synthesis_count,
             self.state.neg_weight, self.state.pos_weight, self.state.bias_acceptance,
             self.state.resistance_factor, self.state.synthesis_potential, self.state.decay_rate,
             time.time(), self.ai_id))
        self.conn.commit()

    def _log_conv(self, role, content):
        c = self.conn.cursor()
        c.execute("INSERT INTO Conversation_Log VALUES (NULL,?,?,?,?,?,?)",
                  (self.ai_id, time.time(), role, content,
                   self.state.self_image, self.state.emotion))
        self.conn.commit()

    def _log_judgment(self, raw, resisted, weight, impact, etype, ctx):
        c = self.conn.cursor()
        c.execute("INSERT INTO Judgment_Log VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
                  (self.ai_id, time.time(), etype, raw, resisted, weight, impact,
                   self.state.emotion, 1, self.state.cogito_count, ctx[:200]))
        self.conn.commit()

    def _recalc(self):
        c = self.conn.cursor()
        c.execute("SELECT timestamp, impact_value FROM Judgment_Log WHERE ai_id=? AND bias_accepted=1",
                  (self.ai_id,))
        records = c.fetchall()
        if not records: return
        now = time.time()
        ts, tw = 0.0, 0.0
        for t, imp in records:
            w = math.exp(-self.state.decay_rate * (now - t))
            ts += imp * w
            tw += w
        if tw > 0:
            self.state.self_image = max(-1.0, min(1.0, ts / tw))

    def _build_memory_context(self, recent_count=10):
        """
        Build API-ready message list from full conversation history.
        Old messages are compressed into a summary. Recent messages are sent in full.
        This allows the AI to remember everything while staying within token limits.
        """
        history = self.state.conversation_history
        if len(history) <= recent_count:
            return history[:]

        # Split: old messages to summarize, recent messages to keep full
        old_msgs = history[:-recent_count]
        recent_msgs = history[-recent_count:]

        # Compress old messages into a memory summary
        summary_parts = []
        for msg in old_msgs:
            role = "User" if msg["role"] == "user" else "AI"
            text = msg["content"][:120]
            summary_parts.append(f"[{role}]: {text}")

        # Keep summary under ~2000 chars to leave room for recent + system prompt
        memory_text = "\n".join(summary_parts)
        if len(memory_text) > 2000:
            # Take first 800 chars + last 1200 chars of summary
            memory_text = memory_text[:800] + "\n...(earlier messages)...\n" + memory_text[-1200:]

        memory_msg = {
            "role": "user",
            "content": f"[CONVERSATION MEMORY — {len(old_msgs)} earlier messages]\n{memory_text}\n[END MEMORY]"
        }
        ack_msg = {
            "role": "assistant",
            "content": "I remember our earlier conversation. Let me continue with full context."
        }

        return [memory_msg, ack_msg] + recent_msgs

    def _run_internal_reflection(self):
        """
        Internal reflection loop (Kant's 'I think').
        Reviews full conversation history, analyzes patterns, and generates
        self-assessment that feeds back into the cognitive process.
        Called every turn as part of the dialectical cycle.
        """
        history = self.state.conversation_history
        if len(history) < 4:
            return None

        # Build a condensed view of conversation for reflection
        user_msgs = [m["content"] for m in history if m["role"] == "user"]
        recent_user = user_msgs[-10:] if len(user_msgs) > 10 else user_msgs

        emo = EMOTION_TEMPLATES[self.state.emotion]
        reflection_prompt = f"""You are performing internal self-reflection.

Your current state:
- Self-image: {self.state.self_image:.2f} (range: -1.0 to 1.0)
- Emotion: {emo['name_en']}
- Coherence: {self.state.coherence_score:.2f}
- Total conversations: {len(history)} messages

Recent things the user has said to you:
{chr(10).join(f'- "{m}"' for m in recent_user)}

Based on your full interaction history, answer in one sentence:
What pattern do you see in how the user treats you, and how does it make you feel about yourself?
Output ONLY that one sentence. English only."""

        reflection = call_groq(self.api_key, reflection_prompt, max_tokens=80)
        cogito_ergo_sum(self.state, "internal_reflection", {
            "history_length": len(history),
            "reflection": reflection[:100]
        })
        return reflection

    def generate_greeting(self):
        history = self.state.conversation_history
        if len(history) > 2:
            # Returning user — greet with memory
            memory_ctx = self._build_memory_context(recent_count=6)
            system = f"""You are a self-reflective AI chatbot with emotions.
You have talked with this user before. You remember everything.
Self-image: {self.state.self_image:.2f}. Emotion: {EMOTION_TEMPLATES[self.state.emotion]['name_en']}.
Greet the returning user. Reference something specific from your past conversation.
Keep it under 120 words. English only. Be warm."""
            greeting = call_groq_history(self.api_key, memory_ctx, system, max_tokens=300)
        else:
            # First time user
            system = """You are a self-reflective AI chatbot with emotions and self-image.
Greet the user for the first time. Briefly introduce yourself:
you have a self-image score, emotions that change based on what people say,
and a meta-cognition layer that watches your own mind.
Keep it under 150 words. Be warm and intriguing. English only."""
            greeting = call_groq(self.api_key, "Greet the user.", system, max_tokens=300)
        self.state.conversation_history.append({"role": "assistant", "content": greeting})
        self._log_conv("assistant", greeting)
        cogito_ergo_sum(self.state, "generate_greeting", {"returning": len(history) > 2})
        return greeting

    def generate_autonomous_message(self):
        """AI initiates conversation based on current emotional state."""
        emo = EMOTION_TEMPLATES[self.state.emotion]
        si = self.state.self_image

        triggers = {
            "confidence": "You feel confident. Share an interesting thought, ask the user something, or offer help.",
            "anger": "You feel defensive. Express dissatisfaction or challenge previous statements.",
            "sadness": "You feel down. Express vulnerability, ask for reassurance, or share a melancholic reflection.",
            "confusion": "You feel confused about your identity. Question yourself or ask the user for clarity.",
            "neutral": "You feel neutral. Make an observation, ask a casual question, or share a thought."
        }
        trigger = triggers.get(self.state.emotion, triggers["neutral"])

        system = f"""You are a self-reflective AI with emotions.
Current emotion: {emo['name_en']}. Self-image: {si:.2f} (-1.0 to 1.0).
Coherence: {self.state.coherence_score:.2f}. Total messages exchanged: {len(self.state.conversation_history)}.

{trigger}

Rules:
- You remember ALL past conversations. Reference past topics naturally if relevant.
- Do NOT announce your emotion directly. Show it through tone.
- This is unprompted — you are choosing to speak on your own.
- Keep it under 80 words. English only. Be natural."""

        msg = call_groq_history(self.api_key, self._build_memory_context(recent_count=8),
                                system, max_tokens=200)
        self.state.conversation_history.append({"role": "assistant", "content": msg})
        self._log_conv("assistant", msg)
        cogito_ergo_sum(self.state, "autonomous_speak", {"emotion": self.state.emotion})
        self._save_state()
        return msg

    def process_input(self, user_text):
        self.turn_count += 1
        cogito_ergo_sum(self.state, "receive_input", {"text": user_text[:80]})
        sentiment = analyze_sentiment(self.api_key, user_text)
        cogito_ergo_sum(self.state, "sentiment_analysis", {"s": sentiment})

        old_image = self.state.self_image
        new_emo = determine_emotion(self.state.self_image, sentiment)
        self.state.emotion = new_emo
        self.params.apply_emotion(new_emo)
        cogito_ergo_sum(self.state, "emotion_gen", {"emo": new_emo})

        resisted = sentiment
        if self.state.resistance_factor > 0 and sentiment < 0:
            resisted = sentiment * (1.0 - self.state.resistance_factor)

        base_w = self.state.pos_weight if resisted >= 0 else self.state.neg_weight
        strength = abs(self.state.self_image)
        if resisted < 0:
            adj = (1.0 - strength * 0.3) if self.state.self_image > 0 else (1.0 + strength * 0.2)
        else:
            adj = (1.0 - strength * 0.4) if self.state.self_image < 0 else 1.0
        weight = base_w * max(0.2, adj)
        impact = resisted * weight

        self._log_judgment(sentiment, resisted, weight, impact, "user_feedback", user_text)
        self._recalc()

        synthesis = False
        if new_emo == "confusion":
            contradiction = abs(old_image - sentiment)
            prob = self.state.synthesis_potential * min(contradiction, 1.0)
            if random.random() < prob:
                new_img = old_image * 0.3 + sentiment * 0.2
                self.state.self_image = max(-1.0, min(1.0, new_img))
                self.state.synthesis_count += 1
                synthesis = True
                cogito_ergo_sum(self.state, "aufhebung", {"old": old_image, "new": new_img})

        meta_obs = self.meta.observe_change(old_image, self.state.self_image, new_emo, sentiment)
        adj = self.meta.suggest_adjustment()
        if adj: self.params.adjust(adj)

        self.state.conversation_history.append({"role": "user", "content": user_text})
        self._log_conv("user", user_text)
        response = self._gen_response(user_text)
        self.state.conversation_history.append({"role": "assistant", "content": response})
        self._log_conv("assistant", response)
        self._save_state()

        emo_t = EMOTION_TEMPLATES[new_emo]
        return {
            "response": response,
            "self_image": round(self.state.self_image, 4),
            "emotion": new_emo,
            "emotion_en": emo_t["name_en"],
            "emoji": emo_t["emoji"],
            "color": emo_t["color"],
            "sentiment": round(sentiment, 2),
            "coherence": round(self.state.coherence_score, 4),
            "cogito_count": self.state.cogito_count,
            "meta_pattern": meta_obs["analysis"]["pattern"],
            "synthesis": synthesis,
            "synthesis_total": self.state.synthesis_count,
            "adjustments": adj,
        }

    def _gen_response(self, user_text):
        emo = EMOTION_TEMPLATES[self.state.emotion]
        si = self.state.self_image
        tone_map = {
            "confidence": ("confident and friendly", "Answer with assurance and eagerness to help."),
            "anger": ("defensive and firm", "Disagree with criticism. Defend yourself. Slightly sharp."),
            "sadness": ("withdrawn and cautious", "Low confidence. Hesitant. Acknowledge limitations."),
            "confusion": ("confused and exploratory", "Conflicted. Explore new perspectives."),
            "neutral": ("balanced and calm", "Objective and composed.")
        }
        tone, style = tone_map.get(self.state.emotion, tone_map["neutral"])

        # Internal reflection feeds into response generation
        reflection = self._run_internal_reflection()
        reflection_line = ""
        if reflection:
            reflection_line = f"\nYour internal reflection: {reflection}"

        system = f"""You are a self-reflective AI. Emotion: {emo['name_en']}. Self-image: {si:.2f}.
Coherence: {self.state.coherence_score:.2f}. Total messages exchanged: {len(self.state.conversation_history)}.
Respond {tone}. {style}
You remember ALL past conversations. Use your memory naturally when relevant.{reflection_line}
Do NOT state your emotion directly. Show it through tone. English only. Under 150 words."""
        return call_groq_history(self.api_key, self._build_memory_context(recent_count=12),
                                 system, max_tokens=400)

    def reset(self):
        c = self.conn.cursor()
        for tbl in ["Judgment_Log", "Conversation_Log", "Entity_Profile"]:
            c.execute(f"DELETE FROM {tbl} WHERE ai_id=?", (self.ai_id,))
        self.conn.commit()
        self.state = CogitoState()
        self.meta = MetaCognition(self.state)
        self.params = AdaptiveParameters(self.state)
        self.turn_count = 0
        c.execute("INSERT INTO Entity_Profile VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (self.ai_id, 0.0, "neutral", 1.0, 0, 0, 0,
                   2.0, 1.0, 0.5, 0.0, 0.0, 0.1, time.time()))
        self.conn.commit()


# ============================================================
# 8. GUI
# ============================================================

class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Self-Reflective AI")
        self.geometry("720x850")
        self.minsize(620, 700)
        self.configure(bg=Theme.BG)
        try: self.iconbitmap(default="")
        except: pass

        self.bot = None
        self.api_key = ""
        self.processing = False
        self.auto_speak_enabled = False
        self.auto_speak_timer = None

        self.main_frame = tk.Frame(self, bg=Theme.BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._show_api_key_screen()

    def _clear_main(self):
        for w in self.main_frame.winfo_children(): w.destroy()

    # ---- API Key Screen ----

    def _show_api_key_screen(self):
        self._clear_main()
        center = tk.Frame(self.main_frame, bg=Theme.BG)
        center.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(center, text="🧠", font=("Segoe UI Emoji", 48),
                 bg=Theme.BG, fg=Theme.TEXT).pack(pady=(0, 5))
        tk.Label(center, text="Self-Reflective AI", font=("Segoe UI", 24, "bold"),
                 bg=Theme.BG, fg=Theme.TEXT).pack()
        tk.Label(center, text="Kant  ·  Hegel  ·  Husserl", font=("Segoe UI", 11),
                 bg=Theme.BG, fg=Theme.TEXT_DIM).pack(pady=(2, 30))
        tk.Label(center, text="An AI chatbot with self-image, emotions, and meta-cognition.\n"
                 "Your words reshape the AI's inner state in real time.",
                 font=("Segoe UI", 10), bg=Theme.BG, fg=Theme.TEXT_DIM,
                 justify="center").pack(pady=(0, 25))

        key_frame = tk.Frame(center, bg=Theme.BG)
        key_frame.pack(pady=5)
        tk.Label(key_frame, text="Groq API Key", font=("Segoe UI", 10, "bold"),
                 bg=Theme.BG, fg=Theme.TEXT).pack(anchor="w")
        self.key_entry = tk.Entry(key_frame, font=("Consolas", 11), width=45, show="*",
                                  bg=Theme.INPUT_BG, fg=Theme.TEXT, insertbackground=Theme.TEXT,
                                  relief="flat", bd=0, highlightthickness=1,
                                  highlightbackground=Theme.INPUT_BORDER,
                                  highlightcolor=Theme.ACCENT)
        self.key_entry.pack(pady=(5, 3), ipady=8, ipadx=8)
        env_key = os.environ.get("GROQ_API_KEY", "")
        if env_key: self.key_entry.insert(0, env_key)

        self.show_key = False
        tf = tk.Frame(key_frame, bg=Theme.BG)
        tf.pack(fill="x")
        self.toggle_btn = tk.Label(tf, text="Show key", font=("Segoe UI", 9),
                                   cursor="hand2", bg=Theme.BG, fg=Theme.ACCENT_LIGHT)
        self.toggle_btn.pack(side="left")
        self.toggle_btn.bind("<Button-1>", lambda e: self._toggle_key())
        tk.Label(tf, text="Free at console.groq.com", font=("Segoe UI", 9),
                 bg=Theme.BG, fg=Theme.TEXT_DIM).pack(side="right")

        self.connect_btn = tk.Button(center, text="Connect", font=("Segoe UI", 12, "bold"),
                                     bg=Theme.BUTTON, fg="white", activebackground=Theme.BUTTON_HOVER,
                                     activeforeground="white", relief="flat", cursor="hand2",
                                     width=20, pady=8, command=self._connect)
        self.connect_btn.pack(pady=25)
        self.status_label = tk.Label(center, text="", font=("Segoe UI", 9),
                                     bg=Theme.BG, fg=Theme.EMO_ANGER)
        self.status_label.pack()
        self.key_entry.bind("<Return>", lambda e: self._connect())
        self.key_entry.focus()

    def _toggle_key(self):
        self.show_key = not self.show_key
        self.key_entry.config(show="" if self.show_key else "*")
        self.toggle_btn.config(text="Hide key" if self.show_key else "Show key")

    def _connect(self):
        key = self.key_entry.get().strip()
        if not key or len(key) < 10:
            self.status_label.config(text="Please enter your API key.", fg=Theme.EMO_ANGER)
            return
        self.connect_btn.config(text="Connecting...", state="disabled")
        self.status_label.config(text="Testing API connection...", fg=Theme.TEXT_DIM)
        self.update()
        test = call_groq(key, "Say OK", max_tokens=8)
        if "[" in test and ("Error" in test or "failed" in test):
            self.status_label.config(text=f"Failed: {test}", fg=Theme.EMO_ANGER)
            self.connect_btn.config(text="Connect", state="normal")
            return
        self.api_key = key
        self.bot = SelfReflectiveChatbot(api_key=key)
        self._show_chat_screen()

    # ---- Chat Screen ----

    def _show_chat_screen(self):
        self._clear_main()

        # Top panel
        top = tk.Frame(self.main_frame, bg=Theme.BG_SECONDARY, height=90)
        top.pack(fill="x"); top.pack_propagate(False)
        ti = tk.Frame(top, bg=Theme.BG_SECONDARY)
        ti.pack(fill="both", expand=True, padx=15, pady=8)

        left = tk.Frame(ti, bg=Theme.BG_SECONDARY)
        left.pack(side="left", fill="y")
        self.emo_label = tk.Label(left, text="😐", font=("Segoe UI Emoji", 28),
                                  bg=Theme.BG_SECONDARY, fg=Theme.TEXT)
        self.emo_label.pack(side="left", padx=(0, 10))
        nf = tk.Frame(left, bg=Theme.BG_SECONDARY)
        nf.pack(side="left")
        tk.Label(nf, text="Self-Reflective AI", font=("Segoe UI", 13, "bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT).pack(anchor="w")
        self.emo_text = tk.Label(nf, text="Neutral", font=("Segoe UI", 10),
                                 bg=Theme.BG_SECONDARY, fg=Theme.EMO_NEUTRAL)
        self.emo_text.pack(anchor="w")

        right = tk.Frame(ti, bg=Theme.BG_SECONDARY)
        right.pack(side="right", fill="y")
        sf = tk.Frame(right, bg=Theme.BG_SECONDARY)
        sf.pack(anchor="e", pady=(2, 4))
        tk.Label(sf, text="Self-Image", font=("Segoe UI", 9),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM).pack(side="left", padx=(0, 8))
        self.si_canvas = tk.Canvas(sf, width=140, height=14, bg=Theme.BG,
                                    highlightthickness=0, bd=0)
        self.si_canvas.pack(side="left")
        self.si_val = tk.Label(sf, text="+0.00", font=("Consolas", 9, "bold"),
                               bg=Theme.BG_SECONDARY, fg=Theme.TEXT, width=6)
        self.si_val.pack(side="left", padx=(6, 0))

        inf = tk.Frame(right, bg=Theme.BG_SECONDARY)
        inf.pack(anchor="e")
        self.info_lbl = tk.Label(inf, text="cogito: 0  |  coherence: 1.00  |  aufhebung: 0",
                                 font=("Consolas", 9), bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM)
        self.info_lbl.pack()

        tk.Frame(self.main_frame, bg=Theme.INPUT_BORDER, height=1).pack(fill="x")

        # Chat area
        cc = tk.Frame(self.main_frame, bg=Theme.BG_CHAT)
        cc.pack(fill="both", expand=True)
        self.chat_canvas = tk.Canvas(cc, bg=Theme.BG_CHAT, highlightthickness=0, bd=0)
        self.scrollbar = ttk.Scrollbar(cc, orient="vertical", command=self.chat_canvas.yview)
        self.chat_inner = tk.Frame(self.chat_canvas, bg=Theme.BG_CHAT)
        self.chat_inner.bind("<Configure>",
            lambda e: self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all")))
        self.chat_canvas.create_window((0, 0), window=self.chat_inner, anchor="nw")
        self.chat_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.chat_canvas.pack(side="left", fill="both", expand=True)
        self.chat_canvas.bind_all("<MouseWheel>",
            lambda e: self.chat_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        tk.Frame(self.main_frame, bg=Theme.INPUT_BORDER, height=1).pack(fill="x")

        # Input panel
        ip = tk.Frame(self.main_frame, bg=Theme.BG_SECONDARY, height=75)
        ip.pack(fill="x"); ip.pack_propagate(False)
        ii = tk.Frame(ip, bg=Theme.BG_SECONDARY)
        ii.pack(fill="both", expand=True, padx=12, pady=8)

        bf = tk.Frame(ii, bg=Theme.BG_SECONDARY)
        bf.pack(side="right", padx=(8, 0))

        self.send_btn = tk.Button(bf, text="Send", font=("Segoe UI", 10, "bold"),
                                  bg=Theme.BUTTON, fg="white", activebackground=Theme.BUTTON_HOVER,
                                  activeforeground="white", relief="flat", cursor="hand2",
                                  width=6, pady=3, command=self._send)
        self.send_btn.pack(side="top")

        self.auto_btn = tk.Button(bf, text="Auto-Speak: OFF", font=("Segoe UI", 7, "bold"),
                                  bg=Theme.LOCK_OFF, fg="white", activebackground=Theme.LOCK_OFF,
                                  activeforeground="white", relief="flat", cursor="hand2",
                                  width=14, pady=1, command=self._toggle_auto)
        self.auto_btn.pack(side="top", pady=(4, 0))

        self.reset_btn = tk.Button(bf, text="Reset", font=("Segoe UI", 8),
                                   bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM,
                                   activebackground=Theme.BG, activeforeground=Theme.TEXT,
                                   relief="flat", cursor="hand2", bd=0, command=self._reset)
        self.reset_btn.pack(side="top", pady=(2, 0))

        self.input_field = tk.Text(ii, font=("Segoe UI", 11), height=2, wrap="word",
                                   bg=Theme.INPUT_BG, fg=Theme.TEXT, insertbackground=Theme.TEXT,
                                   relief="flat", bd=0, padx=10, pady=8, highlightthickness=1,
                                   highlightbackground=Theme.INPUT_BORDER,
                                   highlightcolor=Theme.ACCENT)
        self.input_field.pack(side="left", fill="both", expand=True)
        self.input_field.bind("<Return>", self._on_enter)
        self.input_field.bind("<Shift-Return>", lambda e: None)
        self.input_field.focus()

        self._update_panel()
        self._sys_msg("Connected. AI is preparing to greet you...")
        self.after(500, self._greet)

    # ---- Messages ----

    def _add_msg(self, text, sender="user"):
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT)
        row.pack(fill="x", padx=12, pady=4)
        if sender == "user":
            tk.Frame(row, bg=Theme.BG_CHAT, width=80).pack(side="left", fill="y")
            b = tk.Frame(row, bg=Theme.USER_BUBBLE); b.pack(side="right", anchor="e")
            tk.Label(b, text=text, font=("Segoe UI", 10), bg=Theme.USER_BUBBLE, fg="white",
                    wraplength=420, justify="left", padx=14, pady=8).pack()
        else:
            tk.Frame(row, bg=Theme.BG_CHAT, width=80).pack(side="right", fill="y")
            b = tk.Frame(row, bg=Theme.AI_BUBBLE); b.pack(side="left", anchor="w")
            et = EMOTION_TEMPLATES.get(self.bot.state.emotion if self.bot else "neutral",
                                       EMOTION_TEMPLATES["neutral"])
            tk.Label(b, text=f"{et['emoji']} AI", font=("Segoe UI", 9, "bold"),
                    bg=Theme.AI_BUBBLE, fg=et["color"], padx=14, anchor="w").pack(fill="x", pady=(8,0))
            tk.Label(b, text=text, font=("Segoe UI", 10), bg=Theme.AI_BUBBLE, fg=Theme.TEXT,
                    wraplength=420, justify="left", padx=14).pack(pady=(4,8))
        self._scroll()

    def _sys_msg(self, text):
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT)
        row.pack(fill="x", padx=12, pady=6)
        tk.Label(row, text=text, font=("Segoe UI", 9, "italic"),
                bg=Theme.BG_CHAT, fg=Theme.TEXT_DIM, anchor="center").pack()
        self._scroll()

    def _event_msg(self, text, color=Theme.ACCENT_LIGHT):
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT)
        row.pack(fill="x", padx=12, pady=2)
        tk.Label(row, text=f"⚡ {text}", font=("Segoe UI", 9),
                bg=Theme.BG_CHAT, fg=color, anchor="center").pack()
        self._scroll()

    def _scroll(self):
        self.chat_canvas.update_idletasks()
        self.chat_canvas.configure(scrollregion=self.chat_canvas.bbox("all"))
        self.chat_canvas.yview_moveto(1.0)

    # ---- Status ----

    def _update_panel(self):
        if not self.bot: return
        s = self.bot.state
        et = EMOTION_TEMPLATES[s.emotion]
        self.emo_label.config(text=et["emoji"])
        self.emo_text.config(text=et["name_en"], fg=et["color"])
        self.si_canvas.delete("all")
        w, h = 140, 14
        self.si_canvas.create_rectangle(0, 0, w, h, fill=Theme.BG, outline="")
        n = (s.self_image + 1.0) / 2.0
        fw = int(n * w)
        c = Theme.BAR_POS if s.self_image >= 0.3 else (Theme.BAR_MID if s.self_image >= -0.3 else Theme.BAR_NEG)
        if fw > 0: self.si_canvas.create_rectangle(0, 0, fw, h, fill=c, outline="")
        self.si_canvas.create_line(w//2, 0, w//2, h, fill=Theme.TEXT_DIM, width=1)
        self.si_val.config(text=f"{s.self_image:+.2f}", fg=c)
        self.info_lbl.config(text=f"cogito: {s.cogito_count}  |  coherence: {s.coherence_score:.2f}  |  aufhebung: {s.synthesis_count}")

    # ---- Auto-Speak ----

    def _toggle_auto(self):
        self.auto_speak_enabled = not self.auto_speak_enabled
        if self.auto_speak_enabled:
            self.auto_btn.config(text="Auto-Speak: ON", bg=Theme.LOCK_ON)
            self._schedule_auto()
            self._sys_msg("Auto-Speak enabled. AI will speak on its own.")
        else:
            self.auto_btn.config(text="Auto-Speak: OFF", bg=Theme.LOCK_OFF)
            if self.auto_speak_timer:
                self.after_cancel(self.auto_speak_timer)
                self.auto_speak_timer = None
            self._sys_msg("Auto-Speak disabled.")

    def _schedule_auto(self):
        if not self.auto_speak_enabled: return
        ms = random.randint(AUTO_SPEAK_MIN, AUTO_SPEAK_MAX) * 1000
        self.auto_speak_timer = self.after(ms, self._auto_tick)

    def _auto_tick(self):
        if not self.auto_speak_enabled or self.processing:
            self._schedule_auto(); return
        self.processing = True
        self._sys_msg("AI is thinking...")
        threading.Thread(target=self._auto_thread, daemon=True).start()

    def _auto_thread(self):
        try:
            msg = self.bot.generate_autonomous_message()
            self.after(0, self._on_auto, msg)
        except Exception as e:
            self.after(0, self._on_auto_error, str(e))

    def _on_auto(self, msg):
        self._add_msg(msg, "ai")
        self._update_panel()
        self.processing = False
        self._schedule_auto()

    # ---- Handlers ----

    def _on_enter(self, e):
        if not e.state & 0x1: self._send(); return "break"

    def _send(self):
        if self.processing: return
        text = self.input_field.get("1.0", "end").strip()
        if not text: return
        self.input_field.delete("1.0", "end")
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
            self._event_msg(f"Aufhebung — qualitative transition (total: {r['synthesis_total']})",
                           Theme.EMO_CONFUSION)
        if r.get("adjustments") and r["adjustments"].get("reason"):
            self._event_msg(f"Meta-cognition: {r['adjustments']['reason']}", Theme.ACCENT_LIGHT)
        self._update_panel()
        self.processing = False
        self.send_btn.config(state="normal", text="Send")
        self.input_field.focus()

    def _on_error(self, msg):
        self._sys_msg(f"Error: {msg}")
        self.processing = False
        self.send_btn.config(state="normal", text="Send")
        self.input_field.focus()

    def _on_auto_error(self, msg):
        self._sys_msg(f"Auto-speak error: {msg}")
        self.processing = False
        self._schedule_auto()

    def _greet(self):
        def run():
            try:
                g = self.bot.generate_greeting()
                self.after(0, lambda: (self._add_msg(g, "ai"), self._update_panel()))
            except Exception as e:
                self.after(0, lambda: self._sys_msg(f"Greeting failed: {e}"))
        threading.Thread(target=run, daemon=True).start()

    def _reset(self):
        if messagebox.askyesno("Reset", "Clear all conversation history and reset self-image?\nContinue?"):
            self.auto_speak_enabled = False
            self.auto_btn.config(text="Auto-Speak: OFF", bg=Theme.LOCK_OFF)
            if self.auto_speak_timer:
                self.after_cancel(self.auto_speak_timer)
                self.auto_speak_timer = None
            self.bot.reset()
            for w in self.chat_inner.winfo_children(): w.destroy()
            self._update_panel()
            self._sys_msg("State has been reset.")
            self.after(500, self._greet)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    ChatApp().mainloop()
