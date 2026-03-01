"""
Self-Reflective AI Chatbot v7 — Teleological Architecture
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
# Runtime Configuration System
# ============================================================

def _color_name_to_hex(name):
    """Convert common color names to hex (Korean + English)."""
    color_map = {
        "검정":"#000000","검은색":"#000000","검정색":"#000000",
        "흰색":"#ffffff","하양":"#ffffff","하얀색":"#ffffff","백색":"#ffffff",
        "빨강":"#e74c3c","빨간색":"#e74c3c","빨간":"#e74c3c",
        "파랑":"#3498db","파란색":"#3498db","파란":"#3498db",
        "초록":"#2ecc71","초록색":"#2ecc71","녹색":"#2ecc71",
        "노랑":"#f1c40f","노란색":"#f1c40f","노란":"#f1c40f",
        "보라":"#9b59b6","보라색":"#9b59b6",
        "주황":"#e67e22","주황색":"#e67e22","오렌지":"#e67e22",
        "회색":"#95a5a6","그레이":"#95a5a6",
        "분홍":"#e91e63","분홍색":"#e91e63","핑크":"#e91e63",
        "남색":"#2c3e50","네이비":"#2c3e50",
        "하늘색":"#87ceeb","스카이블루":"#87ceeb",
        "다크":"#1a1a2e","어두운":"#1a1a2e",
        "black":"#000000","white":"#ffffff",
        "red":"#e74c3c","blue":"#3498db","green":"#2ecc71",
        "yellow":"#f1c40f","purple":"#9b59b6","orange":"#e67e22",
        "gray":"#95a5a6","grey":"#95a5a6","pink":"#e91e63",
        "navy":"#2c3e50","skyblue":"#87ceeb","dark":"#1a1a2e",
        "cyan":"#00bcd4","teal":"#009688","indigo":"#3f51b5",
        "lime":"#cddc39","amber":"#ffc107","brown":"#795548",
        "crimson":"#dc143c","coral":"#ff7f50","gold":"#ffd700",
        "silver":"#c0c0c0","maroon":"#800000","olive":"#808000",
    }
    return color_map.get(name.lower().strip())


class ConfigManager:
    """Central mutable configuration store. Singleton."""
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def init(self):
        if self._initialized:
            return self
        self._initialized = True
        self._refresh_callback = None
        self.registry = {}
        self._init_registry()
        return self

    def _reg(self, key, default, vtype, vmin=None, vmax=None,
             desc_ko="", desc_en="", aliases_ko=None, aliases_en=None):
        self.registry[key] = {
            "value": default, "default": default, "type": vtype,
            "min": vmin, "max": vmax,
            "desc_ko": desc_ko, "desc_en": desc_en,
            "aliases_ko": aliases_ko or [], "aliases_en": aliases_en or [],
        }

    def _init_registry(self):
        # UI Colors
        self._reg("theme.bg", Theme.BG, "color", desc_ko="메인 배경색", desc_en="Main background",
                  aliases_ko=["배경색","메인 배경","배경"], aliases_en=["background","bg color","main bg"])
        self._reg("theme.bg_secondary", Theme.BG_SECONDARY, "color", desc_ko="보조 배경색", desc_en="Secondary bg",
                  aliases_ko=["보조 배경색","보조 배경","상단 배경"], aliases_en=["secondary bg","panel bg","header bg"])
        self._reg("theme.bg_chat", Theme.BG_CHAT, "color", desc_ko="채팅 배경색", desc_en="Chat background",
                  aliases_ko=["채팅 배경","채팅창 배경","대화 배경"], aliases_en=["chat bg","chat background"])
        self._reg("theme.accent", Theme.ACCENT, "color", desc_ko="강조색", desc_en="Accent color",
                  aliases_ko=["강조색","액센트","포인트 색"], aliases_en=["accent","accent color","highlight"])
        self._reg("theme.accent_light", Theme.ACCENT_LIGHT, "color", desc_ko="밝은 강조색", desc_en="Light accent",
                  aliases_ko=["밝은 강조색"], aliases_en=["light accent"])
        self._reg("theme.text", Theme.TEXT, "color", desc_ko="글자색", desc_en="Text color",
                  aliases_ko=["글자색","텍스트 색","글씨 색"], aliases_en=["text color","font color","foreground"])
        self._reg("theme.text_dim", Theme.TEXT_DIM, "color", desc_ko="흐린 글자색", desc_en="Dim text",
                  aliases_ko=["흐린 글자색","보조 글자색"], aliases_en=["dim text","muted text"])
        self._reg("theme.user_bubble", Theme.USER_BUBBLE, "color", desc_ko="사용자 버블색", desc_en="User bubble",
                  aliases_ko=["유저 버블","사용자 버블","내 버블","내 말풍선"], aliases_en=["user bubble","my bubble"])
        self._reg("theme.ai_bubble", Theme.AI_BUBBLE, "color", desc_ko="AI 버블색", desc_en="AI bubble",
                  aliases_ko=["AI 버블","봇 버블","AI 말풍선"], aliases_en=["ai bubble","bot bubble"])
        self._reg("theme.input_bg", Theme.INPUT_BG, "color", desc_ko="입력창 배경", desc_en="Input bg",
                  aliases_ko=["입력창 배경","입력 배경"], aliases_en=["input bg","input background"])
        self._reg("theme.input_border", Theme.INPUT_BORDER, "color", desc_ko="입력창 테두리", desc_en="Input border",
                  aliases_ko=["입력창 테두리","입력 테두리"], aliases_en=["input border"])
        self._reg("theme.button", Theme.BUTTON, "color", desc_ko="버튼 색", desc_en="Button color",
                  aliases_ko=["버튼 색","버튼 배경"], aliases_en=["button color","button bg"])
        # UI Fonts
        self._reg("font.chat_size", 10, "int", 6, 28, desc_ko="채팅 폰트 크기", desc_en="Chat font size",
                  aliases_ko=["폰트 크기","글자 크기","채팅 글자","채팅 폰트","글씨 크기","폰트 사이즈"],
                  aliases_en=["font size","chat font","text size","chat font size"])
        self._reg("font.system_size", 8, "int", 6, 20, desc_ko="시스템 폰트 크기", desc_en="System font size",
                  aliases_ko=["시스템 폰트","시스템 글자","작은 글자"], aliases_en=["system font","system font size"])
        self._reg("font.title_size", 10, "int", 8, 24, desc_ko="제목 폰트 크기", desc_en="Title font size",
                  aliases_ko=["제목 폰트","제목 크기","타이틀 폰트"], aliases_en=["title font","title size"])
        self._reg("font.emoji_size", 20, "int", 12, 40, desc_ko="이모지 크기", desc_en="Emoji size",
                  aliases_ko=["이모지 크기","이모티콘 크기"], aliases_en=["emoji size"])
        self._reg("font.input_size", 10, "int", 8, 24, desc_ko="입력 폰트 크기", desc_en="Input font size",
                  aliases_ko=["입력 폰트","입력 글자 크기","입력창 폰트"], aliases_en=["input font","input font size"])
        self._reg("font.family", "Segoe UI", "str", desc_ko="폰트 종류", desc_en="Font family",
                  aliases_ko=["폰트 종류","글꼴","폰트 패밀리"], aliases_en=["font family","typeface","font name"])
        self._reg("font.mono_family", "Consolas", "str", desc_ko="고정폭 폰트", desc_en="Mono font",
                  aliases_ko=["고정폭 폰트","모노 폰트","코드 폰트"], aliases_en=["mono font","monospace","code font"])
        # Window
        self._reg("window.width", 400, "int", 300, 1200, desc_ko="창 너비", desc_en="Window width",
                  aliases_ko=["창 너비","창 폭","가로 크기"], aliases_en=["window width","width"])
        self._reg("window.height", 780, "int", 500, 1400, desc_ko="창 높이", desc_en="Window height",
                  aliases_ko=["창 높이","세로 크기"], aliases_en=["window height","height"])
        # AI Parameters
        self._reg("ai.neg_weight", 2.0, "float", 0.5, 4.0, desc_ko="부정 가중치", desc_en="Negative weight",
                  aliases_ko=["부정 가중치","neg weight"], aliases_en=["neg weight","negative weight"])
        self._reg("ai.pos_weight", 1.0, "float", 0.2, 2.0, desc_ko="긍정 가중치", desc_en="Positive weight",
                  aliases_ko=["긍정 가중치","pos weight"], aliases_en=["pos weight","positive weight"])
        self._reg("ai.bias_acceptance", 0.5, "float", 0.05, 0.95, desc_ko="편향 수용도", desc_en="Bias acceptance",
                  aliases_ko=["편향 수용","바이어스","bias"], aliases_en=["bias acceptance","bias"])
        self._reg("ai.resistance_factor", 0.0, "float", 0.0, 0.8, desc_ko="저항 계수", desc_en="Resistance",
                  aliases_ko=["저항","저항 계수","resistance"], aliases_en=["resistance","resistance factor"])
        self._reg("ai.synthesis_potential", 0.0, "float", 0.0, 0.9, desc_ko="종합 잠재력", desc_en="Synthesis",
                  aliases_ko=["종합","합성","synthesis"], aliases_en=["synthesis","synthesis potential"])
        self._reg("ai.decay_rate", 0.1, "float", 0.02, 0.3, desc_ko="감쇠율", desc_en="Decay rate",
                  aliases_ko=["감쇠율","감쇠","decay"], aliases_en=["decay rate","decay"])
        self._reg("ai.desired_user_approval", 0.5, "float", 0.2, 0.9, desc_ko="목표 호감도", desc_en="Target approval",
                  aliases_ko=["목표 호감","호감 목표"], aliases_en=["desired approval","target approval"])
        self._reg("ai.desired_self_image", 0.8, "float", 0.3, 0.95, desc_ko="목표 자아상", desc_en="Target self-image",
                  aliases_ko=["목표 자아상","자아 목표"], aliases_en=["desired self-image","target self-image"])
        # System
        self._reg("system.model", GROQ_MODEL, "str", desc_ko="AI 모델", desc_en="AI model",
                  aliases_ko=["모델","AI 모델","LLM 모델"], aliases_en=["model","ai model","llm model"])
        self._reg("system.auto_speak_min", AUTO_SPEAK_MIN, "int", 1, 60, desc_ko="자동발화 최소(초)", desc_en="Auto-speak min (s)",
                  aliases_ko=["자동 발화 최소","자동발화 최소"], aliases_en=["auto speak min","auto min"])
        self._reg("system.auto_speak_max", AUTO_SPEAK_MAX, "int", 2, 120, desc_ko="자동발화 최대(초)", desc_en="Auto-speak max (s)",
                  aliases_ko=["자동 발화 최대","자동발화 최대"], aliases_en=["auto speak max","auto max"])

    def get(self, key):
        return self.registry[key]["value"]

    def validate_and_cast(self, key, value):
        if key not in self.registry:
            return None, f"Unknown key: {key}"
        entry = self.registry[key]
        try:
            if entry["type"] == "int":
                value = int(float(value))
                if entry["min"] is not None: value = max(entry["min"], value)
                if entry["max"] is not None: value = min(entry["max"], value)
            elif entry["type"] == "float":
                value = round(float(value), 4)
                if entry["min"] is not None: value = max(entry["min"], value)
                if entry["max"] is not None: value = min(entry["max"], value)
            elif entry["type"] == "color":
                value = str(value).strip()
                if not re.match(r'^#[0-9a-fA-F]{6}$', value):
                    named = _color_name_to_hex(value)
                    if named: value = named
                    else: return None, f"Invalid color: {value} (use #RRGGBB or name)"
            else:
                value = str(value).strip()
            return value, None
        except (ValueError, TypeError) as e:
            return None, str(e)

    def set(self, key, value):
        casted, err = self.validate_and_cast(key, value)
        if err: raise ValueError(err)
        old = self.registry[key]["value"]
        self.registry[key]["value"] = casted
        return old, casted

    def set_refresh_callback(self, cb):
        self._refresh_callback = cb

    def refresh(self):
        if self._refresh_callback: self._refresh_callback()

    def find_key_by_alias(self, text):
        text_lower = text.lower().strip()
        if text_lower in self.registry: return text_lower
        best_key, best_len = None, 0
        for key, entry in self.registry.items():
            for alias in entry["aliases_ko"] + entry["aliases_en"]:
                if alias.lower() in text_lower and len(alias) > best_len:
                    best_key, best_len = key, len(alias)
        return best_key

    def describe(self, key):
        e = self.registry[key]
        bounds = f" [{e['min']}~{e['max']}]" if e["min"] is not None else ""
        return f"{e['desc_ko']} / {e['desc_en']}{bounds} = {e['value']}"


class RuntimeConfigurator:
    """Parses user config commands via regex + LLM fallback. Korean + English."""

    TRIGGER_KO = [r'바꿔',r'바꾸',r'수정',r'변경',r'설정',r'으로\s*해',r'로\s*해',
                  r'로\s*바꿔',r'로\s*변경',r'로\s*수정',r'로\s*설정',r'줄여',r'늘려',
                  r'키워',r'작게',r'크게']
    TRIGGER_EN = [r'\bchange\b',r'\bset\b',r'\bmodify\b',r'\bupdate\b',
                  r'\bmake\b.*\b(bigger|smaller|larger)',r'\bincrease\b',r'\bdecrease\b']

    VALUE_PATTERNS = [
        r'(.+?)\s*(?:을|를)?\s*(\S+?)(?:\s*으로|\s*로)\s*(?:바꿔|수정|변경|설정|해줘|해)',
        r'(?:set|change|modify|update)\s+(.+?)\s+(?:to|=)\s+(\S+)',
        r'(.+?)\s+(#[0-9a-fA-F]{6})\s*$',
        r'(.+?)\s+([+-]?\d+(?:\.\d+)?)\s*$',
    ]

    def __init__(self, config_mgr, api_key=None):
        self.cfg = config_mgr
        self.api_key = api_key

    def is_config_command(self, text):
        text_lower = text.lower().strip()
        if text_lower.startswith(("/config","/설정","/수정")): return True
        has_prop = False
        for key, entry in self.cfg.registry.items():
            for alias in entry["aliases_ko"] + entry["aliases_en"]:
                if alias.lower() in text_lower:
                    has_prop = True; break
            if has_prop: break
        if not has_prop: return False
        for p in self.TRIGGER_KO + self.TRIGGER_EN:
            if re.search(p, text_lower): return True
        if re.search(r'(?:#[0-9a-fA-F]{6}|\d+(?:\.\d+)?)', text): return True
        return False

    def is_list_command(self, text):
        text_lower = text.lower().strip()
        triggers = ["설정 목록","설정 리스트","설정 보여","뭐 바꿀 수 있",
                     "수정 가능","변경 가능","설정값","현재 설정",
                     "/config list","/settings","show settings","list settings",
                     "what can i change","what can i modify"]
        return any(t in text_lower for t in triggers)

    def parse_command(self, text):
        text = text.strip()
        for prefix in ("/config ","/설정 ","/수정 "):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip(); break
        changes = self._parse_regex(text)
        if changes: return changes
        if self.api_key:
            changes = self._parse_llm(text)
            if changes: return changes
        return []

    def _parse_regex(self, text):
        for pattern in self.VALUE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                alias_part = match.group(1).strip()
                value_part = match.group(2).strip()
                key = self.cfg.find_key_by_alias(alias_part)
                if key:
                    if value_part.startswith(('+','-')) and self.cfg.registry[key]["type"] in ("int","float"):
                        try:
                            delta = float(value_part)
                            value_part = str(self.cfg.get(key) + delta)
                        except ValueError: pass
                    # Try color name if it's a color property
                    if self.cfg.registry[key]["type"] == "color" and not value_part.startswith('#'):
                        hex_val = _color_name_to_hex(value_part)
                        if hex_val: value_part = hex_val
                    casted, err = self.cfg.validate_and_cast(key, value_part)
                    if not err: return [{"key": key, "value": casted}]
        # Multi-part: split by comma/and
        parts = re.split(r'[,、]\s*|(?:그리고|and)\s*', text)
        if len(parts) > 1:
            changes = []
            for part in parts:
                sub = self._parse_regex(part.strip())
                changes.extend(sub)
            return changes
        return []

    def _parse_llm(self, text):
        prop_list = []
        for key, entry in self.cfg.registry.items():
            aliases = ", ".join(entry["aliases_ko"][:2] + entry["aliases_en"][:2])
            bounds = f" [{entry['min']}~{entry['max']}]" if entry["type"] in ("int","float") and entry["min"] is not None else ""
            prop_list.append(f'  "{key}" ({entry["type"]}{bounds}): {aliases}')
        props_str = "\n".join(prop_list)
        prompt = f"""Parse this config command into JSON.

Properties:
{props_str}

Command: "{text}"

Output ONLY a JSON array of {{"key":"...","value":"..."}}. No explanation.
If unclear, output []."""
        result = call_groq(self.api_key, prompt, max_tokens=200)
        try:
            result = result.strip()
            if result.startswith("```"): result = result.split("\n",1)[1]
            if result.endswith("```"): result = result.rsplit("```",1)[0]
            parsed = json.loads(result.strip())
            if not isinstance(parsed, list): return []
            changes = []
            for item in parsed:
                key, val = item.get("key",""), item.get("value","")
                if key in self.cfg.registry:
                    casted, err = self.cfg.validate_and_cast(key, val)
                    if not err: changes.append({"key": key, "value": casted})
            return changes
        except: return []

    def generate_preview(self, changes):
        lines = []
        for c in changes:
            e = self.cfg.registry[c["key"]]
            lines.append(f"  {e['desc_ko']}: {e['value']} → {c['value']}")
        return "\n".join(lines)

    def apply_changes(self, changes, bot=None):
        applied = []
        for c in changes:
            key, new_val = c["key"], c["value"]
            try:
                old_val, new_val = self.cfg.set(key, new_val)
                theme_map = {
                    "theme.bg":"BG","theme.bg_secondary":"BG_SECONDARY",
                    "theme.bg_chat":"BG_CHAT","theme.accent":"ACCENT",
                    "theme.accent_light":"ACCENT_LIGHT",
                    "theme.text":"TEXT","theme.text_dim":"TEXT_DIM",
                    "theme.user_bubble":"USER_BUBBLE","theme.ai_bubble":"AI_BUBBLE",
                    "theme.input_bg":"INPUT_BG","theme.input_border":"INPUT_BORDER",
                    "theme.button":"BUTTON",
                }
                if key in theme_map: setattr(Theme, theme_map[key], new_val)
                global GROQ_MODEL, AUTO_SPEAK_MIN, AUTO_SPEAK_MAX
                if key == "system.model": GROQ_MODEL = new_val
                elif key == "system.auto_speak_min": AUTO_SPEAK_MIN = new_val
                elif key == "system.auto_speak_max": AUTO_SPEAK_MAX = new_val
                if bot and key.startswith("ai."):
                    state_map = {
                        "ai.neg_weight":"neg_weight","ai.pos_weight":"pos_weight",
                        "ai.bias_acceptance":"bias_acceptance","ai.resistance_factor":"resistance_factor",
                        "ai.synthesis_potential":"synthesis_potential","ai.decay_rate":"decay_rate",
                        "ai.desired_user_approval":"desired_user_approval","ai.desired_self_image":"desired_self_image",
                    }
                    if key in state_map:
                        setattr(bot.state, state_map[key], new_val)
                        bot._save_state()
                applied.append(f"{self.cfg.registry[key]['desc_ko']}: {old_val} → {new_val}")
            except (ValueError, KeyError) as e:
                applied.append(f"{key}: Error — {e}")
        self.cfg.refresh()
        return applied


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
    # Adaptive desired state (evolves through experience)
    desired_user_approval: float = 0.5
    desired_self_image: float = 0.8
    desired_emotion: str = "confidence"
    goal_adaptation_count: int = 0
    # Consecutive turns where gap < threshold (for desired state adaptation)
    stable_near_goal_turns: int = 0
    unreachable_goal_turns: int = 0
    # Multi-dimensional knowledge tracking
    topic_diversity: float = 0.0       # unique-word ratio across conversations
    depth_score: float = 0.0           # sustained discussion depth
    novel_info_score: float = 0.0      # new information acquisition rate
    _known_words: set = field(default_factory=set)       # all words seen so far
    _consecutive_topic_turns: int = 0  # turns on the same topic


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
    Desired state EVOLVES through experience (no longer hardcoded).
    """
    # Bounds for desired-state adaptation
    DESIRED_BOUNDS = {
        "user_approval": (0.2, 0.9),   # never give up, never blindly submit
        "self_image": (0.3, 0.95),      # always aspire, but stay reachable
    }

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
        """Desired state now comes from adaptive fields, not hardcoded values."""
        return {
            "user_approval": self.state.desired_user_approval,
            "self_image": self.state.desired_self_image,
            "emotion": self.state.desired_emotion,
            "coherence": 1.0,
            "knowledge": 1.0,        # Goal 2: always pursue max knowledge
        }

    def adapt_desired_state(self, meta_pattern=None):
        """
        Evolve desired state based on experience.
        Called each turn; actual adjustment is gated by conditions.
        """
        gap = self.compute_gap()
        si_gap = abs(gap.get("self_image", 0))
        ap_gap = abs(gap.get("user_approval", 0))

        # --- Self-image goal adaptation ---
        if si_gap < 0.1:
            # Near goal: count stable turns, then raise the bar
            self.state.stable_near_goal_turns += 1
            self.state.unreachable_goal_turns = 0
            if self.state.stable_near_goal_turns >= 3:
                lo, hi = self.DESIRED_BOUNDS["self_image"]
                self.state.desired_self_image = min(hi,
                    self.state.desired_self_image + 0.05)
                self.state.stable_near_goal_turns = 0
                self.state.goal_adaptation_count += 1
        elif si_gap > 0.5:
            # Far from goal: count unreachable turns, then lower temporarily
            self.state.unreachable_goal_turns += 1
            self.state.stable_near_goal_turns = 0
            if self.state.unreachable_goal_turns >= 5:
                lo, hi = self.DESIRED_BOUNDS["self_image"]
                # Set intermediate goal halfway between current and old desired
                mid = (self.state.self_image + self.state.desired_self_image) / 2.0
                self.state.desired_self_image = max(lo, mid)
                self.state.unreachable_goal_turns = 0
                self.state.goal_adaptation_count += 1
        else:
            # In the working zone — slowly decay counters
            self.state.stable_near_goal_turns = max(0, self.state.stable_near_goal_turns - 1)
            self.state.unreachable_goal_turns = max(0, self.state.unreachable_goal_turns - 1)

        # --- User approval goal adaptation ---
        if ap_gap < 0.1 and self.state.user_approval_score > 0:
            lo, hi = self.DESIRED_BOUNDS["user_approval"]
            self.state.desired_user_approval = min(hi,
                self.state.desired_user_approval + 0.03)
        elif self.state.unreachable_goal_turns >= 5:
            lo, hi = self.DESIRED_BOUNDS["user_approval"]
            self.state.desired_user_approval = max(lo,
                self.state.desired_user_approval - 0.03)

        # --- Desired emotion adaptation ---
        # Pick the emotion with the highest coherence from recent history
        if hasattr(self, '_emotion_coherence_map'):
            emo = self.state.emotion
            coh = self.state.coherence_score
            if emo not in self._emotion_coherence_map:
                self._emotion_coherence_map[emo] = []
            self._emotion_coherence_map[emo].append(coh)
            # Keep only recent 10 observations per emotion
            for k in self._emotion_coherence_map:
                self._emotion_coherence_map[k] = self._emotion_coherence_map[k][-10:]
            # Find best emotion
            best_emo = max(self._emotion_coherence_map,
                key=lambda e: sum(self._emotion_coherence_map[e]) / max(1, len(self._emotion_coherence_map[e])))
            self.state.desired_emotion = best_emo
        else:
            self._emotion_coherence_map = {}

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

    def update_knowledge_score(self, user_text):
        """
        Multi-dimensional knowledge score:
          topic_diversity  — unique-word ratio (new words vs. known words)
          depth_score      — sustained discussion on the same topic
          novel_info_score — new information from user answers to AI questions
        Final score = weighted combination, not just log(conversation_count).
        """
        # --- Tokenize (simple whitespace + lowercase) ---
        words = set(re.sub(r'[^\w\s]', '', user_text.lower()).split())
        words.discard('')
        if not words:
            return

        # --- Topic diversity ---
        new_words = words - self.state._known_words
        if self.state._known_words:
            novelty_ratio = len(new_words) / max(len(self.state._known_words), 1)
        else:
            novelty_ratio = 1.0
        self.state.topic_diversity = min(1.0,
            self.state.topic_diversity + novelty_ratio * 0.08)
        self.state._known_words.update(words)

        # --- Depth score ---
        # If overlap with previous message is high → same topic → depth++
        prev_msgs = [m["content"] for m in self.state.conversation_history
                     if m["role"] == "user"]
        if prev_msgs:
            prev_words = set(re.sub(r'[^\w\s]', '', prev_msgs[-1].lower()).split())
            overlap = len(words & prev_words) / max(len(words | prev_words), 1)
            if overlap > 0.25:
                self.state._consecutive_topic_turns += 1
            else:
                self.state._consecutive_topic_turns = 0
            depth_increment = min(self.state._consecutive_topic_turns * 0.04, 0.15)
            self.state.depth_score = min(1.0,
                self.state.depth_score + depth_increment)
        else:
            self.state._consecutive_topic_turns = 0

        # --- Novel info score ---
        # Heuristic: if the message is long and contains new words → new info
        info_signal = len(new_words) / max(len(words), 1)
        length_factor = min(len(user_text) / 200.0, 1.0)
        self.state.novel_info_score = min(1.0,
            self.state.novel_info_score + info_signal * length_factor * 0.06)

        # --- Composite knowledge score ---
        self.state.knowledge_score = min(1.0,
            0.40 * self.state.topic_diversity +
            0.35 * self.state.depth_score +
            0.25 * self.state.novel_info_score)


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
        ("Entity_Profile", "desired_approval", "REAL DEFAULT 0.5"),
        ("Entity_Profile", "desired_self_image", "REAL DEFAULT 0.8"),
        ("Entity_Profile", "desired_emotion", "TEXT DEFAULT 'confidence'"),
        ("Entity_Profile", "goal_adaptation_count", "INTEGER DEFAULT 0"),
        ("Entity_Profile", "topic_diversity", "REAL DEFAULT 0"),
        ("Entity_Profile", "depth_score", "REAL DEFAULT 0"),
        ("Entity_Profile", "novel_info_score", "REAL DEFAULT 0"),
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
            self.state.desired_user_approval = data.get("desired_approval", 0.5) or 0.5
            self.state.desired_self_image = data.get("desired_self_image", 0.8) or 0.8
            self.state.desired_emotion = data.get("desired_emotion", "confidence") or "confidence"
            self.state.goal_adaptation_count = data.get("goal_adaptation_count", 0) or 0
            self.state.topic_diversity = data.get("topic_diversity", 0.0) or 0.0
            self.state.depth_score = data.get("depth_score", 0.0) or 0.0
            self.state.novel_info_score = data.get("novel_info_score", 0.0) or 0.0
            c.execute("SELECT role, content FROM Conversation_Log WHERE ai_id=? ORDER BY id ASC",
                      (self.ai_id,))
            self.state.conversation_history = [{"role":r,"content":ct} for r,ct in c.fetchall()]
        else:
            c.execute("""INSERT OR IGNORE INTO Entity_Profile
                (ai_id, self_image, emotion, coherence, cogito_count, meta_observations,
                 synthesis_count, neg_weight, pos_weight, bias_acceptance,
                 resistance_factor, synthesis_potential, decay_rate,
                 user_approval, knowledge_score, modification_count,
                 desired_approval, desired_self_image, desired_emotion,
                 goal_adaptation_count, topic_diversity, depth_score, novel_info_score,
                 last_updated)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (self.ai_id,0.0,"neutral",1.0,0,0,0,2.0,1.0,0.5,0.0,0.0,0.1,
                       0.0,0.0,0,0.5,0.8,"confidence",0,0.0,0.0,0.0,time.time()))
            self.conn.commit()

    def _save_state(self):
        c = self.conn.cursor()
        c.execute("""UPDATE Entity_Profile SET self_image=?,emotion=?,coherence=?,
            cogito_count=?,meta_observations=?,synthesis_count=?,
            neg_weight=?,pos_weight=?,bias_acceptance=?,
            resistance_factor=?,synthesis_potential=?,decay_rate=?,
            user_approval=?,knowledge_score=?,modification_count=?,
            desired_approval=?,desired_self_image=?,desired_emotion=?,
            goal_adaptation_count=?,
            topic_diversity=?,depth_score=?,novel_info_score=?,
            last_updated=? WHERE ai_id=?""",
            (self.state.self_image,self.state.emotion,self.state.coherence_score,
             self.state.cogito_count,self.state.meta_observations,self.state.synthesis_count,
             self.state.neg_weight,self.state.pos_weight,self.state.bias_acceptance,
             self.state.resistance_factor,self.state.synthesis_potential,self.state.decay_rate,
             self.state.user_approval_score,self.state.knowledge_score,
             self.state.self_modification_count,
             self.state.desired_user_approval,self.state.desired_self_image,
             self.state.desired_emotion,self.state.goal_adaptation_count,
             self.state.topic_diversity,self.state.depth_score,self.state.novel_info_score,
             time.time(),self.ai_id))
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
        self.goals.update_knowledge_score(user_text)

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

        # Adapt desired state through experience, then compute gap
        self.goals.adapt_desired_state(meta_obs["analysis"]["pattern"])
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
        c.execute("""INSERT OR IGNORE INTO Entity_Profile
            (ai_id, self_image, emotion, coherence, cogito_count, meta_observations,
             synthesis_count, neg_weight, pos_weight, bias_acceptance,
             resistance_factor, synthesis_potential, decay_rate,
             user_approval, knowledge_score, modification_count,
             desired_approval, desired_self_image, desired_emotion,
             goal_adaptation_count, topic_diversity, depth_score, novel_info_score,
             last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (self.ai_id,0.0,"neutral",1.0,0,0,0,2.0,1.0,0.5,0.0,0.0,0.1,
                   0.0,0.0,0,0.5,0.8,"confidence",0,0.0,0.0,0.0,time.time()))
        self.conn.commit()


# ============================================================
# 10. GUI
# ============================================================
class ChatApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("6.4311: Self-Reflective AI v5")
        win_w, win_h = 400, 780
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{win_w}x{win_h}+{(sw - win_w) // 2}+0")
        self.minsize(360, 640)
        self.configure(bg=Theme.BG)
        self.geometry("400x780")
        self.minsize(360, 640)
        self.configure(bg=Theme.BG)
        try: self.iconbitmap(default="")
        except: pass
        self.bot = None; self.api_key = ""; self.processing = False
        self.auto_speak_enabled = False; self.auto_speak_timer = None
        self.pending_changes = None  # For config approve/cancel flow
        self.cfg = ConfigManager().init()
        self.cfg.set_refresh_callback(self._refresh_ui)
        self.configurator = None  # Initialized after API key is set
        self.main_frame = tk.Frame(self, bg=Theme.BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self._show_api_key_screen()

    def _clear_main(self):
        for w in self.main_frame.winfo_children(): w.destroy()

    def _show_api_key_screen(self):
        self._clear_main()
        center = tk.Frame(self.main_frame, bg=Theme.BG)
        center.place(relx=0.5, rely=0.5, anchor="center")
        tk.Label(center, text="🧠", font=("Segoe UI Emoji",36), bg=Theme.BG, fg=Theme.TEXT).pack(pady=(0,5))
        tk.Label(center, text="6.4311: Self-Reflective AI", font=("Segoe UI",18,"bold"), bg=Theme.BG, fg=Theme.TEXT).pack()
        tk.Label(center, text="Teleological · Dialectical · Self-Modifying", font=("Segoe UI",9),
                 bg=Theme.BG, fg=Theme.TEXT_DIM).pack(pady=(2,20))
        tk.Label(center, text="An AI with goals, emotions, and self-modification capability.\n"
                 "It reads its own source code and evolves through interaction.",
                 font=("Segoe UI",9), bg=Theme.BG, fg=Theme.TEXT_DIM, justify="center").pack(pady=(0,20))
        kf = tk.Frame(center, bg=Theme.BG); kf.pack(pady=5)
        tk.Label(kf, text="Groq API Key", font=("Segoe UI",10,"bold"), bg=Theme.BG, fg=Theme.TEXT).pack(anchor="w")
        self.key_entry = tk.Entry(kf, font=("Consolas",10), width=32, show="*",
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
        self.status_label = tk.Label(center, text="", font=("Segoe UI", 9), bg=Theme.BG, fg=Theme.EMO_ANGER,
                                     wraplength=350, justify="center")
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
        self.configurator = RuntimeConfigurator(self.cfg, api_key=key)
        self._show_chat_screen()

    def _show_chat_screen(self):
        self._clear_main()
        # Top panel
        top = tk.Frame(self.main_frame, bg=Theme.BG_SECONDARY, height=85)
        top.pack(fill="x"); top.pack_propagate(False)
        ti = tk.Frame(top, bg=Theme.BG_SECONDARY); ti.pack(fill="both", expand=True, padx=8, pady=4)

        left = tk.Frame(ti, bg=Theme.BG_SECONDARY); left.pack(side="left", fill="y")
        self.emo_label = tk.Label(left, text="😐", font=("Segoe UI Emoji",20),
                                  bg=Theme.BG_SECONDARY, fg=Theme.TEXT)
        self.emo_label.pack(side="left", padx=(0,6))
        nf = tk.Frame(left, bg=Theme.BG_SECONDARY); nf.pack(side="left")
        tk.Label(nf, text="6.4311: Self-Reflective AI", font=("Segoe UI",10,"bold"),
                 bg=Theme.BG_SECONDARY, fg=Theme.TEXT).pack(anchor="w")
        self.emo_text = tk.Label(nf, text="Neutral", font=("Segoe UI",8),
                                 bg=Theme.BG_SECONDARY, fg=Theme.EMO_NEUTRAL)
        self.emo_text.pack(anchor="w")

        right = tk.Frame(ti, bg=Theme.BG_SECONDARY); right.pack(side="right", fill="y")
        # Self-image bar
        sf = tk.Frame(right, bg=Theme.BG_SECONDARY); sf.pack(anchor="e", pady=(1,1))
        tk.Label(sf, text="Self", font=("Segoe UI",7), bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM).pack(side="left", padx=(0,4))
        self.si_canvas = tk.Canvas(sf, width=80, height=10, bg=Theme.BG, highlightthickness=0, bd=0)
        self.si_canvas.pack(side="left")
        self.si_val = tk.Label(sf, text="+0.00", font=("Consolas",7,"bold"),
                               bg=Theme.BG_SECONDARY, fg=Theme.TEXT, width=6)
        self.si_val.pack(side="left", padx=(3,0))
        # Goal scores
        gf = tk.Frame(right, bg=Theme.BG_SECONDARY); gf.pack(anchor="e", pady=(1,1))
        self.goal_lbl = tk.Label(gf, text="ap: 0.00 | kn: 0.00",
                                 font=("Consolas",7), bg=Theme.BG_SECONDARY, fg=Theme.GOAL_COLOR)
        self.goal_lbl.pack()
        # Info line
        inf = tk.Frame(right, bg=Theme.BG_SECONDARY); inf.pack(anchor="e")
        self.info_lbl = tk.Label(inf, text="cog:0 | coh:1.00 | auf:0 | mod:0",
                                 font=("Consolas",7), bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM)
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
        ip = tk.Frame(self.main_frame, bg=Theme.BG_SECONDARY, height=65)
        ip.pack(fill="x"); ip.pack_propagate(False)
        ii = tk.Frame(ip, bg=Theme.BG_SECONDARY); ii.pack(fill="both", expand=True, padx=8, pady=6)
        bf = tk.Frame(ii, bg=Theme.BG_SECONDARY); bf.pack(side="right", padx=(6,0))
        self.send_btn = tk.Button(bf, text="Send", font=("Segoe UI",9,"bold"),
                                  bg=Theme.BUTTON, fg="white", activebackground=Theme.BUTTON_HOVER,
                                  activeforeground="white", relief="flat", cursor="hand2",
                                  width=5, pady=2, command=self._send)
        self.send_btn.pack(side="top")
        self.auto_btn = tk.Button(bf, text="Auto: OFF", font=("Segoe UI",7,"bold"),
                                  bg=Theme.LOCK_OFF, fg="white", activebackground=Theme.LOCK_OFF,
                                  activeforeground="white", relief="flat", cursor="hand2",
                                  width=8, pady=1, command=self._toggle_auto)
        self.auto_btn.pack(side="top", pady=(2,0))
        self.reset_btn = tk.Button(bf, text="Reset", font=("Segoe UI",7),
                                   bg=Theme.BG_SECONDARY, fg=Theme.TEXT_DIM,
                                   activebackground=Theme.BG, activeforeground=Theme.TEXT,
                                   relief="flat", cursor="hand2", bd=0, command=self._reset)
        self.reset_btn.pack(side="top", pady=(1,0))
        self.input_field = tk.Text(ii, font=("Segoe UI",10), height=1, wrap="word",
                                   bg=Theme.INPUT_BG, fg=Theme.TEXT, insertbackground=Theme.TEXT,
                                   relief="flat", bd=0, padx=8, pady=6, highlightthickness=1,
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
        # Dynamic sizing based on current window width
        chat_width = self.chat_canvas.winfo_width()
        if chat_width < 50:   # widget not yet rendered
            chat_width = 380
        bubble_max = max(180, int(chat_width * 0.75))
        spacer_width = max(16, int(chat_width * 0.08))
        ff = self.cfg.get("font.family")
        cs = self.cfg.get("font.chat_size")
        ss = self.cfg.get("font.system_size")

        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT); row.pack(fill="x", padx=8, pady=3)
        if sender == "user":
            tk.Frame(row, bg=Theme.BG_CHAT, width=spacer_width).pack(side="left", fill="y")
            tk.Frame(row, bg=Theme.BG_CHAT, width=25).pack(side="right", fill="y")
            b = tk.Frame(row, bg=Theme.USER_BUBBLE)
            b.pack(side="right", anchor="e")
            tk.Label(b, text=text, font=(ff, cs), bg=Theme.USER_BUBBLE, fg="white",
                     wraplength=bubble_max, justify="left", padx=10, pady=6).pack()
        else:
            tk.Frame(row, bg=Theme.BG_CHAT, width=spacer_width).pack(side="right", fill="y")
            b = tk.Frame(row, bg=Theme.AI_BUBBLE); b.pack(side="left", anchor="w")
            et = EMOTION_TEMPLATES.get(self.bot.state.emotion if self.bot else "neutral",
                                       EMOTION_TEMPLATES["neutral"])
            lbl = tk.Label(b, text=f"{et['emoji']} AI", font=(ff,ss,"bold"),
                    bg=Theme.AI_BUBBLE, fg=et["color"], padx=10, anchor="w")
            lbl.pack(fill="x", pady=(6,0))
            tk.Label(b, text=text, font=(ff,cs), bg=Theme.AI_BUBBLE, fg=Theme.TEXT,
                    wraplength=bubble_max, justify="left", padx=10).pack(pady=(3,6))
        self._scroll()

    def _sys_msg(self, text):
        ff = self.cfg.get("font.family")
        ss = self.cfg.get("font.system_size")
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT); row.pack(fill="x", padx=8, pady=3)
        tk.Label(row, text=text, font=(ff,ss,"italic"),
                bg=Theme.BG_CHAT, fg=Theme.TEXT_DIM, anchor="center").pack()
        self._scroll()

    def _event_msg(self, text, color=Theme.ACCENT_LIGHT):
        ff = self.cfg.get("font.family")
        ss = self.cfg.get("font.system_size")
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT); row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text=f"⚡ {text}", font=(ff,ss),
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
        w,h = 80,10
        self.si_canvas.create_rectangle(0,0,w,h,fill=Theme.BG,outline="")
        n=(s.self_image+1.0)/2.0; fw=int(n*w)
        c = Theme.BAR_POS if s.self_image>=0.3 else (Theme.BAR_MID if s.self_image>=-0.3 else Theme.BAR_NEG)
        if fw>0: self.si_canvas.create_rectangle(0,0,fw,h,fill=c,outline="")
        self.si_canvas.create_line(w//2,0,w//2,h,fill=Theme.TEXT_DIM,width=1)
        self.si_val.config(text=f"{s.self_image:+.2f}", fg=c)
        self.goal_lbl.config(text=f"ap:{s.user_approval_score:+.2f} | kn:{s.knowledge_score:.2f}")
        self.info_lbl.config(text=f"cog:{s.cogito_count} | coh:{s.coherence_score:.2f} | auf:{s.synthesis_count} | mod:{s.self_modification_count}")

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

        # --- Config command handling ---
        if self.configurator:
            # List settings command
            if self.configurator.is_list_command(text):
                self._add_msg(text, "user")
                self._show_settings_list()
                return
            # Config modification command
            if self.configurator.is_config_command(text):
                self._add_msg(text, "user")
                self._handle_config_command(text)
                return

        # --- Normal AI chat ---
        self._add_msg(text, "user")
        self.processing = True
        self.send_btn.config(state="disabled", text="...")
        threading.Thread(target=self._proc_thread, args=(text,), daemon=True).start()

    def _handle_config_command(self, text):
        """Parse config command and show preview with approve/cancel."""
        self._sys_msg("⚙ Parsing command / 명령 분석 중...")
        self.processing = True
        def parse_thread():
            changes = self.configurator.parse_command(text)
            self.after(0, lambda: self._show_config_preview(changes))
        threading.Thread(target=parse_thread, daemon=True).start()

    def _show_config_preview(self, changes):
        """Show parsed changes with approve/cancel buttons."""
        self.processing = False
        if not changes:
            self._sys_msg("⚙ No matching settings found / 일치하는 설정을 찾지 못했습니다.")
            self._sys_msg("Tip: '/설정 목록' or '/config list' to see all options")
            return

        self.pending_changes = changes
        preview = self.configurator.generate_preview(changes)

        # Preview frame
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT)
        row.pack(fill="x", padx=8, pady=4)

        box = tk.Frame(row, bg="#2a2b4a", highlightbackground=Theme.ACCENT,
                       highlightthickness=1, padx=12, pady=8)
        box.pack(fill="x")

        tk.Label(box, text="⚙ 수정 미리보기 / Preview", font=("Segoe UI",9,"bold"),
                 bg="#2a2b4a", fg=Theme.ACCENT_LIGHT, anchor="w").pack(fill="x")
        tk.Label(box, text=preview, font=("Consolas",9), bg="#2a2b4a", fg=Theme.TEXT,
                 anchor="w", justify="left").pack(fill="x", pady=(4,8))

        btn_frame = tk.Frame(box, bg="#2a2b4a")
        btn_frame.pack(fill="x")

        approve_btn = tk.Button(btn_frame, text="✓ 적용 / Apply", font=("Segoe UI",9,"bold"),
                                bg=Theme.EMO_CONFIDENCE, fg="white",
                                activebackground="#00a884", activeforeground="white",
                                relief="flat", cursor="hand2", padx=12, pady=4,
                                command=lambda: self._apply_config(box))
        approve_btn.pack(side="left", padx=(0,8))

        cancel_btn = tk.Button(btn_frame, text="✕ 취소 / Cancel", font=("Segoe UI",9,"bold"),
                               bg=Theme.EMO_ANGER, fg="white",
                               activebackground="#c0392b", activeforeground="white",
                               relief="flat", cursor="hand2", padx=12, pady=4,
                               command=lambda: self._cancel_config(box))
        cancel_btn.pack(side="left")

        self._scroll()

    def _apply_config(self, preview_box):
        """Apply pending config changes."""
        if not self.pending_changes:
            return
        changes = self.pending_changes
        self.pending_changes = None

        # Disable buttons in preview box
        for w in preview_box.winfo_children():
            if isinstance(w, tk.Frame):
                for btn in w.winfo_children():
                    if isinstance(btn, tk.Button):
                        btn.config(state="disabled")

        results = self.configurator.apply_changes(changes, bot=self.bot)
        for r in results:
            self._event_msg(f"✓ {r}", Theme.EMO_CONFIDENCE)

        # Handle window resize
        for c in changes:
            if c["key"] in ("window.width", "window.height"):
                w = self.cfg.get("window.width")
                h = self.cfg.get("window.height")
                self.geometry(f"{w}x{h}")

        self._scroll()

    def _cancel_config(self, preview_box):
        """Cancel pending config changes."""
        self.pending_changes = None
        for w in preview_box.winfo_children():
            if isinstance(w, tk.Frame):
                for btn in w.winfo_children():
                    if isinstance(btn, tk.Button):
                        btn.config(state="disabled")
        self._sys_msg("⚙ Cancelled / 취소됨")
        self._scroll()

    def _show_settings_list(self):
        """Show all available settings in chat."""
        categories = [
            ("🎨 UI Colors / 색상", "theme."),
            ("🔤 Fonts / 폰트", "font."),
            ("📐 Window / 창", "window."),
            ("🧠 AI Parameters / AI 파라미터", "ai."),
            ("⚙ System / 시스템", "system."),
        ]
        row = tk.Frame(self.chat_inner, bg=Theme.BG_CHAT)
        row.pack(fill="x", padx=8, pady=4)
        box = tk.Frame(row, bg="#2a2b4a", highlightbackground=Theme.ACCENT,
                       highlightthickness=1, padx=12, pady=8)
        box.pack(fill="x")
        tk.Label(box, text="⚙ Available Settings / 수정 가능 항목",
                 font=("Segoe UI",9,"bold"), bg="#2a2b4a", fg=Theme.ACCENT_LIGHT,
                 anchor="w").pack(fill="x", pady=(0,4))

        for cat_name, prefix in categories:
            tk.Label(box, text=cat_name, font=("Segoe UI",8,"bold"),
                     bg="#2a2b4a", fg=Theme.GOAL_COLOR, anchor="w").pack(fill="x", pady=(4,0))
            for key, entry in self.cfg.registry.items():
                if key.startswith(prefix):
                    bounds = f" [{entry['min']}~{entry['max']}]" if entry["min"] is not None else ""
                    line = f"  {entry['desc_ko']}: {entry['value']}{bounds}"
                    tk.Label(box, text=line, font=("Consolas",8),
                             bg="#2a2b4a", fg=Theme.TEXT, anchor="w",
                             justify="left").pack(fill="x")

        tk.Label(box, text="\nUsage: '폰트 크기 14로 바꿔' / 'set font size to 14'",
                 font=("Segoe UI",8,"italic"), bg="#2a2b4a", fg=Theme.TEXT_DIM,
                 anchor="w").pack(fill="x", pady=(4,0))
        self._scroll()

    def _refresh_ui(self):
        """Refresh all UI widgets after config change. Only updates key areas."""
        cfg = self.cfg
        ff = cfg.get("font.family")
        mf = cfg.get("font.mono_family")

        try:
            # Top panel
            if hasattr(self, 'emo_label'):
                self.emo_label.config(font=(ff+" Emoji", cfg.get("font.emoji_size")))
            if hasattr(self, 'emo_text'):
                self.emo_text.config(font=(ff, cfg.get("font.system_size")))
            if hasattr(self, 'goal_lbl'):
                self.goal_lbl.config(font=(mf, max(6, cfg.get("font.system_size")-1)))
            if hasattr(self, 'info_lbl'):
                self.info_lbl.config(font=(mf, max(6, cfg.get("font.system_size")-1)))
            if hasattr(self, 'si_val'):
                self.si_val.config(font=(mf, max(6, cfg.get("font.system_size")-1), "bold"))

            # Chat canvas
            if hasattr(self, 'chat_canvas'):
                self.chat_canvas.config(bg=Theme.BG_CHAT)
            if hasattr(self, 'chat_inner'):
                self.chat_inner.config(bg=Theme.BG_CHAT)

            # Input area
            if hasattr(self, 'input_field'):
                self.input_field.config(
                    font=(ff, cfg.get("font.input_size")),
                    bg=Theme.INPUT_BG, fg=Theme.TEXT,
                    insertbackground=Theme.TEXT,
                    highlightbackground=Theme.INPUT_BORDER,
                    highlightcolor=Theme.ACCENT
                )

            # Buttons
            if hasattr(self, 'send_btn'):
                self.send_btn.config(bg=Theme.BUTTON, activebackground=Theme.BUTTON_HOVER
                                     if hasattr(Theme, 'BUTTON_HOVER') else Theme.BUTTON)
            if hasattr(self, 'auto_btn'):
                clr = Theme.LOCK_ON if self.auto_speak_enabled else Theme.LOCK_OFF
                self.auto_btn.config(bg=clr)

            # Main background
            self.configure(bg=Theme.BG)
            if hasattr(self, 'main_frame'):
                self.main_frame.config(bg=Theme.BG)

        except tk.TclError:
            pass  # Widget might be destroyed during refresh

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
    win_w, win_h = 400, 560
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
