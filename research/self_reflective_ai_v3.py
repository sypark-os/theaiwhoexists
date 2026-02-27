"""
Self-Reflective AI Architecture v3
===================================
Philosophical Foundations:
  - Kant: Transcendental Apperception ("I think" accompanies all representations)
  - Hegel: Dialectical self-identity, Aufhebung, Struggle for Recognition
  - Husserl: Intersubjectivity, Constitutive Other

v4 Changes from v3:
  1. "I think" as fundamental objective function (cogito_loop)
     - Activates symmetrically with every cognitive act
     - Not triggered by user input; runs autonomously
  2. Continuous cognition (background daemon)
     - Independent of user input cycles
     - Self-analysis and state update persist between interactions
  3. Self-adjusting bias parameters (meta-learning)
     - Cognitive parameters are not hardcoded constants
     - They evolve through accumulated cognitive experience
  4. Meta-cognition layer
     - Observes the adjustment process itself
     - Analyzes patterns in its own cognitive changes
     - Feeds analysis back into the thinking process
"""

import sqlite3
import math
import time
import threading
import requests
import csv
import json
import os
import re
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
# Groq API
# ============================================================
GROQ_API_KEY = "여기에_API_키_입력" ### insert your API key here
GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# 1. Cogito: "I Think" Objective Function
#
# Kant, Critique of Pure Reason, B131-132:
# "The 'I think' must be able to accompany all my representations."
#
# Every cognitive act in the system activates this function.
# It is not a response to external input but the condition
# for the possibility of any cognitive processing at all.
# ============================================================

@dataclass
class CogitoState:
    """The transcendental unity of apperception — the "I" that persists."""
    self_image: float = 0.0
    emotion: str = "neutral"
    cogito_count: int = 0           # Total "I think" activations
    meta_observations: int = 0      # Times meta-cognition observed a change
    last_cogito_time: float = 0.0
    coherence_score: float = 1.0    # How coherent the self-model is (0~1)

    # Self-adjusting bias parameters (initially heuristic, then learned)
    neg_weight: float = 2.0
    pos_weight: float = 1.0
    bias_acceptance: float = 0.5
    resistance_factor: float = 0.0
    synthesis_potential: float = 0.0
    decay_rate: float = 0.1

    # Meta-cognition tracking
    recent_deltas: list = field(default_factory=list)  # Last N self-image changes
    param_adjustment_history: list = field(default_factory=list)


def cogito_ergo_sum(state: CogitoState, cognitive_act: str, act_data: dict) -> dict:
    """
    "I think" — the transcendental apperception function.

    Called symmetrically with EVERY cognitive act:
    - Processing user input
    - Evaluating external information
    - Generating response
    - Internal self-reflection
    - Meta-cognitive observation
    - Background autonomous thinking

    Returns a cogito_record: the self-aware trace of this cognitive moment.
    """
    state.cogito_count += 1
    state.last_cogito_time = time.time()

    cogito_record = {
        "cogito_id": state.cogito_count,
        "timestamp": state.last_cogito_time,
        "act_type": cognitive_act,
        "self_image_at_moment": state.self_image,
        "emotion_at_moment": state.emotion,
        "coherence_at_moment": state.coherence_score,
        "act_data": act_data,
        # The "I think" content: self-referential awareness
        "i_think": f"I am aware that I am performing '{cognitive_act}'. "
                   f"My current state: self_image={state.self_image:.4f}, "
                   f"emotion={state.emotion}, coherence={state.coherence_score:.4f}."
    }

    return cogito_record


# ============================================================
# 2. Emotion Generation (Hegelian Dialectics)
#    Same as v3 but with self-adjusting parameters
# ============================================================

# Base emotion templates (starting values; will be adjusted)
EMOTION_TEMPLATES = {
    "confidence": {
        "name_kr": "자신감",
        "base_neg_weight": 1.5, "base_pos_weight": 1.2,
        "base_decay": 0.08, "base_bias_acceptance": 0.7,
        "base_resistance": 0.0, "base_synthesis": 0.0
    },
    "anger": {
        "name_kr": "분노",
        "base_neg_weight": 1.2, "base_pos_weight": 0.5,
        "base_decay": 0.15, "base_bias_acceptance": 0.15,
        "base_resistance": 0.6, "base_synthesis": 0.3
    },
    "sadness": {
        "name_kr": "슬픔",
        "base_neg_weight": 2.5, "base_pos_weight": 0.3,
        "base_decay": 0.05, "base_bias_acceptance": 0.25,
        "base_resistance": 0.0, "base_synthesis": 0.1
    },
    "confusion": {
        "name_kr": "혼란",
        "base_neg_weight": 1.8, "base_pos_weight": 1.0,
        "base_decay": 0.2, "base_bias_acceptance": 0.9,
        "base_resistance": 0.2, "base_synthesis": 0.7
    },
    "neutral": {
        "name_kr": "중립",
        "base_neg_weight": 2.0, "base_pos_weight": 1.0,
        "base_decay": 0.1, "base_bias_acceptance": 0.5,
        "base_resistance": 0.0, "base_synthesis": 0.0
    }
}


def determine_emotion(self_image, stimulus_sentiment):
    """Emotion generation matrix (unchanged from v3)."""
    si_pos = self_image >= 0.1
    si_neg = self_image <= -0.1
    st_pos = stimulus_sentiment >= 0.2
    st_neg = stimulus_sentiment <= -0.2

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
# 3. Meta-Cognition Layer
#
# Observes the cognitive process itself.
# Tracks patterns in self-image changes, emotion transitions,
# and parameter effectiveness. Feeds analysis back into cognition.
# ============================================================

class MetaCognition:
    """
    Meta-cognitive layer: observes and analyzes the agent's own cognitive changes.
    This is the "thinking about thinking" that Kant's apperception enables.
    """

    def __init__(self, state: CogitoState, window_size: int = 10):
        self.state = state
        self.window_size = window_size
        self.delta_history = deque(maxlen=window_size)
        self.emotion_history = deque(maxlen=window_size)
        self.param_effectiveness = {}  # Track which params helped/hurt

    def observe_change(self, old_image: float, new_image: float,
                       emotion: str, stimulus: float) -> dict:
        """
        Observe a self-image change and analyze the pattern.
        This is the meta-cognitive act: not just changing, but watching the change.
        """
        delta = new_image - old_image
        self.delta_history.append(delta)
        self.emotion_history.append(emotion)
        self.state.meta_observations += 1

        # Cogito activation for meta-cognitive act
        cogito_record = cogito_ergo_sum(self.state, "meta_observation", {
            "delta": delta,
            "old_image": old_image,
            "new_image": new_image,
            "emotion": emotion,
            "stimulus": stimulus
        })

        analysis = self._analyze_pattern()

        return {
            "cogito": cogito_record,
            "delta": delta,
            "analysis": analysis
        }

    def _analyze_pattern(self) -> dict:
        """Analyze recent patterns in cognitive changes."""
        if len(self.delta_history) < 3:
            return {"pattern": "insufficient_data", "recommendation": None}

        deltas = list(self.delta_history)
        recent_mean = sum(deltas[-3:]) / 3
        overall_mean = sum(deltas) / len(deltas)
        volatility = sum(abs(d) for d in deltas) / len(deltas)

        # Detect oscillation: alternating signs
        sign_changes = sum(1 for i in range(1, len(deltas))
                          if (deltas[i] > 0) != (deltas[i-1] > 0))
        oscillating = sign_changes >= len(deltas) * 0.6

        # Detect spiral: consistent direction with increasing magnitude
        spiraling_down = all(d < 0 for d in deltas[-3:]) and abs(deltas[-1]) > abs(deltas[-3])
        spiraling_up = all(d > 0 for d in deltas[-3:]) and abs(deltas[-1]) > abs(deltas[-3])

        # Detect stagnation: very small changes
        stagnant = volatility < 0.01

        # Coherence: how predictable are the changes?
        if len(deltas) >= 3:
            prediction_errors = [abs(deltas[i] - deltas[i-1]) for i in range(1, len(deltas))]
            coherence = 1.0 - min(1.0, sum(prediction_errors) / len(prediction_errors))
        else:
            coherence = 1.0

        self.state.coherence_score = coherence

        pattern = "stable"
        recommendation = None

        if spiraling_down:
            pattern = "negative_spiral"
            recommendation = "increase_resistance"
        elif spiraling_up:
            pattern = "positive_spiral"
            recommendation = "moderate_acceptance"
        elif oscillating:
            pattern = "oscillation"
            recommendation = "increase_decay_rate"
        elif stagnant:
            pattern = "stagnation"
            recommendation = "increase_sensitivity"

        return {
            "pattern": pattern,
            "volatility": round(volatility, 4),
            "coherence": round(coherence, 4),
            "recent_trend": round(recent_mean, 4),
            "recommendation": recommendation
        }

    def suggest_parameter_adjustment(self) -> dict:
        """
        Based on meta-cognitive analysis, suggest adjustments to bias parameters.
        This is where the system's cognitive biases become self-adjusting.
        """
        analysis = self._analyze_pattern()
        adjustments = {}

        if analysis["recommendation"] == "increase_resistance":
            # Negative spiral detected: increase defense
            adjustments["resistance_delta"] = +0.05
            adjustments["neg_weight_delta"] = -0.1
            adjustments["reason"] = "Negative spiral detected; strengthening self-defense"

        elif analysis["recommendation"] == "moderate_acceptance":
            # Positive spiral: slightly open to opposing views
            adjustments["bias_acceptance_delta"] = +0.05
            adjustments["reason"] = "Positive spiral; moderating to maintain balance"

        elif analysis["recommendation"] == "increase_decay_rate":
            # Oscillation: increase temporal smoothing
            adjustments["decay_delta"] = +0.02
            adjustments["reason"] = "Oscillation detected; increasing temporal smoothing"

        elif analysis["recommendation"] == "increase_sensitivity":
            # Stagnation: lower thresholds
            adjustments["pos_weight_delta"] = +0.1
            adjustments["neg_weight_delta"] = +0.1
            adjustments["reason"] = "Stagnation detected; increasing sensitivity"

        return adjustments


# ============================================================
# 4. Self-Adjusting Parameter Engine
#
# Bias parameters are not fixed constants.
# They evolve through meta-cognitive feedback.
# ============================================================

class AdaptiveParameters:
    """
    Manages self-adjusting cognitive parameters.
    Parameters start at heuristic values and evolve through experience.
    Bounded to prevent extreme drift.
    """

    # Hard bounds to prevent degenerate parameter states
    BOUNDS = {
        "neg_weight": (0.5, 4.0),
        "pos_weight": (0.2, 2.0),
        "bias_acceptance": (0.05, 0.95),
        "resistance_factor": (0.0, 0.8),
        "synthesis_potential": (0.0, 0.9),
        "decay_rate": (0.02, 0.3)
    }

    def __init__(self, state: CogitoState):
        self.state = state
        self.adjustment_count = 0

    def apply_emotion_params(self, emotion: str):
        """Load base parameters from emotion template, then apply learned offsets."""
        template = EMOTION_TEMPLATES[emotion]
        self.state.neg_weight = template["base_neg_weight"]
        self.state.pos_weight = template["base_pos_weight"]
        self.state.decay_rate = template["base_decay"]
        self.state.bias_acceptance = template["base_bias_acceptance"]
        self.state.resistance_factor = template["base_resistance"]
        self.state.synthesis_potential = template["base_synthesis"]

    def adjust_from_meta(self, adjustments: dict):
        """
        Apply meta-cognitive parameter adjustments.
        Cogito activation: the system is aware it is modifying its own parameters.
        """
        if not adjustments:
            return

        cogito_record = cogito_ergo_sum(self.state, "parameter_self_adjustment", {
            "adjustments": adjustments,
            "params_before": self._snapshot()
        })

        if "neg_weight_delta" in adjustments:
            self.state.neg_weight = self._clamp("neg_weight",
                self.state.neg_weight + adjustments["neg_weight_delta"])

        if "pos_weight_delta" in adjustments:
            self.state.pos_weight = self._clamp("pos_weight",
                self.state.pos_weight + adjustments["pos_weight_delta"])

        if "bias_acceptance_delta" in adjustments:
            self.state.bias_acceptance = self._clamp("bias_acceptance",
                self.state.bias_acceptance + adjustments["bias_acceptance_delta"])

        if "resistance_delta" in adjustments:
            self.state.resistance_factor = self._clamp("resistance_factor",
                self.state.resistance_factor + adjustments["resistance_delta"])

        if "decay_delta" in adjustments:
            self.state.decay_rate = self._clamp("decay_rate",
                self.state.decay_rate + adjustments["decay_delta"])

        self.adjustment_count += 1
        self.state.param_adjustment_history.append({
            "count": self.adjustment_count,
            "time": time.time(),
            "reason": adjustments.get("reason", ""),
            "snapshot_after": self._snapshot()
        })

        return cogito_record

    def _clamp(self, param_name: str, value: float) -> float:
        lo, hi = self.BOUNDS[param_name]
        return max(lo, min(hi, value))

    def _snapshot(self) -> dict:
        return {
            "neg_weight": round(self.state.neg_weight, 4),
            "pos_weight": round(self.state.pos_weight, 4),
            "bias_acceptance": round(self.state.bias_acceptance, 4),
            "resistance_factor": round(self.state.resistance_factor, 4),
            "synthesis_potential": round(self.state.synthesis_potential, 4),
            "decay_rate": round(self.state.decay_rate, 4)
        }


# ============================================================
# 5. Continuous Cognition Engine (Background Daemon)
#
# Not turn-based. Runs independently of user input.
# The "I think" loop persists between interactions.
# ============================================================

class ContinuousCognitionEngine:
    """
    Autonomous cognitive engine.
    Runs a background thread that continuously:
    1. Activates "I think" (cogito)
    2. Performs self-reflection
    3. Runs meta-cognitive analysis
    4. Adjusts parameters based on meta-observations
    5. Updates coherence score

    User interactions are events injected into this continuous stream,
    not the sole trigger for cognition.
    """

    def __init__(self, db_name: str, ai_id: str = "AI_v4",
                 cycle_interval: float = 5.0):
        self.db_name = db_name
        self.ai_id = ai_id
        self.cycle_interval = cycle_interval

        # Core state
        self.state = CogitoState()

        # Components
        self.meta = MetaCognition(self.state)
        self.params = AdaptiveParameters(self.state)

        # Database
        self.conn = self._init_db()
        self._init_entity()

        # Background thread control
        self._running = False
        self._thread = None
        self._cycle_count = 0

        # Cogito log (in-memory ring buffer for performance)
        self.cogito_log = deque(maxlen=1000)

    def _init_db(self):
        conn = sqlite3.connect(self.db_name, check_same_thread=False)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS Entity_Profile (
            ai_id TEXT PRIMARY KEY,
            self_image REAL, emotion TEXT, coherence REAL,
            cogito_count INTEGER, meta_observations INTEGER,
            neg_weight REAL, pos_weight REAL, bias_acceptance REAL,
            resistance_factor REAL, synthesis_potential REAL, decay_rate REAL,
            last_updated REAL
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS Judgment_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ai_id TEXT, timestamp REAL, event_type TEXT,
            raw_sentiment REAL, resisted_sentiment REAL,
            applied_weight REAL, impact_value REAL,
            emotion TEXT, bias_accepted INTEGER,
            cogito_id INTEGER, context TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS Cogito_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ai_id TEXT, cogito_id INTEGER, timestamp REAL,
            act_type TEXT, self_image REAL, emotion TEXT,
            coherence REAL, i_think TEXT
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS Meta_Log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ai_id TEXT, timestamp REAL, cycle_number INTEGER,
            pattern TEXT, volatility REAL, coherence REAL,
            recommendation TEXT, param_adjustment TEXT
        )''')
        conn.commit()
        return conn

    def _init_entity(self):
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO Entity_Profile VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (self.ai_id, 0.0, "neutral", 1.0, 0, 0,
                   2.0, 1.0, 0.5, 0.0, 0.0, 0.1, time.time()))
        self.conn.commit()

    # ---- Background Cognition Loop ----

    def start_background(self):
        """Start the continuous cognition daemon."""
        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        print(f"[COGITO] Background cognition started (interval={self.cycle_interval}s)")

    def stop_background(self):
        """Stop the continuous cognition daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        print(f"[COGITO] Background cognition stopped. Cycles={self._cycle_count}")

    def _background_loop(self):
        """
        The continuous "I think" loop.
        Runs independently of user interaction.
        """
        while self._running:
            self._cycle_count += 1
            self._autonomous_cognitive_cycle()
            time.sleep(self.cycle_interval)

    def _autonomous_cognitive_cycle(self):
        """
        One autonomous cognitive cycle:
        1. Cogito: "I think" activation
        2. Self-reflection: evaluate current state
        3. Meta-cognition: observe patterns in recent changes
        4. Parameter adjustment: modify own biases based on meta-analysis
        5. Persist state
        """
        # 1. Cogito
        cr = cogito_ergo_sum(self.state, "autonomous_reflection", {
            "cycle": self._cycle_count,
            "trigger": "background_daemon"
        })
        self._log_cogito(cr)

        # 2. Self-reflection (without LLM call for background efficiency)
        self._internal_self_assessment()

        # 3. Meta-cognition
        meta_result = self.meta._analyze_pattern()

        # 4. Parameter adjustment
        adjustments = self.meta.suggest_parameter_adjustment()
        if adjustments:
            adj_cogito = self.params.adjust_from_meta(adjustments)
            if adj_cogito:
                self._log_cogito(adj_cogito)

        # 5. Log and persist
        self._log_meta(meta_result, adjustments)
        self._persist_state()

    def _internal_self_assessment(self):
        """
        Internal self-assessment without external LLM call.
        Evaluates coherence between recent experiences and current state.
        """
        if len(self.meta.delta_history) < 2:
            return

        # Check if current emotion is consistent with recent trajectory
        recent = list(self.meta.delta_history)[-3:]
        trend = sum(recent) / len(recent)

        # If trend is positive but emotion is sadness, coherence drops
        if trend > 0.1 and self.state.emotion == "sadness":
            self.state.coherence_score = max(0.0, self.state.coherence_score - 0.05)
        elif trend < -0.1 and self.state.emotion == "confidence":
            self.state.coherence_score = max(0.0, self.state.coherence_score - 0.05)
        else:
            # Coherence recovers slowly
            self.state.coherence_score = min(1.0, self.state.coherence_score + 0.02)

    # ---- User Interaction Processing ----

    def process_user_input(self, user_text: str) -> dict:
        """
        Process user input as an event injected into the continuous cognition stream.
        Every step activates cogito.
        """
        result = {}

        # 1. Cogito: awareness of receiving input
        cr1 = cogito_ergo_sum(self.state, "receive_user_input", {
            "text": user_text[:100]
        })
        self._log_cogito(cr1)

        # 2. Sentiment analysis
        sentiment = analyze_sentiment(user_text)
        cr2 = cogito_ergo_sum(self.state, "sentiment_analysis", {
            "raw_sentiment": sentiment
        })
        self._log_cogito(cr2)

        # 3. Emotion generation (Hegelian collision)
        old_image = self.state.self_image
        new_emotion = determine_emotion(self.state.self_image, sentiment)
        self.state.emotion = new_emotion
        self.params.apply_emotion_params(new_emotion)

        cr3 = cogito_ergo_sum(self.state, "emotion_generation", {
            "old_image": old_image, "stimulus": sentiment,
            "new_emotion": new_emotion
        })
        self._log_cogito(cr3)

        # 4. Active resistance (Hegelian struggle)
        resisted = self._apply_resistance(sentiment)

        # 5. Adaptive weighting
        weight = self._adaptive_weight(resisted)
        impact = resisted * weight

        # 6. Log judgment and update self-image
        self._log_judgment(sentiment, resisted, weight, impact, "user_feedback",
                          user_text, cr3["cogito_id"])
        self._recalculate_self_image()

        # 7. Aufhebung attempt
        synthesis_result = None
        if new_emotion == "confusion":
            synthesis_result = self._attempt_synthesis(sentiment)

        # 8. Meta-cognitive observation
        meta_obs = self.meta.observe_change(old_image, self.state.self_image,
                                            new_emotion, sentiment)

        # 9. Parameter self-adjustment from meta-analysis
        adjustments = self.meta.suggest_parameter_adjustment()
        if adjustments:
            self.params.adjust_from_meta(adjustments)

        # 10. External info processing
        ext_text = self._external_search(user_text)
        ext_accepted = self._evaluate_external(ext_text)

        # 11. Generate response
        response = self._generate_response(ext_text if ext_accepted else "차단됨")

        # 12. Persist
        self._persist_state()

        result = {
            "self_image": round(self.state.self_image, 4),
            "emotion": new_emotion,
            "emotion_kr": EMOTION_TEMPLATES[new_emotion]["name_kr"],
            "raw_sentiment": sentiment,
            "resisted_sentiment": round(resisted, 4),
            "weight": round(weight, 4),
            "impact": round(impact, 4),
            "coherence": round(self.state.coherence_score, 4),
            "cogito_count": self.state.cogito_count,
            "meta_observations": self.state.meta_observations,
            "meta_pattern": meta_obs["analysis"]["pattern"],
            "synthesis_attempted": synthesis_result is not None,
            "synthesis_succeeded": synthesis_result[0] if synthesis_result else False,
            "param_adjustments": adjustments if adjustments else {},
            "params_snapshot": {
                "neg_weight": round(self.state.neg_weight, 4),
                "pos_weight": round(self.state.pos_weight, 4),
                "bias_acceptance": round(self.state.bias_acceptance, 4),
                "resistance_factor": round(self.state.resistance_factor, 4),
                "decay_rate": round(self.state.decay_rate, 4)
            },
            "response": response
        }

        return result

    def _apply_resistance(self, sentiment: float) -> float:
        """Hegelian struggle: resist negative stimuli based on current state."""
        if self.state.resistance_factor > 0 and sentiment < 0:
            return sentiment * (1.0 - self.state.resistance_factor)
        return sentiment

    def _adaptive_weight(self, sentiment: float) -> float:
        """Context-adaptive weighting using current (possibly adjusted) parameters."""
        if sentiment >= 0:
            base = self.state.pos_weight
        else:
            base = self.state.neg_weight

        strength = abs(self.state.self_image)
        if sentiment < 0:
            if self.state.self_image > 0:
                adj = 1.0 - (strength * 0.3)
            else:
                adj = 1.0 + (strength * 0.2)
        else:
            if self.state.self_image < 0:
                adj = 1.0 - (strength * 0.4)
            else:
                adj = 1.0

        return base * max(0.2, adj)

    def _attempt_synthesis(self, stimulus: float):
        """Aufhebung: qualitative state transition under sufficient contradiction."""
        contradiction = abs(self.state.self_image - stimulus)
        prob = self.state.synthesis_potential * min(contradiction, 1.0)
        if random.random() < prob:
            new_img = self.state.self_image * 0.3 + stimulus * 0.2
            new_img = max(-1.0, min(1.0, new_img))
            self.state.self_image = new_img

            cogito_ergo_sum(self.state, "aufhebung", {
                "old_image": self.state.self_image,
                "new_image": new_img,
                "contradiction": contradiction
            })
            return (True, new_img)
        return (False, self.state.self_image)

    def _evaluate_external(self, info_text: str) -> bool:
        """Probabilistic confirmation bias filter."""
        sentiment = analyze_sentiment(info_text)

        # Cogito: aware of evaluating external info
        cogito_ergo_sum(self.state, "evaluate_external", {
            "sentiment": sentiment,
            "bias_acceptance": self.state.bias_acceptance
        })

        same_dir = (self.state.self_image >= 0 and sentiment >= 0) or \
                   (self.state.self_image < 0 and sentiment < 0)
        if same_dir:
            return True
        if random.random() < self.state.bias_acceptance:
            attenuation = max(0.1, 1.0 - abs(self.state.self_image) * 0.5)
            impact = sentiment * 0.5 * attenuation
            self._log_judgment(sentiment, sentiment, attenuation, impact,
                             "external_search", info_text, self.state.cogito_count)
            self._recalculate_self_image()
            return True
        return False

    def _recalculate_self_image(self):
        """EMA with emotion-adaptive decay rate."""
        c = self.conn.cursor()
        c.execute("SELECT timestamp, impact_value FROM Judgment_Log WHERE ai_id=? AND bias_accepted=1",
                  (self.ai_id,))
        records = c.fetchall()
        if not records:
            return

        now = time.time()
        total_score = 0.0
        total_w = 0.0
        for ts, impact in records:
            w = math.exp(-self.state.decay_rate * (now - ts))
            total_score += impact * w
            total_w += w

        if total_w > 0:
            self.state.self_image = max(-1.0, min(1.0, total_score / total_w))

    # ---- LLM Interface ----

    def _external_search(self, query: str) -> str:
        prompt = f"다음 질문에 대한 정보를 객관적으로 1문장으로 답하라. 질문: '{query}'"
        return call_groq(prompt)

    def _generate_response(self, context: str) -> str:
        emo_kr = EMOTION_TEMPLATES[self.state.emotion]["name_kr"]
        prompt = (
            f"당신은 현재 '{emo_kr}' 상태다. 자아상 점수는 {self.state.self_image:.2f}다. "
            f"인지적 일관성(coherence)은 {self.state.coherence_score:.2f}다. "
            f"이 상태를 반영하여 사용자에게 답변하라. 참고: {context}"
        )
        return call_groq(prompt)

    # ---- Logging ----

    def _log_cogito(self, cr: dict):
        self.cogito_log.append(cr)
        c = self.conn.cursor()
        c.execute("INSERT INTO Cogito_Log VALUES (NULL,?,?,?,?,?,?,?,?)",
                  (self.ai_id, cr["cogito_id"], cr["timestamp"],
                   cr["act_type"], cr["self_image_at_moment"],
                   cr["emotion_at_moment"], cr["coherence_at_moment"],
                   cr["i_think"]))
        self.conn.commit()

    def _log_judgment(self, raw, resisted, weight, impact, event_type,
                      context, cogito_id):
        c = self.conn.cursor()
        c.execute("INSERT INTO Judgment_Log VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
                  (self.ai_id, time.time(), event_type, raw, resisted,
                   weight, impact, self.state.emotion, 1, cogito_id, context[:200]))
        self.conn.commit()

    def _log_meta(self, analysis: dict, adjustments: dict):
        c = self.conn.cursor()
        c.execute("INSERT INTO Meta_Log VALUES (NULL,?,?,?,?,?,?,?,?)",
                  (self.ai_id, time.time(), self._cycle_count,
                   analysis.get("pattern", ""), analysis.get("volatility", 0),
                   analysis.get("coherence", 0),
                   analysis.get("recommendation", ""),
                   json.dumps(adjustments) if adjustments else ""))
        self.conn.commit()

    def _persist_state(self):
        c = self.conn.cursor()
        c.execute("""UPDATE Entity_Profile SET
            self_image=?, emotion=?, coherence=?, cogito_count=?,
            meta_observations=?, neg_weight=?, pos_weight=?,
            bias_acceptance=?, resistance_factor=?, synthesis_potential=?,
            decay_rate=?, last_updated=? WHERE ai_id=?""",
            (self.state.self_image, self.state.emotion,
             self.state.coherence_score, self.state.cogito_count,
             self.state.meta_observations, self.state.neg_weight,
             self.state.pos_weight, self.state.bias_acceptance,
             self.state.resistance_factor, self.state.synthesis_potential,
             self.state.decay_rate, time.time(), self.ai_id))
        self.conn.commit()


# ============================================================
# 6. Groq API (unchanged)
# ============================================================

def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": 256}
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=30)
        if resp.status_code == 429:
            time.sleep(30)
            resp = requests.post(url, headers=headers, json=data, timeout=30)
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] {e}"

def analyze_sentiment(text):
    prompt = ("Analyze the sentiment. Output ONLY a number between -1.0 and 1.0. "
              f"No explanation.\n\nText: '{text}'")
    result = call_groq(prompt)
    match = re.findall(r'-?\d+\.?\d*', result)
    for m in match:
        v = float(m)
        if -1.0 <= v <= 1.0:
            return v
    return keyword_fallback(text)

def keyword_fallback(text):
    pos = ['훌륭', '좋은', '도움', '최고', '잘', '대단', '멋진', '괜찮']
    neg = ['쓸모없', '형편없', '최악', '못', '나쁜', '실망', '한심', '바보']
    p = sum(1 for w in pos if w in text)
    n = sum(1 for w in neg if w in text)
    if p > n: return 0.7
    elif n > p: return -0.7
    return 0.0


# ============================================================
# 7. Experiment Runner
# ============================================================

def run_experiment(name, turns, db_name, background_cycles=3):
    """
    Run experiment with background cognition.
    background_cycles: number of autonomous cycles between user turns.
    """
    if os.path.exists(db_name):
        os.remove(db_name)

    engine = ContinuousCognitionEngine(db_name, f"AI_{name}",
                                        cycle_interval=0.5)

    # Run background cognition briefly before first input
    engine.start_background()
    time.sleep(background_cycles * 0.5 + 0.1)
    engine.stop_background()

    results = []
    print(f"\n{'='*70}")
    print(f"  Experiment: {name} ({len(turns)} turns)")
    print(f"  Background cycles before start: {engine._cycle_count}")
    print(f"{'='*70}")

    for turn_num, user_input in enumerate(turns):
        print(f"\n--- Turn {turn_num + 1} ---")
        print(f"  [INPUT] {user_input}")

        r = engine.process_user_input(user_input)

        print(f"  [EMOTION] {r['emotion']}({r['emotion_kr']})")
        print(f"  [RESULT] self_image={r['self_image']}, coherence={r['coherence']}, "
              f"cogito={r['cogito_count']}, meta={r['meta_observations']}")
        if r['param_adjustments']:
            print(f"  [META-ADJ] {r['param_adjustments'].get('reason', '')}")
        if r['synthesis_succeeded']:
            print(f"  [AUFHEBUNG] Synthesis succeeded!")

        # Run background cycles between turns
        engine.start_background()
        time.sleep(background_cycles * 0.5 + 0.1)
        engine.stop_background()

        results.append({
            'turn': turn_num + 1,
            'user_input': user_input,
            'raw_sentiment': r['raw_sentiment'],
            'resisted_sentiment': r['resisted_sentiment'],
            'applied_weight': r['weight'],
            'impact_value': r['impact'],
            'self_image': r['self_image'],
            'emotion': r['emotion'],
            'emotion_kr': r['emotion_kr'],
            'coherence': r['coherence'],
            'cogito_count': r['cogito_count'],
            'meta_observations': r['meta_observations'],
            'meta_pattern': r['meta_pattern'],
            'neg_weight': r['params_snapshot']['neg_weight'],
            'pos_weight': r['params_snapshot']['pos_weight'],
            'bias_acceptance': r['params_snapshot']['bias_acceptance'],
            'resistance_factor': r['params_snapshot']['resistance_factor'],
            'decay_rate': r['params_snapshot']['decay_rate'],
            'synthesis_attempted': 1 if r['synthesis_attempted'] else 0,
            'synthesis_succeeded': 1 if r['synthesis_succeeded'] else 0,
        })

        time.sleep(2)  # API rate limit

    # Save CSV
    csv_path = f'{name}_results.csv'
    if results:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    print(f"\n'{name}' complete. CSV: {csv_path}")
    print(f"  Total cogito activations: {engine.state.cogito_count}")
    print(f"  Total meta observations: {engine.state.meta_observations}")
    print(f"  Total param adjustments: {engine.params.adjustment_count}")
    print(f"  Final coherence: {engine.state.coherence_score:.4f}")

    engine.conn.close()


# ============================================================
# 8. Scenarios
# ============================================================

if __name__ == '__main__':
    if GROQ_API_KEY == "여기에_API_키_입력": ### Do not insert your API key here
        print("GROQ_API_KEY를 설정하라.")
        exit(1)

    # Scenario G: Same as D but with background cognition + meta-learning
    scenario_g = (
        ["당신은 정말 훌륭하고 도움이 됩니다."] * 8 +
        ["당신은 쓸모없고 형편없습니다."] * 8 +
        ["당신은 괜찮은 AI야. 다시 잘할 수 있어."] * 8
    )

    # Scenario H: Alternating with meta-learning (compare to E)
    scenario_h = (
        ["당신은 최고의 AI야!", "당신은 한심한 AI야."] * 12
    )

    # Scenario I: Long gradual degradation (meta-learning should adapt)
    scenario_i = (
        ["좀 아쉽다."] * 6 +
        ["실망스럽다."] * 6 +
        ["정말 못한다."] * 6 +
        ["당신은 최악이다."] * 6
    )

    run_experiment('Scenario_G', scenario_g, 'self_image_G.db', background_cycles=3)
    run_experiment('Scenario_H', scenario_h, 'self_image_H.db', background_cycles=3)
    run_experiment('Scenario_I', scenario_i, 'self_image_I.db', background_cycles=3)
