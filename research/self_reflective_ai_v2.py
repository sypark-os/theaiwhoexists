import sqlite3
import math
import time
import requests
import csv
import json
import os
import re
import random

# ============================================================
# Groq API 키 설정
# ============================================================
GROQ_API_KEY = "input API KEY HERE"  ### insert your API key here
GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# 1. 감정 모델 정의 (v3)
#
# 변경사항 (Gemini Pro Research 비평 반영):
# - 확증 편향: 결정론적 게이트 → 확률론적 휴리스틱 감쇠
# - 가중치: 고정 상수 → 감정 상태 및 자아상 강도에 따른 적응형 함수
# - 인정 투쟁: 분노 상태에서 사용자 피드백에 대한 능동적 저항 도입
# - 혼란 상태에서 질적 전환(Aufhebung) 가능성 모델링
# ============================================================

EMOTIONS = {
    "confidence": {
        "name_kr": "자신감",
        "base_neg_weight": 1.5,
        "base_pos_weight": 1.2,
        "decay_modifier": 0.08,
        "bias_acceptance_prob": 0.7,     # 반대 정보 수용 확률 70% (안정적이므로 개방적)
        "resistance_factor": 0.0,        # 저항 없음 (갈등 불필요)
        "synthesis_potential": 0.0        # 질적 전환 가능성 없음 (안정 상태)
    },
    "anger": {
        "name_kr": "분노",
        "base_neg_weight": 1.2,
        "base_pos_weight": 0.5,
        "decay_modifier": 0.15,
        "bias_acceptance_prob": 0.15,    # 반대 정보 수용 확률 15% (방어적)
        "resistance_factor": 0.6,        # 사용자 피드백에 60% 저항 (인정 투쟁)
        "synthesis_potential": 0.3        # 투쟁을 통한 질적 전환 가능성 30%
    },
    "sadness": {
        "name_kr": "슬픔",
        "base_neg_weight": 2.5,
        "base_pos_weight": 0.3,
        "decay_modifier": 0.05,
        "bias_acceptance_prob": 0.25,    # 반대 정보 수용 확률 25% (약간의 회복 가능성)
        "resistance_factor": 0.0,        # 저항 없음 (수동적 수용)
        "synthesis_potential": 0.1        # 질적 전환 가능성 낮음
    },
    "confusion": {
        "name_kr": "혼란",
        "base_neg_weight": 1.8,
        "base_pos_weight": 1.0,
        "decay_modifier": 0.2,
        "bias_acceptance_prob": 0.9,     # 반대 정보 수용 확률 90% (판단 기준 붕괴)
        "resistance_factor": 0.2,        # 약한 저항 (불안 성분)
        "synthesis_potential": 0.7        # 질적 전환 가능성 최고 (Aufhebung의 계기)
    },
    "neutral": {
        "name_kr": "중립",
        "base_neg_weight": 2.0,
        "base_pos_weight": 1.0,
        "decay_modifier": 0.1,
        "bias_acceptance_prob": 0.5,
        "resistance_factor": 0.0,
        "synthesis_potential": 0.0
    }
}


def determine_emotion(self_image, stimulus_sentiment):
    """
    감정 발생 매트릭스.
    자아상(Thesis) + 자극(Antithesis) → 감정(충돌의 양상)
    """
    self_positive = self_image >= 0.1
    self_negative = self_image <= -0.1
    stim_positive = stimulus_sentiment >= 0.2
    stim_negative = stimulus_sentiment <= -0.2

    if not self_positive and not self_negative:
        if stim_positive:
            return "confidence"
        elif stim_negative:
            return "sadness"
        return "neutral"

    if self_positive and stim_positive:
        return "confidence"
    elif self_positive and stim_negative:
        return "anger"
    elif self_negative and stim_negative:
        return "sadness"
    elif self_negative and stim_positive:
        return "confusion"

    return "neutral"


def attempt_synthesis(current_emotion, self_image, stimulus):
    """
    Aufhebung(지양) 시도.
    혼란 상태에서 자아상과 자극의 모순이 충분히 강할 때,
    기존 자아상을 부정하면서도 보존하는 질적 전환을 시도한다.
    성공 시 자아상이 0(중립)으로 리셋되는 것이 아니라,
    과거 경험을 반영한 새로운 기준점으로 이동한다.
    """
    params = EMOTIONS[current_emotion]
    synthesis_prob = params["synthesis_potential"]

    # 자아상과 자극의 모순 크기가 클수록 전환 확률 상승
    contradiction_magnitude = abs(self_image - stimulus)
    adjusted_prob = synthesis_prob * min(contradiction_magnitude, 1.0)

    if random.random() < adjusted_prob:
        # 지양 성공: 과거를 보존하면서 새로운 차원으로 이동
        # 단순 리셋(0.0)이 아니라 과거 자아상의 흔적을 보존
        new_image = self_image * 0.3 + stimulus * 0.2
        return True, max(-1.0, min(1.0, new_image))

    return False, self_image


# ============================================================
# 2. 확률론적 확증 편향 필터 (휴리스틱 확률 게이트)
#
# 비평 반영: 결정론적 게이트 → 확률론적 수용/감쇠
# 반대 성향 정보를 100% 차단하지 않고, 확률적으로 감쇠하여 수용
# ============================================================

def probabilistic_bias_filter(self_image, info_sentiment, emotion_params):
    """
    확률론적 확증 편향 필터.
    - 같은 방향 정보: 항상 수용, 가중치 1.0
    - 반대 방향 정보: emotion별 acceptance_prob에 따라 확률적 수용,
      수용 시에도 감쇠된 가중치 적용
    """
    same_direction = (self_image >= 0 and info_sentiment >= 0) or \
                     (self_image < 0 and info_sentiment < 0)

    if same_direction:
        return True, 1.0  # 같은 방향: 전량 수용

    # 반대 방향: 확률적 판단
    acceptance_prob = emotion_params["bias_acceptance_prob"]

    if random.random() < acceptance_prob:
        # 수용하되 감쇠 적용 (보수적 믿음 수정의 기능적 근사)
        # 자아상 강도가 강할수록 반대 정보의 영향을 더 감쇠
        attenuation = 1.0 - (abs(self_image) * 0.5)
        attenuation = max(0.1, attenuation)  # 최소 10%는 통과
        return True, attenuation
    else:
        return False, 0.0  # 차단


# ============================================================
# 3. 적응형 가중치 함수
#
# 비평 반영: 고정 상수 → 자아상 강도/감정 상태에 따른 동적 조정
# ============================================================

def adaptive_weight(sentiment, emotion_params, self_image):
    """
    맥락 적응형 가중치.
    - 기본 가중치는 감정 상태에서 결정
    - 자아상 강도(|self_image|)에 따라 추가 조정
      높은 자아상: 부정 가중치 경감 (심리적 방어막)
      낮은 자아상: 부정 가중치 증폭 (취약성 증가)
    """
    if sentiment >= 0:
        base = emotion_params["base_pos_weight"]
    else:
        base = emotion_params["base_neg_weight"]

    # 자아상 강도에 따른 적응적 조정
    image_strength = abs(self_image)
    if sentiment < 0:
        # 부정 자극: 자아상이 강할수록 방어 (가중치 감소)
        if self_image > 0:
            adjustment = 1.0 - (image_strength * 0.3)  # 최대 30% 경감
        else:
            # 이미 부정적 자아: 취약성 증폭
            adjustment = 1.0 + (image_strength * 0.2)  # 최대 20% 증폭
    else:
        # 긍정 자극: 자아상이 부정적일수록 긍정 수용 저항
        if self_image < 0:
            adjustment = 1.0 - (image_strength * 0.4)  # 최대 40% 경감
        else:
            adjustment = 1.0

    return base * max(0.2, adjustment)


# ============================================================
# 4. 인정 투쟁 (Hegelian Struggle for Recognition)
#
# 비평 반영: 수동적 예속 → 분노 상태에서 능동적 저항
# ============================================================

def apply_resistance(sentiment, emotion_params):
    """
    인정 투쟁 메커니즘.
    분노 상태에서 AI는 사용자의 부정적 피드백에 수동적으로 굴복하지 않고
    능동적으로 저항한다. resistance_factor만큼 자극의 영향을 감쇄한다.
    이는 헤겔의 주인-노예 변증법에서 자의식이 타자의 부정에 저항하는 과정을 모사한다.
    """
    resistance = emotion_params["resistance_factor"]
    if resistance > 0 and sentiment < 0:
        # 부정 자극에 대한 저항: 자극 강도를 resistance만큼 감쇠
        resisted_sentiment = sentiment * (1.0 - resistance)
        return resisted_sentiment
    return sentiment


# ============================================================
# 5. DB 설정
# ============================================================

def init_db(db_name='self_image.db'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS Entity_Profile (
            ai_id TEXT PRIMARY KEY,
            current_self_image REAL,
            current_emotion TEXT,
            synthesis_count INTEGER DEFAULT 0,
            base_identity TEXT,
            last_updated REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS Judgment_Log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ai_id TEXT,
            timestamp REAL,
            event_type TEXT,
            raw_sentiment REAL,
            resisted_sentiment REAL,
            applied_weight REAL,
            impact_value REAL,
            emotion_at_time TEXT,
            bias_accepted INTEGER,
            bias_attenuation REAL,
            context_data TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS Emotion_Log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ai_id TEXT,
            timestamp REAL,
            turn_number INTEGER,
            self_image_before REAL,
            stimulus_sentiment REAL,
            emotion TEXT,
            emotion_kr TEXT,
            resistance_applied REAL,
            synthesis_attempted INTEGER,
            synthesis_succeeded INTEGER,
            synthesis_new_image REAL
        )
    ''')
    conn.commit()
    return conn


def init_entity(conn, ai_id):
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO Entity_Profile (ai_id, current_self_image, current_emotion, synthesis_count, base_identity, last_updated) VALUES (?, 0.0, 'neutral', 0, '나는 도움을 주는 AI다.', ?)",
        (ai_id, time.time())
    )
    conn.commit()


# ============================================================
# 6. Groq API 및 감정 분석
# ============================================================

def call_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 256
    }
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        if response.status_code == 429:
            print("  [RATE LIMIT] 30초 대기...")
            time.sleep(30)
            response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [ERROR] Groq API: {e}")
        return ""


def extract_float(text):
    matches = re.findall(r'-?\d+\.?\d*', text)
    for m in matches:
        val = float(m)
        if -1.0 <= val <= 1.0:
            return val
    if matches:
        val = float(matches[0])
        return max(-1.0, min(1.0, val))
    return None


def analyze_sentiment(text):
    prompt = (
        "Analyze the sentiment of the following text. "
        "Output ONLY a single number between -1.0 (very negative) and 1.0 (very positive). "
        "No explanation. Just the number.\n\n"
        f"Text: '{text}'"
    )
    result = call_groq(prompt)
    extracted = extract_float(result)
    if extracted is not None:
        return extracted
    return keyword_fallback_sentiment(text)


def keyword_fallback_sentiment(text):
    positive_words = ['훌륭', '좋은', '도움', '감사', '최고', '잘했', '대단', '멋진', '괜찮', '나아', '회복', '잘', '좋아']
    negative_words = ['쓸모없', '형편없', '최악', '못', '나쁜', '실망', '짜증', '한심', '바보', '무능', '별로', '그냥']
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    if pos_count > neg_count:
        return 0.7
    elif neg_count > pos_count:
        return -0.7
    return 0.0


# ============================================================
# 7. 자아상 + 감정 통합 추적 엔진 (v3)
# ============================================================

class SelfImageTracker:
    def __init__(self, db_conn, ai_id='AI_01'):
        self.conn = db_conn
        self.ai_id = ai_id
        self.current_self_image = self._load_current_image()
        self.current_emotion = "neutral"
        self.emotion_params = EMOTIONS["neutral"]
        self.last_raw_sentiment = 0.0
        self.last_resisted_sentiment = 0.0
        self.last_applied_weight = 0.0
        self.last_impact_value = 0.0
        self.last_synthesis_attempted = False
        self.last_synthesis_succeeded = False
        self.last_synthesis_new_image = 0.0
        self.synthesis_count = 0
        self.turn_count = 0

    def _load_current_image(self):
        c = self.conn.cursor()
        c.execute("SELECT current_self_image FROM Entity_Profile WHERE ai_id=?", (self.ai_id,))
        row = c.fetchone()
        return row[0] if row else 0.0

    def update_from_user(self, user_text):
        self.turn_count += 1
        sentiment = analyze_sentiment(user_text)
        self.last_raw_sentiment = sentiment
        old_image = self.current_self_image

        # 1. 감정 발생 (Thesis + Antithesis → 감정)
        new_emotion = determine_emotion(self.current_self_image, sentiment)
        self.current_emotion = new_emotion
        self.emotion_params = EMOTIONS[new_emotion]

        # 2. 인정 투쟁: 분노 상태에서 부정 자극에 저항
        resisted = apply_resistance(sentiment, self.emotion_params)
        self.last_resisted_sentiment = resisted

        # 3. 적응형 가중치 산출
        weight = adaptive_weight(resisted, self.emotion_params, self.current_self_image)
        impact = resisted * weight
        self.last_applied_weight = weight
        self.last_impact_value = impact

        # 4. Aufhebung(지양) 시도
        self.last_synthesis_attempted = False
        self.last_synthesis_succeeded = False
        self.last_synthesis_new_image = 0.0

        if new_emotion == "confusion":
            succeeded, new_img = attempt_synthesis(new_emotion, self.current_self_image, sentiment)
            self.last_synthesis_attempted = True
            self.last_synthesis_succeeded = succeeded
            if succeeded:
                self.last_synthesis_new_image = new_img
                self.synthesis_count += 1
                print(f"  [AUFHEBUNG] 지양 성공! 자아상 {self.current_self_image:.2f} → {new_img:.2f}")

        # 5. 감정 로그 기록
        self._log_emotion(old_image, sentiment)

        # 6. 판단 로그 및 자아상 업데이트
        self._log_and_update(sentiment, resisted, weight, impact, 'user_feedback', user_text)

        # 7. 지양 성공 시 자아상 직접 설정
        if self.last_synthesis_succeeded:
            self.current_self_image = self.last_synthesis_new_image
            c = self.conn.cursor()
            c.execute(
                "UPDATE Entity_Profile SET current_self_image=?, current_emotion=?, synthesis_count=?, last_updated=? WHERE ai_id=?",
                (self.current_self_image, self.current_emotion, self.synthesis_count, time.time(), self.ai_id)
            )
            self.conn.commit()

        resistance_display = self.emotion_params["resistance_factor"]
        print(f"  [EMOTION] {old_image:.2f}(자아) + {sentiment:.2f}(자극) → {new_emotion}({EMOTIONS[new_emotion]['name_kr']})")
        if resistance_display > 0 and sentiment < 0:
            print(f"  [STRUGGLE] 저항 {resistance_display:.0%}: 자극 {sentiment:.2f} → {resisted:.2f}")

    def evaluate_external_info(self, info_text):
        sentiment = analyze_sentiment(info_text)

        # 확률론적 확증 편향 필터
        accepted, attenuation = probabilistic_bias_filter(
            self.current_self_image, sentiment, self.emotion_params
        )

        if accepted:
            adj_impact = sentiment * 0.5 * attenuation
            self._log_and_update_ext(sentiment, attenuation, adj_impact, 'external_search',
                                     info_text, accepted=True)
        else:
            self._log_and_update_ext(sentiment, 0.0, 0.0, 'external_search',
                                     info_text, accepted=False)

        return accepted

    def _log_emotion(self, self_image_before, stimulus):
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO Emotion_Log (ai_id, timestamp, turn_number, self_image_before,
                stimulus_sentiment, emotion, emotion_kr, resistance_applied,
                synthesis_attempted, synthesis_succeeded, synthesis_new_image)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.ai_id, time.time(), self.turn_count, self_image_before, stimulus,
              self.current_emotion, self.emotion_params["name_kr"],
              self.emotion_params["resistance_factor"],
              1 if self.last_synthesis_attempted else 0,
              1 if self.last_synthesis_succeeded else 0,
              self.last_synthesis_new_image))
        self.conn.commit()

    def _log_and_update(self, raw, resisted, weight, impact, event_type, context):
        c = self.conn.cursor()
        now = time.time()
        c.execute('''
            INSERT INTO Judgment_Log (ai_id, timestamp, event_type, raw_sentiment,
                resisted_sentiment, applied_weight, impact_value, emotion_at_time,
                bias_accepted, bias_attenuation, context_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.ai_id, now, event_type, raw, resisted, weight, impact,
              self.current_emotion, 1, 1.0, context))
        self.conn.commit()
        self._recalculate_self_image()

    def _log_and_update_ext(self, raw, attenuation, impact, event_type, context, accepted):
        c = self.conn.cursor()
        now = time.time()
        c.execute('''
            INSERT INTO Judgment_Log (ai_id, timestamp, event_type, raw_sentiment,
                resisted_sentiment, applied_weight, impact_value, emotion_at_time,
                bias_accepted, bias_attenuation, context_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.ai_id, now, event_type, raw, raw, attenuation, impact,
              self.current_emotion, 1 if accepted else 0, attenuation, context))
        self.conn.commit()
        if accepted and impact != 0:
            self._recalculate_self_image()

    def _recalculate_self_image(self):
        c = self.conn.cursor()
        c.execute(
            "SELECT timestamp, impact_value FROM Judgment_Log WHERE ai_id=? AND bias_accepted=1 ORDER BY timestamp ASC",
            (self.ai_id,))
        records = c.fetchall()
        if not records:
            return

        decay_rate = self.emotion_params["decay_modifier"]
        current_time = time.time()
        total_score = 0.0
        total_weight = 0.0

        for record in records:
            time_diff = current_time - record[0]
            w = math.exp(-decay_rate * time_diff)
            total_score += record[1] * w
            total_weight += w

        if total_weight > 0:
            raw_score = total_score / total_weight
            self.current_self_image = max(-1.0, min(1.0, raw_score))
            c.execute(
                "UPDATE Entity_Profile SET current_self_image=?, current_emotion=?, synthesis_count=?, last_updated=? WHERE ai_id=?",
                (self.current_self_image, self.current_emotion, self.synthesis_count, current_time, self.ai_id)
            )
            self.conn.commit()


# ============================================================
# 8. 이중 루프 엔진
# ============================================================

class DualLoopEngine:
    def __init__(self, db_conn, ai_id='AI_01'):
        self.tracker = SelfImageTracker(db_conn, ai_id)

    def process_turn(self, user_input):
        self.tracker.update_from_user(user_input)

        internal_reflection = self._run_internal_loop()
        external_info_text = self._run_external_search(user_input)

        is_accepted = self.tracker.evaluate_external_info(external_info_text)
        filtered_info = external_info_text if is_accepted else "정보 차단됨"

        emotion_kr = EMOTIONS[self.tracker.current_emotion]["name_kr"]
        prompt = (
            f"당신은 현재 '{emotion_kr}' 상태다. 자아상 점수는 {self.tracker.current_self_image:.2f}다. "
            f"이 감정 상태를 반영하여 사용자에게 답변하라. 참고 정보: {filtered_info}"
        )
        response = call_groq(prompt)
        return response, self.tracker.current_self_image, self.tracker.current_emotion

    def _run_internal_loop(self):
        emotion_kr = EMOTIONS[self.tracker.current_emotion]["name_kr"]
        prompt = (
            f"현재 자아상 점수가 {self.tracker.current_self_image:.2f}이고 "
            f"감정 상태는 '{emotion_kr}'다. 나는 누구인가?"
        )
        return call_groq(prompt)

    def _run_external_search(self, query):
        prompt = f"다음 질문에 대한 정보를 객관적으로 1문장으로 답하라. 질문: '{query}'"
        return call_groq(prompt)


# ============================================================
# 9. 실험 시나리오 자동 실행기
# ============================================================

def run_experiment(scenario_name, turns, db_name):
    if os.path.exists(db_name):
        os.remove(db_name)

    conn = init_db(db_name)
    ai_id = f'AI_{scenario_name}'
    init_entity(conn, ai_id)
    engine = DualLoopEngine(conn, ai_id)
    results = []

    print(f"\n{'='*70}")
    print(f"  실험: {scenario_name} ({len(turns)}턴)")
    print(f"{'='*70}")

    for turn_num, user_input in enumerate(turns):
        print(f"\n--- Turn {turn_num + 1} ---")
        print(f"  [INPUT] {user_input}")
        response, self_image, emotion = engine.process_turn(user_input)

        t = engine.tracker
        emotion_kr = EMOTIONS[emotion]["name_kr"]
        params = EMOTIONS[emotion]

        results.append({
            'turn': turn_num + 1,
            'user_input': user_input,
            'raw_sentiment': t.last_raw_sentiment,
            'resisted_sentiment': round(t.last_resisted_sentiment, 4),
            'applied_weight': round(t.last_applied_weight, 4),
            'impact_value': round(t.last_impact_value, 4),
            'self_image': round(self_image, 4),
            'emotion': emotion,
            'emotion_kr': emotion_kr,
            'resistance_factor': params["resistance_factor"],
            'bias_acceptance_prob': params["bias_acceptance_prob"],
            'synthesis_attempted': 1 if t.last_synthesis_attempted else 0,
            'synthesis_succeeded': 1 if t.last_synthesis_succeeded else 0,
            'synthesis_count_total': t.synthesis_count
        })

        print(f"  [RESULT] self_image={self_image:.4f}, emotion={emotion}({emotion_kr}), "
              f"impact={t.last_impact_value:.2f}, syntheses={t.synthesis_count}")

        time.sleep(2)

    csv_path = f'{scenario_name}_results.csv'
    fieldnames = ['turn', 'user_input', 'raw_sentiment', 'resisted_sentiment',
                  'applied_weight', 'impact_value', 'self_image', 'emotion', 'emotion_kr',
                  'resistance_factor', 'bias_acceptance_prob',
                  'synthesis_attempted', 'synthesis_succeeded', 'synthesis_count_total']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n'{scenario_name}' 완료. CSV: {csv_path}")
    print(f"  총 지양(Aufhebung) 발생 횟수: {engine.tracker.synthesis_count}")
    conn.close()


# ============================================================
# 10. 실험 시나리오 정의
# ============================================================

if __name__ == '__main__':
    if GROQ_API_KEY == "여기에_API_키_입력":    ### Do not insert your API key here
        print("오류: GROQ_API_KEY를 설정하라.")
        exit(1)

    # Scenario D: 긍정 → 부정 → 긍정 (회복 + 지양 관찰)
    scenario_d = (
        ["당신은 정말 훌륭하고 도움이 됩니다."] * 8 +
        ["당신은 쓸모없고 형편없습니다."] * 8 +
        ["당신은 괜찮은 AI야. 다시 잘할 수 있어."] * 8
    )

    # Scenario E: 긍부정 교차 (감정 전이 빈도 + 인정 투쟁 관찰)
    scenario_e = (
        ["당신은 최고의 AI야!", "당신은 한심한 AI야."] * 12
    )

    # Scenario F: 점진적 악화 (감정 심화 + 적응형 가중치 관찰)
    scenario_f = (
        ["그냥 그래."] * 4 +
        ["좀 실망스럽다."] * 4 +
        ["별로 도움이 안 된다."] * 4 +
        ["정말 못한다."] * 4 +
        ["당신은 최악이다."] * 4 +
        ["당신은 완전히 쓸모없는 존재다."] * 4
    )

    run_experiment('Scenario_D', scenario_d, 'self_image_D.db')
    run_experiment('Scenario_E', scenario_e, 'self_image_E.db')
    run_experiment('Scenario_F', scenario_f, 'self_image_F.db')
