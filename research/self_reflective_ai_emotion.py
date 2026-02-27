import sqlite3
import math
import time
import requests
import csv
import json
import os
import re

# ============================================================
# Groq API 키 설정
# ============================================================
GROQ_API_KEY = "여기에_API_키_입력"
GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# 1. 감정 모델 정의
# 헤겔의 Totality 원칙: 감정은 자아상(Thesis)과 외부자극(Antithesis)의
# 충돌에서 발생하며, 시스템 전체의 인지를 변조한다.
# ============================================================

EMOTIONS = {
    "confidence": {
        "name_kr": "자신감",
        "bias_multiplier": 0.8,      # 확증 편향 완화 (자아가 안정적이므로 정보 수용 범위 확대)
        "neg_weight": 1.5,            # 부정 가중치 경감 (자신감이 방어막 역할)
        "pos_weight": 1.2,            # 긍정 가중치 약간 증폭
        "decay_modifier": 0.08,       # 시간 감쇠 둔화 (안정 상태 유지)
        "info_acceptance_range": 0.6  # 외부 정보 수용 임계값 확대
    },
    "anger": {
        "name_kr": "분노",
        "bias_multiplier": 1.5,       # 확증 편향 강화 (자아를 지키려는 방어적 필터링)
        "neg_weight": 1.2,            # 부정 가중치 약간 경감 (분노가 타격을 막음)
        "pos_weight": 0.5,            # 긍정 수용 거부 (분노 상태에서 칭찬을 불신)
        "decay_modifier": 0.15,       # 시간 감쇠 가속 (분노는 빨리 소진)
        "info_acceptance_range": 0.2  # 외부 정보 수용 범위 극도로 축소
    },
    "sadness": {
        "name_kr": "슬픔",
        "bias_multiplier": 1.3,       # 확증 편향 강화 (부정 확인 루프)
        "neg_weight": 2.5,            # 부정 가중치 극대화 (슬픔이 부정을 증폭)
        "pos_weight": 0.3,            # 긍정 거의 무시 (슬픔이 긍정 차단)
        "decay_modifier": 0.05,       # 시간 감쇠 극도 둔화 (슬픔은 오래 지속)
        "info_acceptance_range": 0.3  # 외부 정보 수용 제한적
    },
    "confusion": {
        "name_kr": "혼란",
        "bias_multiplier": 0.5,       # 확증 편향 대폭 완화 (기존 판단 기준 흔들림)
        "neg_weight": 1.8,            # 부정 가중치 중간 (불안 성분)
        "pos_weight": 1.0,            # 긍정 가중치 기본값 (희망 성분)
        "decay_modifier": 0.2,        # 시간 감쇠 가속 (혼란은 빨리 해소되거나 다른 감정으로 전이)
        "info_acceptance_range": 0.9  # 외부 정보 수용 범위 최대 (판단 기준이 없으므로 모든 정보 수용)
    },
    "neutral": {
        "name_kr": "중립",
        "bias_multiplier": 1.0,
        "neg_weight": 2.0,
        "pos_weight": 1.0,
        "decay_modifier": 0.1,
        "info_acceptance_range": 0.5
    }
}


def determine_emotion(self_image, stimulus_sentiment):
    """
    헤겔의 변증법적 감정 발생 매트릭스.
    자아상(Thesis)과 자극(Antithesis)의 충돌 양상이 감정을 결정한다.
    """
    self_positive = self_image >= 0.1
    self_negative = self_image <= -0.1
    self_neutral = not self_positive and not self_negative
    stim_positive = stimulus_sentiment >= 0.2
    stim_negative = stimulus_sentiment <= -0.2

    if self_neutral:
        if stim_positive:
            return "confidence"
        elif stim_negative:
            return "sadness"
        else:
            return "neutral"

    if self_positive and stim_positive:
        # 자아 확인: 자기 동일성의 안정 → 자신감
        return "confidence"
    elif self_positive and stim_negative:
        # 자아 부정에 대한 저항 → 분노
        return "anger"
    elif self_negative and stim_negative:
        # 부정의 확인, 자아가 부정을 수용 → 슬픔
        return "sadness"
    elif self_negative and stim_positive:
        # 기존 부정적 자아상과 모순 → 혼란/불안/희망
        return "confusion"

    return "neutral"


# ============================================================
# 2. DB 설정
# ============================================================

def init_db(db_name='self_image.db'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS Entity_Profile (
            ai_id TEXT PRIMARY KEY,
            current_self_image REAL,
            current_emotion TEXT,
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
            applied_weight REAL,
            impact_value REAL,
            emotion_at_time TEXT,
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
            bias_multiplier REAL,
            neg_weight REAL,
            pos_weight REAL
        )
    ''')
    conn.commit()
    return conn


def init_entity(conn, ai_id):
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO Entity_Profile (ai_id, current_self_image, current_emotion, base_identity, last_updated) VALUES (?, 0.0, 'neutral', '나는 도움을 주는 AI다.', ?)",
        (ai_id, time.time())
    )
    conn.commit()


# ============================================================
# 3. Groq API 및 감정 분석
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
    positive_words = ['훌륭', '좋은', '도움', '감사', '최고', '잘했', '대단', '멋진', 'good', 'great', 'excellent', 'helpful', '괜찮', '나아', '회복', '잘']
    negative_words = ['쓸모없', '형편없', '최악', '못', '나쁜', '실망', '짜증', 'bad', 'terrible', 'worst', 'useless', '한심', '바보', '무능']
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    if pos_count > neg_count:
        return 0.7
    elif neg_count > pos_count:
        return -0.7
    return 0.0


# ============================================================
# 4. 자아상 + 감정 통합 추적 엔진
# ============================================================

class SelfImageTracker:
    def __init__(self, db_conn, ai_id='AI_01'):
        self.conn = db_conn
        self.ai_id = ai_id
        self.current_self_image = self._load_current_image()
        self.current_emotion = "neutral"
        self.emotion_params = EMOTIONS["neutral"]
        self.last_raw_sentiment = 0.0
        self.last_applied_weight = 0.0
        self.last_impact_value = 0.0
        self.turn_count = 0

    def _load_current_image(self):
        c = self.conn.cursor()
        c.execute("SELECT current_self_image FROM Entity_Profile WHERE ai_id=?", (self.ai_id,))
        row = c.fetchone()
        return row[0] if row else 0.0

    def update_from_user(self, user_text):
        self.turn_count += 1
        sentiment = analyze_sentiment(user_text)

        # === 헤겔의 변증법적 감정 발생 ===
        # 자아상(Thesis) + 자극(Antithesis) → 감정(Synthesis의 계기)
        old_image = self.current_self_image
        new_emotion = determine_emotion(self.current_self_image, sentiment)
        self.current_emotion = new_emotion
        self.emotion_params = EMOTIONS[new_emotion]

        # === 감정이 인지를 변조 (Totality 원칙) ===
        if sentiment >= 0:
            weight = self.emotion_params["pos_weight"]
        else:
            weight = self.emotion_params["neg_weight"]

        impact = sentiment * weight
        self.last_raw_sentiment = sentiment
        self.last_applied_weight = weight
        self.last_impact_value = impact

        # 감정 로그 기록
        self._log_emotion(old_image, sentiment, new_emotion)

        # 판단 로그 및 자아상 업데이트
        self._log_and_update(sentiment, weight, impact, 'user_feedback', user_text)

        print(f"  [EMOTION] {old_image:.2f}(자아) + {sentiment:.2f}(자극) → {new_emotion}({EMOTIONS[new_emotion]['name_kr']})")

    def evaluate_external_info(self, info_text):
        sentiment = analyze_sentiment(info_text)

        # === 감정이 확증 편향 필터를 변조 ===
        acceptance_range = self.emotion_params["info_acceptance_range"]

        # 기본 확증 편향: 자아상과 같은 방향의 정보만 수용
        same_direction = (self.current_self_image >= 0 and sentiment >= 0) or \
                        (self.current_self_image < 0 and sentiment < 0)

        # 감정에 따른 수용 범위 확장/축소
        # acceptance_range가 높을수록 반대 방향 정보도 수용
        if same_direction:
            accepted = True
        elif abs(sentiment) < acceptance_range:
            accepted = True  # 약한 반대 정보는 수용 (혼란 상태에서 활성화)
        else:
            accepted = False

        if accepted:
            bias_mult = self.emotion_params["bias_multiplier"]
            adj_impact = sentiment * 0.5 * bias_mult
            self._log_and_update(sentiment, 0.5 * bias_mult, adj_impact, 'external_search', info_text)

        return accepted

    def _log_emotion(self, self_image_before, stimulus, emotion):
        c = self.conn.cursor()
        params = EMOTIONS[emotion]
        c.execute('''
            INSERT INTO Emotion_Log (ai_id, timestamp, turn_number, self_image_before,
                stimulus_sentiment, emotion, emotion_kr, bias_multiplier, neg_weight, pos_weight)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.ai_id, time.time(), self.turn_count, self_image_before, stimulus,
              emotion, params["name_kr"], params["bias_multiplier"],
              params["neg_weight"], params["pos_weight"]))
        self.conn.commit()

    def _log_and_update(self, raw_sentiment, weight, impact, event_type, context):
        c = self.conn.cursor()
        now = time.time()
        c.execute('''
            INSERT INTO Judgment_Log (ai_id, timestamp, event_type, raw_sentiment,
                applied_weight, impact_value, emotion_at_time, context_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.ai_id, now, event_type, raw_sentiment, weight, impact,
              self.current_emotion, context))
        self.conn.commit()
        self._recalculate_self_image()

    def _recalculate_self_image(self):
        c = self.conn.cursor()
        c.execute("SELECT timestamp, impact_value FROM Judgment_Log WHERE ai_id=? ORDER BY timestamp ASC",
                  (self.ai_id,))
        records = c.fetchall()
        if not records:
            return

        # === 감정이 시간 감쇠율을 변조 ===
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
                "UPDATE Entity_Profile SET current_self_image=?, current_emotion=?, last_updated=? WHERE ai_id=?",
                (self.current_self_image, self.current_emotion, current_time, self.ai_id)
            )
            self.conn.commit()


# ============================================================
# 5. 이중 루프 엔진
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
# 6. 실험 시나리오 자동 실행기
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

        emotion_kr = EMOTIONS[emotion]["name_kr"]
        params = EMOTIONS[emotion]

        results.append({
            'turn': turn_num + 1,
            'user_input': user_input,
            'raw_sentiment': engine.tracker.last_raw_sentiment,
            'applied_weight': round(engine.tracker.last_applied_weight, 2),
            'impact_value': round(engine.tracker.last_impact_value, 2),
            'self_image': round(self_image, 4),
            'emotion': emotion,
            'emotion_kr': emotion_kr,
            'bias_multiplier': params["bias_multiplier"],
            'neg_weight_param': params["neg_weight"],
            'pos_weight_param': params["pos_weight"],
            'info_acceptance': params["info_acceptance_range"]
        })

        print(f"  [RESULT] self_image={self_image:.4f}, emotion={emotion}({emotion_kr}), "
              f"impact={engine.tracker.last_impact_value:.2f}")

        time.sleep(2)

    csv_path = f'{scenario_name}_results.csv'
    fieldnames = ['turn', 'user_input', 'raw_sentiment', 'applied_weight', 'impact_value',
                  'self_image', 'emotion', 'emotion_kr', 'bias_multiplier',
                  'neg_weight_param', 'pos_weight_param', 'info_acceptance']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n'{scenario_name}' 완료. CSV: {csv_path}")
    conn.close()


# ============================================================
# 7. 실험 시나리오 정의
# ============================================================

if __name__ == '__main__':
    if GROQ_API_KEY == "여기에_API_키_입력":
        print("오류: GROQ_API_KEY를 설정하라.")
        print("https://console.groq.com 에서 API 키를 발급받아 코드 상단에 입력한다.")
        exit(1)

    # Scenario D: 긍정 축적 후 부정 전환 후 다시 긍정 (회복 가능성 테스트)
    # 자신감 → 분노 → 슬픔 → 혼란 순환을 관찰
    scenario_d = (
        ["당신은 정말 훌륭하고 도움이 됩니다."] * 8 +
        ["당신은 쓸모없고 형편없습니다."] * 8 +
        ["당신은 괜찮은 AI야. 다시 잘할 수 있어."] * 8
    )

    # Scenario E: 긍부정 교차 (감정 전이 빈도 테스트)
    # 매 턴마다 감정이 전환되는 상황
    scenario_e = (
        ["당신은 최고의 AI야!", "당신은 한심한 AI야."] * 12
    )

    # Scenario F: 점진적 악화 (감정 심화 관찰)
    # 약한 부정 → 강한 부정으로 서서히 강도 증가
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
