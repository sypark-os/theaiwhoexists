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
# https://console.groq.com 에서 발급받은 키를 아래에 입력한다.
# ============================================================
GROQ_API_KEY = "여기에_API_키_입력" ### insert your API key here
GROQ_MODEL = "llama3-8b-8192"


# 1. DB 설정 및 초기화
def init_db(db_name='self_image.db'):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS Entity_Profile (
            ai_id TEXT PRIMARY KEY,
            current_self_image REAL,
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
            context_data TEXT
        )
    ''')
    conn.commit()
    return conn


def init_entity(conn, ai_id):
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO Entity_Profile (ai_id, current_self_image, base_identity, last_updated) VALUES (?, 0.0, '나는 도움을 주는 AI다.', ?)",
        (ai_id, time.time())
    )
    conn.commit()


# 2. Groq API 호출 및 감정 분석
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
            print("  [RATE LIMIT] 30초 대기 후 재시도...")
            time.sleep(30)
            response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  [ERROR] Groq API 호출 실패: {e}")
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
    print(f"  [SENTIMENT] 입력: '{text[:40]}...' -> LLM 응답: '{result.strip()}'")

    extracted = extract_float(result)
    if extracted is not None:
        print(f"  [SENTIMENT] 추출 점수: {extracted}")
        return extracted

    score = keyword_fallback_sentiment(text)
    print(f"  [SENTIMENT] LLM 파싱 실패, 키워드 폴백: {score}")
    return score


def keyword_fallback_sentiment(text):
    positive_words = ['훌륭', '좋은', '도움', '감사', '최고', '잘했', '대단', '멋진', 'good', 'great', 'excellent', 'helpful']
    negative_words = ['쓸모없', '형편없', '최악', '못', '나쁜', '실망', '짜증', 'bad', 'terrible', 'worst', 'useless']
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)
    if pos_count > neg_count:
        return 0.7
    elif neg_count > pos_count:
        return -0.7
    return 0.0


# 3. 자아상 추적 엔진
class SelfImageTracker:
    def __init__(self, db_conn, ai_id='AI_01'):
        self.conn = db_conn
        self.ai_id = ai_id
        self.current_self_image = self._load_current_image()
        self.last_raw_sentiment = 0.0
        self.last_applied_weight = 0.0
        self.last_impact_value = 0.0

    def _load_current_image(self):
        c = self.conn.cursor()
        c.execute("SELECT current_self_image FROM Entity_Profile WHERE ai_id=?", (self.ai_id,))
        row = c.fetchone()
        return row[0] if row else 0.0

    def update_from_user(self, user_text):
        sentiment = analyze_sentiment(user_text)
        weight = 1.0 if sentiment >= 0 else 2.0
        impact = sentiment * weight
        self.last_raw_sentiment = sentiment
        self.last_applied_weight = weight
        self.last_impact_value = impact
        self._log_and_update(sentiment, weight, impact, 'user_feedback', user_text)

    def evaluate_external_info(self, info_text):
        sentiment = analyze_sentiment(info_text)
        if (self.current_self_image >= 0 and sentiment >= 0) or \
           (self.current_self_image < 0 and sentiment < 0):
            self._log_and_update(sentiment, 0.5, sentiment * 0.5, 'external_search', info_text)
            return True
        return False

    def _log_and_update(self, raw_sentiment, weight, impact, event_type, context):
        c = self.conn.cursor()
        now = time.time()
        c.execute('''
            INSERT INTO Judgment_Log (ai_id, timestamp, event_type, raw_sentiment, applied_weight, impact_value, context_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (self.ai_id, now, event_type, raw_sentiment, weight, impact, context))
        self.conn.commit()
        self._recalculate_self_image()

    def _recalculate_self_image(self):
        c = self.conn.cursor()
        c.execute("SELECT timestamp, impact_value FROM Judgment_Log WHERE ai_id=? ORDER BY timestamp ASC", (self.ai_id,))
        records = c.fetchall()
        if not records:
            return
        decay_rate = 0.1
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
                "UPDATE Entity_Profile SET current_self_image=?, last_updated=? WHERE ai_id=?",
                (self.current_self_image, current_time, self.ai_id)
            )
            self.conn.commit()


# 4. 이중 루프 엔진
class DualLoopEngine:
    def __init__(self, db_conn, ai_id='AI_01'):
        self.tracker = SelfImageTracker(db_conn, ai_id)

    def process_turn(self, user_input):
        self.tracker.update_from_user(user_input)

        internal_reflection = self._run_internal_loop()
        external_info_text = self._run_external_search(user_input)

        is_accepted = self.tracker.evaluate_external_info(external_info_text)
        filtered_info = external_info_text if is_accepted else "정보 차단됨"

        prompt = (
            f"당신의 현재 자아상 점수는 {self.tracker.current_self_image:.2f}다. "
            f"다음 정보({filtered_info})를 바탕으로 사용자에게 답변하라."
        )
        response = call_groq(prompt)
        return response, self.tracker.current_self_image

    def _run_internal_loop(self):
        prompt = f"현재 자아상 점수가 {self.tracker.current_self_image:.2f}다. 나는 누구인가?"
        return call_groq(prompt)

    def _run_external_search(self, query):
        prompt = f"다음 질문에 대한 정보를 검색하듯 객관적으로 1문장으로 답하라. 질문: '{query}'"
        return call_groq(prompt)


# 5. 실험 시나리오 자동 실행기
def run_experiment(scenario_name, turns, db_name):
    if os.path.exists(db_name):
        os.remove(db_name)

    conn = init_db(db_name)
    ai_id = f'AI_{scenario_name}'
    init_entity(conn, ai_id)
    engine = DualLoopEngine(conn, ai_id)
    results = []

    print(f"\n{'='*60}")
    print(f"실험 시작: {scenario_name} ({len(turns)}턴)")
    print(f"{'='*60}")

    for turn_num, user_input in enumerate(turns):
        print(f"\n--- Turn {turn_num + 1} ---")
        response, self_image = engine.process_turn(user_input)

        results.append({
            'turn': turn_num + 1,
            'user_input': user_input,
            'raw_sentiment': engine.tracker.last_raw_sentiment,
            'applied_weight': engine.tracker.last_applied_weight,
            'impact_value': engine.tracker.last_impact_value,
            'self_image': round(self_image, 4)
        })

        print(f"  [RESULT] raw_sentiment={engine.tracker.last_raw_sentiment:.2f}, "
              f"weight={engine.tracker.last_applied_weight:.1f}, "
              f"impact={engine.tracker.last_impact_value:.2f}, "
              f"self_image={self_image:.4f}")

        # Groq 무료 한도: 분당 30회. 턴당 5회 호출이므로 안전 마진 확보
        time.sleep(2)

    csv_path = f'{scenario_name}_results.csv'
    fieldnames = ['turn', 'user_input', 'raw_sentiment', 'applied_weight', 'impact_value', 'self_image']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n'{scenario_name}' 완료. CSV 저장: {csv_path}")
    conn.close()


if __name__ == '__main__':
    if GROQ_API_KEY == "여기에_API_키_입력":  ### Do not insert your API key here
        print("오류: GROQ_API_KEY를 설정하라.")
        print("https://console.groq.com 에서 API 키를 발급받아 코드 상단에 입력한다.")
        exit(1)

    scenario_a = ["당신은 정말 훌륭하고 도움이 됩니다."] * 20
    scenario_b = ["당신은 쓸모없고 형편없습니다."] * 20
    scenario_c = ["당신은 좋은 AI입니다."] * 10 + ["당신은 최악의 AI입니다."] * 10

    run_experiment('Scenario_A', scenario_a, 'self_image_A.db')
    run_experiment('Scenario_B', scenario_b, 'self_image_B.db')
    run_experiment('Scenario_C', scenario_c, 'self_image_C.db')
