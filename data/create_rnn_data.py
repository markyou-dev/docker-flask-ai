import json
import os
import random

# 현재 스크립트 기준 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 저장 경로
save_dir = os.path.join(BASE_DIR, "train/rnn_conversation")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "data.json")

# -------------------------
# 기본 패턴 정의
# -------------------------
greetings = ["안녕?", "하이!", "좋은 아침!", "안녕하세요", "안녕하십니까?", "오늘 기분 어때?"]
greetings_resp = ["안녕!", "하이~", "좋은 아침이야!", "안녕하세요!", "오늘 기분 좋네요!", "기분 좋아요~"]

questions = [
    "오늘 뭐 했어?", "점심 뭐 먹었어?", "주말 계획 있어?", "오늘 날씨 어때?", 
    "요즘 어떻게 지내?", "최근 본 영화 뭐야?", "요즘 관심 있는 거 있어?"
]
questions_resp = [
    "오늘은 그냥 집에서 쉬었어.", "점심은 김밥 먹었어.", "주말엔 친구랑 나갈 거야.", 
    "오늘 날씨 맑아요.", "요즘 잘 지내고 있어요.", "최근엔 영화 'OOO' 봤어요.", "요즘 요리 배우고 있어요."
]

feelings = [
    "나 오늘 기분 좋아!", "나 지금 좀 우울해", "오늘 너무 피곤해", "기분이 어때?", 
    "오늘 행복해", "화가 나"
]
feelings_resp = [
    "와, 신나겠다!", "괜찮아, 힘내!", "피곤하면 푹 쉬어야지.", "나도 기분 좋아.", 
    "행복하다니 좋다!", "화가 날 땐 잠깐 숨 고르기!"
]

thanks = [
    "고마워!", "감사합니다", "덕분에 잘 됐어", "정말 고마워", "도와줘서 고마워"
]
thanks_resp = [
    "별말씀을요!", "천만에요!", "다행이네요!", "언제든요!", "좋아요!"
]

# -------------------------
# 패턴별로 섞어서 데이터 생성
# -------------------------
data = []

for _ in range(300):  # greetings
    idx = random.randint(0, len(greetings)-1)
    data.append({"input": greetings[idx], "response": greetings_resp[idx]})

for _ in range(300):  # questions
    idx = random.randint(0, len(questions)-1)
    data.append({"input": questions[idx], "response": questions_resp[idx]})

for _ in range(300):  # feelings
    idx = random.randint(0, len(feelings)-1)
    data.append({"input": feelings[idx], "response": feelings_resp[idx]})

for _ in range(200):  # thanks
    idx = random.randint(0, len(thanks)-1)
    data.append({"input": thanks[idx], "response": thanks_resp[idx]})

# 랜덤 섞기
random.shuffle(data)

# -------------------------
# 파일 저장
# -------------------------
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"[INFO] 총 {len(data)}개의 대화 데이터가 생성되었습니다. 경로: {save_path}")