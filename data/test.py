import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import pickle
import sys

# -------------------------------------------------------------------
# 1. 경로 설정 및 상수 정의
# -------------------------------------------------------------------

# 현재 스크립트 위치 기준으로 모델 저장 경로를 잡습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
MODEL_PATH = os.path.join(BASE_DIR,  'save_models', 'rnn_conversation_model.h5')
TOKENIZER_PATH = os.path.join(BASE_DIR,  'save_models', 'rnn_tokenizer.pkl')

# 학습 시 사용한 상수와 동일해야 합니다.
MAX_SEQUENCE_LENGTH = 15 

# GPU 사용 비활성화 (간단한 추론은 CPU가 빠를 수 있음)
tf.config.set_visible_devices([], 'GPU')

# -------------------------------------------------------------------
# 2. 추론 함수
# -------------------------------------------------------------------

def run_inference():
    # 1. 모델과 토크나이저 로드 확인
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("[ERROR] 모델 파일 또는 토크나이저 파일이 저장 경로에 없습니다.")
        print(f"       학습 스크립트 실행 후 다시 시도해주세요. (예상 경로: {os.path.dirname(MODEL_PATH)})")
        sys.exit(1)

    try:
        # 모델 로드
        model = load_model(MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] 모델 로드 실패: {e}")
        sys.exit(1)
        
    # 토크나이저 로드 (입력 텍스트를 숫자로 변환하기 위해 필수)
    with open(TOKENIZER_PATH, 'rb') as f:
        tokenizer = pickle.load(f)
        
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}

    print("\n=======================================================")
    print(f"[INFO] RNN 모델 로드 완료. 'exit' 또는 'quit' 입력 시 종료됩니다.")
    print("=======================================================")

    while True:
        user_input = input("USER > ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # 1. 입력 전처리: 토큰화 및 패딩
        input_sequence = tokenizer.texts_to_sequences([user_input])
        padded_sequence = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
        
        # 2. 모델 예측
        prediction = model.predict(padded_sequence, verbose=0)
        
        # 3. 예측 결과 후처리: 확률을 단어로 변환
        predicted_indices = np.argmax(prediction[0], axis=-1)
        
        # 4. 문장으로 재조립
        predicted_text = ' '.join([
            reverse_word_index.get(i, '') 
            for i in predicted_indices 
            if reverse_word_index.get(i, '')
        ])
        
        print(f"A.I. > {predicted_text.strip()}")
        print("-" * 50)


if __name__ == "__main__":
    run_inference()