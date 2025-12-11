import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os
import pickle
import json

# -------------------------------------------------------------------
# 1. 경로 및 하이퍼파라미터
# -------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, '..', 'train', 'rnn_conversation', 'data.json')
SAVE_MODEL_DIR = os.path.join(BASE_DIR, '..', 'save_models')
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

# -------------------------------------------------------------------
# 학습 하이퍼파라미터 (권장 세팅)
# -------------------------------------------------------------------

MAX_WORDS = 10000        
# 토크나이저가 고려할 최대 단어 수.
# - 자주 사용되는 상위 10,000 단어만 학습에 사용
# - 희귀 단어는 <unk>로 대체
# - 데이터가 적으면 MAX_WORDS를 5000~10000 정도로 낮춰도 충분

MAX_SEQUENCE_LENGTH = 15 
# 입력과 출력 시퀀스의 최대 길이
# - 너무 짧으면 문장이 잘릴 수 있음
# - 너무 길면 패딩이 많아지고 연산 부담 증가
# - 일반적인 짧은 대화문에서는 15~20 정도 권장

EMBEDDING_DIM = 100      
# 단어를 벡터로 변환할 때 차원 수
# - 값이 크면 의미 표현력이 높아지지만 모델 크기와 연산량 증가
# - 소규모 데이터에는 50~100 정도 적당

LSTM_UNITS = 128         
# LSTM 레이어 은닉 상태 크기
# - 값이 크면 모델이 더 많은 패턴을 학습 가능
# - 데이터가 적으면 과적합 위험 존재
# - 64~128 사이가 소규모 챗봇에 적합

EPOCHS = 30              
# 전체 데이터셋을 반복 학습하는 횟수
# - 값이 작으면 모델이 충분히 학습하지 못함
# - 값이 크면 과적합 위험
# - 데이터가 적으면 20~40 정도가 적당

BATCH_SIZE = 4           
# 한 번에 모델에 입력되는 샘플 수
# - 작으면 메모리 부담 적고 일반화에 유리
# - 크면 학습 속도 빨라짐
# - 소규모 데이터에서는 4~8 정도 추천

# -------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -------------------------------------------------------------------

def load_conversations(json_path, repeat=1):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"학습 데이터 파일이 없습니다: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    input_texts = [item['input'] for item in data] * repeat
    target_texts = [item['response'] for item in data] * repeat
    return input_texts, target_texts

def prepare_data(input_texts, target_texts):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(input_texts + target_texts) 
    
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)

    X = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    Y_seq = pad_sequences(target_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Y를 원-핫 인코딩하여 RNN의 출력 형식에 맞춤
    vocab_size = len(tokenizer.word_index) + 1
    Y = to_categorical(Y_seq, num_classes=vocab_size)

    return X, Y, tokenizer, vocab_size

# -------------------------------------------------------------------
# 3. RNN 모델 구축
# -------------------------------------------------------------------

def build_rnn_model(vocab_size):
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(LSTM_UNITS, return_sequences=True),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
    return model

# -------------------------------------------------------------------
# 4. 학습 및 저장
# -------------------------------------------------------------------

def main_train_and_save():
    input_texts, target_texts = load_conversations(DATA_PATH, 30)
    X, Y, tokenizer, vocab_size = prepare_data(input_texts, target_texts)

    model = build_rnn_model(vocab_size)
    print(f"[INFO] RNN 모델 학습 시작. 파라미터 수: {model.count_params()}")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    # 모델 저장
    model_path = os.path.join(SAVE_MODEL_DIR, 'rnn_conversation_model.h5')
    model.save(model_path)
    print(f"[INFO] 모델 저장 완료: {model_path}")

    # 토크나이저 저장
    tokenizer_path = os.path.join(SAVE_MODEL_DIR, 'rnn_tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] 토크나이저 저장 완료: {tokenizer_path}")

if __name__ == "__main__":
    main_train_and_save()
