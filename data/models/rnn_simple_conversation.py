import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# 스크립트 기준 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# 1. 설정 및 데이터 (작성자님의 5세 대화 패턴)
# -------------------------------------------------------------------

# (User Input, AI Response) 쌍
CONVERSATIONS = [
    ("안녕?", "안녕 반가워요."),
    ("나 오늘 기분 좋아!", "와아, 신나겠다!"),
    ("토끼는 뭘 먹을까?", "토끼는 당근을 좋아해요."),
    ("로봇이 뭐야?", "저는 여러분의 친구, 로봇이에요."),
    ("나 졸려", "잠깐 눈을 붙이는 게 좋겠어요."),
    ("고마워!", "별말씀을요!")
] * 30 # 학습 체감을 위해 데이터를 30회 반복하여 늘림 (수정: 30회로 늘림)

MAX_WORDS = 10000  # 사용할 단어의 최대 개수
MAX_SEQUENCE_LENGTH = 15 # 문장의 최대 길이
EMBEDDING_DIM = 100 # 임베딩 벡터의 차원 (단어의 의미를 저장하는 공간)
LSTM_UNITS = 128 # LSTM 층의 메모리 크기
EPOCHS = 50 
BATCH_SIZE = 4 # GTX 2060을 고려하여 작게 설정

# 모델 저장 경로 설정 (CNN 스크립트와 동일한 로직)
save_model_dir = os.path.join(BASE_DIR, '..', 'save_models')
os.makedirs(save_model_dir, exist_ok=True) # save_models 폴더 없으면 생성

# -------------------------------------------------------------------
# 2. 데이터 전처리 및 토큰화
# -------------------------------------------------------------------

def prepare_data(conversations):
    """
    대화 데이터를 토크나이징하고, RNN 학습에 적합한 형태로 변환합니다.
    """
    
    # 입력(X)과 출력(Y) 데이터 준비
    input_texts = [pair[0] for pair in conversations]
    target_texts = [pair[1] for pair in conversations]
    
    # 텍스트를 숫자로 변환하는 토크나이저 생성
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(input_texts + target_texts) # 모든 단어에서 어휘집 구축
    
    # 문장을 토큰 ID 시퀀스로 변환
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)
    
    # 시퀀스 길이를 통일 (padding)
    X = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    Y_sequences = pad_sequences(target_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Y를 원-핫 인코딩하여 RNN의 출력 형식(Multi-class Classification)에 맞춤
    vocab_size = len(tokenizer.word_index) + 1
    Y = np.zeros((len(Y_sequences), MAX_SEQUENCE_LENGTH, vocab_size), dtype='float32')
    
    for i, seq in enumerate(Y_sequences):
        for t, word_index in enumerate(seq):
            if word_index > 0:
                Y[i, t, word_index] = 1.0 # 해당 인덱스에 1.0 표시
                
    return X, Y, tokenizer, vocab_size

# -------------------------------------------------------------------
# 3. RNN 모델 구축 (Keras Sequential API)
# -------------------------------------------------------------------

def build_rnn_model(vocab_size):
    """
    가장 기본적인 시퀀스 투 시퀀스(Sequence-to-Sequence) 구조의 RNN 모델을 정의합니다.
    """
    model = Sequential([
        # 1. 임베딩 층 (단어 ID를 의미가 담긴 벡터로 변환)
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        
        # 2. LSTM 층 (RNN의 핵심 메모리 역할. 순차적으로 문맥을 기억)
        LSTM(LSTM_UNITS, return_sequences=True), # 다음 층으로 시퀀스를 계속 전달
        Dropout(0.2),
        
        # 3. 출력 층 (다음 단어 예측)
        #   - vocab_size 크기의 벡터를 출력하며, 가장 높은 확률을 가진 단어가 선택됨
        Dense(vocab_size, activation='softmax')
    ])
    
    # 모델 컴파일 (CNN 때와 마찬가지로 손실 함수와 최적화 함수 사용)
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
                  
    return model

# -------------------------------------------------------------------
# 4. 학습 및 예측 함수
# -------------------------------------------------------------------

def train_and_predict():
    X, Y, tokenizer, vocab_size = prepare_data(CONVERSATIONS)
    model = build_rnn_model(vocab_size)
    
    # 모델 학습 (CNN 때와 동일한 model.fit() 함수 사용)
    print("RNN 모델 학습 시작...")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # 모델 저장 (경로 지정)
    model_path = os.path.join(save_model_dir, 'rnn_conversation_model.h5')
    model.save(model_path)
    print(f"\n[INFO] 모델 저장 완료: {model_path}")

    # 간단한 예측 예시
    print("\n--- 학습 결과 예측 테스트 ---")
    input_text = "안녕?"
    
    # 입력 문장을 토큰화 및 패딩
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(input_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # 모델 예측
    prediction = model.predict(padded_sequence) # (1, 15, vocab_size) 형태의 행렬 출력
    
    # 예측 결과를 단어로 변환 (가장 높은 확률을 가진 단어 선택)
    predicted_indices = np.argmax(prediction[0], axis=-1)
    
    # 예측 단어 시퀀스를 문장으로 변환
    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
    predicted_text = ' '.join([reverse_word_index.get(i, '') for i in predicted_indices if reverse_word_index.get(i, '')])
    
    print(f"입력: {input_text}")
    print(f"예측: {predicted_text.strip()}")
    print("\n[NOTE] RNN의 한계로 예측 결과가 어색하거나, 빈 단어가 포함될 수 있습니다.")


if __name__ == "__main__":
    train_and_predict()