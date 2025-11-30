import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical # to_categorical 임포트 추가 (필수)
import os
import pickle # 👈 토크나이저 저장을 위해 추가

# 스크립트 기준 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------
# 1. 경로 설정 및 하이퍼파라미터 정의 (CNN 스타일 유지)
# -------------------------------------------------------------------

# (User Input, AI Response) 쌍
CONVERSATIONS = [
    ("안녕?", "안녕 반가워요."),
    ("나 오늘 기분 좋아!", "와아, 신나겠다!"),
    ("토끼는 뭘 먹을까?", "토끼는 당근을 좋아해요."),
    ("로봇이 뭐야?", "저는 여러분의 친구, 로봇이에요."),
    ("나 졸려", "잠깐 눈을 붙이는 게 좋겠어요."),
    ("고마워!", "별말씀을요!")
] * 30 # 학습 체감을 위해 데이터를 30회 반복하여 늘림

MAX_WORDS = 10000 
MAX_SEQUENCE_LENGTH = 15
EMBEDDING_DIM = 100 
LSTM_UNITS = 128 
EPOCHS = 30 # 학습 시간 단축을 위해 에포크 조정
BATCH_SIZE = 4 

# 모델 저장 경로 설정
save_model_dir = os.path.join(BASE_DIR, '..', 'save_models')
os.makedirs(save_model_dir, exist_ok=True) # save_models 폴더 없으면 생성

# -------------------------------------------------------------------
# 2. 데이터 전처리 및 토큰화
# -------------------------------------------------------------------

def prepare_data(conversations):
    """
    대화 데이터를 토크나이징하고, RNN 학습에 적합한 형태로 변환합니다.
    """
    input_texts = [pair[0] for pair in conversations]
    target_texts = [pair[1] for pair in conversations]
    
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<unk>")
    tokenizer.fit_on_texts(input_texts + target_texts) 
    
    input_sequences = tokenizer.texts_to_sequences(input_texts)
    target_sequences = tokenizer.texts_to_sequences(target_texts)
    
    X = pad_sequences(input_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    Y_sequences = pad_sequences(target_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Y를 원-핫 인코딩하여 RNN의 출력 형식에 맞춤
    vocab_size = len(tokenizer.word_index) + 1
    Y = to_categorical(Y_sequences, num_classes=vocab_size)
                
    return X, Y, tokenizer, vocab_size

# -------------------------------------------------------------------
# 3. RNN 모델 구축 및 학습
# -------------------------------------------------------------------

def build_rnn_model(vocab_size):
    """
    가장 기본적인 시퀀스 투 시퀀스(Sequence-to-Sequence) 구조의 RNN 모델을 정의합니다.
    """
    model = Sequential([
        Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        LSTM(LSTM_UNITS, return_sequences=True), 
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    # 👈 이 라인을 추가하여 모델을 수동으로 빌드합니다.
    # input_shape는 (배치 사이즈는 제외하고) (MAX_SEQUENCE_LENGTH) 입니다.
    model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
                  
    return model

# -------------------------------------------------------------------
# 4. 학습 실행 및 저장 (순수 학습 로직)
# -------------------------------------------------------------------

def main_train_and_save():
    X, Y, tokenizer, vocab_size = prepare_data(CONVERSATIONS)
    model = build_rnn_model(vocab_size)
    
    # 모델 학습 (CNN 때와 동일한 model.fit() 함수 사용)
    print(f"[INFO] RNN 모델 학습 시작. 파라미터 수: {model.count_params()}")
    model.fit(X, Y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    # 모델 저장
    model_path = os.path.join(save_model_dir, 'rnn_conversation_model.h5')
    model.save(model_path)
    print(f"\n[INFO] 모델 저장 완료: {model_path}")
    
    # 👈 추론 시 필수! 토크나이저 저장
    tokenizer_path = os.path.join(save_model_dir, 'rnn_tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"[INFO] 토크나이저 저장 완료: {tokenizer_path}")

if __name__ == "__main__":
    main_train_and_save()