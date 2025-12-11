import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class RnnConversation:
    def __init__(self, model_path, tokenizer_path, max_len=20):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_len = max_len

        self.model = None
        self.tokenizer = None

        self.load_model()
        self.load_tokenizer()

    # ------------------------
    # 모델 로드
    # ------------------------
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RNN model not found: {self.model_path}")
        self.model = load_model(self.model_path)

    # ------------------------
    # 토크나이저 로드
    # ------------------------
    def load_tokenizer(self):
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")

        with open(self.tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

    # ------------------------
    # 예측 함수
    # ------------------------
    def predict(self, text):
        seq = self.tokenizer.texts_to_sequences([text])
        seq = pad_sequences(seq, maxlen=self.max_len, padding='post')

        pred = self.model.predict(seq)

        # pred shape = (1, max_len, vocab_size)
        predicted_indices = np.argmax(pred[0], axis=-1)

        # reverse mapping 사용
        reverse_index = {v: k for k, v in self.tokenizer.word_index.items()}

        # 문장 생성
        words = [
            reverse_index.get(i, "") 
            for i in predicted_indices 
            if reverse_index.get(i, "")
        ]

        return " ".join(words).strip()

    # ------------------------
    # (선택) 역매핑 함수
    # tokenizer.word_index 는 {단어: 인덱스}
    # 예측 시 필요한 {인덱스: 단어} 구조를 만든다
    # ------------------------
    def _index_to_word(self, index):
        return self.tokenizer.index_word.get(index, "")
