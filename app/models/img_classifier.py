import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

class ImgClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """모델 로드"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = load_model(self.model_path)

    def predict(self, image_path):
        """이미지 예측"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        if self.model is None:
            self.load_model()

        # 이미지 전처리
        img = image.load_img(image_path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        # 예측
        pred = self.model.predict(x)
        score = float(pred[0])

        # 결과 반환
        if score > 0.5:
            return "개", score
        else:
            return "고양이", 1 - score
