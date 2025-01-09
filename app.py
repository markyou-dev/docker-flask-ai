from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import subprocess

app = Flask(__name__)

# 모델 로드 경로
MODEL_PATH = "cat_dog_model.h5"

# 업로드 디렉토리 설정
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# 모델이 없을 경우 학습 함수
def train_model():
    subprocess.run(["python", "train_cat_dog_model.py"])


# 모델이 없으면 학습
if not os.path.exists(MODEL_PATH):
    train_model()


# 메인
@app.route("/")
def index():
    return render_template("index.html")

# 결과화면
@app.route("/result", methods=["GET", "POST"])
def result():
    file = request.files["file"]
    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # 모델 로드s
        model = load_model(MODEL_PATH)

        # 예측 수행
        result, cat_confidence, dog_confidence = predict_image(filepath, model)

        return render_template(
            "result.html",
            image_path=filepath,
            result=result,
            cat_confidence=cat_confidence,
            dog_confidence=dog_confidence,
        )

def predict_image(filepath, model):
    image = load_img(filepath, target_size=(150, 150))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    dog_confidence = prediction[0][0]  # 개일 확률
    cat_confidence = 1 - dog_confidence  # 고양이일 확률

    result = "고양이" if cat_confidence > 0.5 else "개"
    return result, cat_confidence, dog_confidence


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7080, debug=True)
