# 앱 초기화 모듈
from flask import Flask
from .config import Config
from .routes import blueprints
from .models.img_classifier import ImgClassifier
from .models.rnn_conversation import RnnConversation
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 업로드 폴더 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # AI 모델 초기화
    # 이미지 모델
    app.classifier = ImgClassifier(app.config["IMG_MODEL_PATH"])

    # RNN 모델
    app.rnn = RnnConversation(
        model_path=app.config["RNN_MODEL_PATH"],
        tokenizer_path=app.config["RNN_TOKENIZER_PATH"]
    )

    # 블루프린트 등록
    for bp in blueprints:
        app.register_blueprint(bp)

    return app
