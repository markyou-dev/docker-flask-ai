# 앱 초기화 모듈
from flask import Flask
from .config import Config
from .routes import blueprints
from .models.img_classifier import ImgClassifier
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 업로드 폴더 생성
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # AI 모델 초기화
    app.classifier = ImgClassifier(app.config['MODEL_PATH'])

    # 블루프린트 등록
    for bp in blueprints:
        app.register_blueprint(bp)

    return app
