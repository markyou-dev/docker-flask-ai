import os

class Config:
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    DEBUG = bool(int(os.environ.get('FLASK_DEBUG', 1)))

    # app/ 폴더 경로
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))

    # 프로젝트 루트 (app 의 상위 폴더)
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

    # ------------------------
    # 모델 / 데이터 경로
    # ------------------------
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    MODEL_DIR = os.path.join(DATA_DIR, "save_models")

    # RNN
    RNN_MODEL_PATH = os.path.join(MODEL_DIR, "rnn_conversation_model.h5")
    RNN_TOKENIZER_PATH = os.path.join(MODEL_DIR, "rnn_tokenizer.pkl")

    # 이미지 분류 모델 경로
    IMG_MODEL_PATH = os.path.join(MODEL_DIR, "img_classifier_model.h5")

    # 업로드 / 로그
    UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
    LOG_FOLDER = os.path.join(ROOT_DIR, "logs")