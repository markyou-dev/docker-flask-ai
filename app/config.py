import os
class Config:
    PORT = int(os.environ.get('FLASK_PORT', 5000))
    DEBUG = bool(int(os.environ.get('FLASK_DEBUG', 1)))
    UPLOAD_FOLDER = '/usr/src/uploads'
    MODEL_PATH = '/usr/src/data/save_models/img_classifier.h5'
