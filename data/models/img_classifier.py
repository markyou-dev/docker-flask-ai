import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 스크립트 기준 절대 경로
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 데이터 전처리
train_data_dir = os.path.join(BASE_DIR, '..', 'train', 'img_classifier')
save_model_dir = os.path.join(BASE_DIR, '..', 'save_models')
os.makedirs(save_model_dir, exist_ok=True)  # save_models 폴더 없으면 생성

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,         # 절대 경로로 변경
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 모델 학습
model.fit(train_generator, epochs=10)

# 모델 저장
model_path = os.path.join(save_model_dir, 'img_classifier.h5')
model.save(model_path)
print(f"[INFO] 모델 저장 완료: {model_path}")