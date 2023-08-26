import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.src.utils import load_img, img_to_array
import numpy as np

# 데이터 경로 설정
train_data_dir = 'training_data'
validation_data_dir = 'testing_data'
batch_size = 32


# 데이터 로딩 및 전처리
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # 모델의 기대 입력 크기로 조정
    img = img_to_array(img) / 255.0  # 이미지를 0-1 범위로 스케일 조정
    img = np.expand_dims(img, axis=0)
    return img


# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=load_and_preprocess_image)  # 전처리 함수 적용

validation_datagen = ImageDataGenerator(preprocessing_function=load_and_preprocess_image)

# 데이터 로딩 및 증강
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# MobileNetV2 모델 로딩 (사전 훈련된 가중치 사용)
base_model = MobileNetV2(weights='imagenet', include_top=False)

# 모델 아키텍처 수정
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# 사전 훈련된 층 동결
for layer in base_model.layers:
    layer.trainable = False

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# 모델 저장
model.save('image_classifier_model.h5')
