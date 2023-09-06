import numpy as np
from keras import models, layers
from keras.applications import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

# 데이터 경로 설정
train_data_dir = 'training_data/'
validation_data_dir = 'testing_data/'
batch_size = 32
input_shape = (224, 224)


# 데이터 로딩 및 전처리
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=input_shape)  # 모델의 기대 입력 크기로 조정
    img = img_to_array(img) / 255.0  # 이미지를 0-1 범위로 스케일 조정
    imgArray = np.expand_dims(img, axis=0)
    return imgArray


# 데이터 증강 설정
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)  # 전처리 함수 적용

validation_datagen = ImageDataGenerator(
    # preprocessing_function = load_and_preprocess_image
)

# 데이터 로딩 및 증강
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical')

# 모델 아키텍처 수정
model = models.Sequential([
    layers.Input(shape=(224, 224, 3,)),
    MobileNetV2(weights='imagenet', include_top=False),  # MobileNetV2 모델 로딩 (사전 훈련된 가중치 사용)
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_generator.num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# 모델 훈련
history = model.fit(
    train_generator,
    # steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,
    validation_data=validation_generator
    # validation_steps=validation_generator.samples // batch_size
)

# 모델 저장
model.save('image_classifier_model.keras')

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()
