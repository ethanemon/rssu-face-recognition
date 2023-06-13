
import cv2
import tensorflow as tf
from keras import layers
import numpy as np
import os

# Функция для удаления фона из изображения
def remove_background(image):
    # Применение алгоритма выделения переднего плана GrabCut
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Применение маски для удаления фона
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image = image * mask[:, :, np.newaxis]

    return image

# Загрузка и предобработка данных для обучения нейронной сети
def load_and_preprocess_data():
    images = []
    labels = []
    student_names = os.listdir("photos")
    for i, path in enumerate(student_names):
        image = cv2.imread(os.path.join("photos", path))
        # Удаление фона из изображения
        image = remove_background(image)

        resized_image = cv2.resize(image, (64, 64))
        processed_image = resized_image.astype('float32') / 255.0
        images.append(processed_image)
        labels.append(i)  # Используем имя студента в качестве метки
    return np.array(images), np.array(labels)

# Определение архитектуры нейронной сети
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(104, activation='softmax'))  # Количество уникальных студентов

# Компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Проверка, существует ли сохраненная модель
if os.path.exists("model.h5"):
    # Загрузка сохраненной модели
    model = tf.keras.models.load_model("model.h5")
else:
    # Загрузка и предобработка данных
    train_images, train_labels = load_and_preprocess_data()

    # Обучение модели
    model.fit(train_images, train_labels, epochs=200)

    # Сохранение обученной модели
    model.save("model.h5")

# Функция для предобработки и распознавания лица на изображении
def recognize_face(image):
    # Удаление фона из изображения
    image = remove_background(image)

    resized_image = cv2.resize(image, (64, 64))
    processed_image = resized_image.astype('float32') / 255.0
    result = model.predict(np.expand_dims(processed_image, axis=0))
    if np.any(result[0]):
        max_index = result.argmax()
        max_value = result[0][max_index]
        print("Процент уверенности", max_value)
        student_names = os.listdir("photos")
        student_name = student_names[max_index]

        if max_value > 0.70:
            return max_index
        else:
            return None
    else:
        return None

# Функция для загрузки изображения с веб-камеры
def load_image_from_camera():
    capture = cv2.VideoCapture(0)  # Использование первой доступной камеры
    while True:
        ret, frame = capture.read()
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    return frame

