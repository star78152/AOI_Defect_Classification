# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 16:41:41 2025

@author: Hao
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

# 1. TensorFlow 使用 GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Using GPU:", gpus)
    except RuntimeError as e:
        print(e)

# 啟用 mixed precision，提高 GPU 訓練效能
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# 設定輸出目錄
epoch = 500
save_folder = f"""vgg_epoch{epoch}"""
output_dir = f"""D:/AIdea/AOI_defect_classification/aoi/model/{save_folder}"""
os.makedirs(output_dir, exist_ok=True)

# 2. 讀取資料
csv_path = "train.csv"  # CSV 文件的路徑
image_folder = "train_images/"  # 圖片資料夾的路徑

# 讀取 CSV 檔案
data = pd.read_csv(csv_path)

# 確保 CSV 檔包含 'ID' 和 'Label' 欄位
filenames = data['ID'].values
labels = data['Label'].values

# 將標籤轉換為數字類別
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# 讀取影像資料並調整大小
def load_image(file):
    img_path = image_folder + file
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))  # VGG16 要求 224x224
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(img_array)  # 使用 VGG16 的預處理
    return img_array

# 使用 NumPy 陣列加速影像讀取
images = np.array([load_image(file) for file in filenames], dtype=np.float32)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# 3. 使用 VGG16 模型進行遷移學習
with tf.device('/GPU:0'):  # 確保運行在 GPU
    vgg16_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg16_base.trainable = False  # 只訓練新加的 Dense 層

    model = models.Sequential([
        vgg16_base,
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(label_encoder.classes_), activation='softmax', dtype=tf.float32)  # 確保輸出層使用 float32
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

# 4. 資料增強與 tf.data pipeline
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow(X_train, y_train, batch_size=32)

# 5. 訓練日誌 Callback
class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file=os.path.join(output_dir, "training_log.csv")):
        super().__init__()
        self.log_file = log_file
        with open(self.log_file, "w") as f:
            f.write("Epoch,Train Loss,Train Accuracy,Validation Loss,Validation Accuracy\n")

    def on_epoch_end(self, epoch, logs=None):
        with open(self.log_file, "a") as f:
            f.write(f"{epoch + 1},{logs['loss']:.4f},{logs['accuracy']:.4f},"
                    f"{logs['val_loss']:.4f},{logs['val_accuracy']:.4f}\n")

log_callback = TrainingLogger()

# 6. 訓練模型
history = model.fit(train_generator,
                    validation_data=(X_test, y_test),
                    epochs=epoch,
                    callbacks=[log_callback])

# 7. 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# 8. 繪製並保存訓練曲線
plt.figure(figsize=(12, 4))

# 繪製準確率
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

# 繪製損失
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 保存圖表
plot_path = os.path.join(output_dir, "training_history.png")
plt.savefig(plot_path)
plt.show()

print(f"Training history plot saved as '{plot_path}'")

# 9. 儲存模型
model_save_path = os.path.join(output_dir, "model")
tf.saved_model.save(model, model_save_path)

print(f"Model saved at: {model_save_path}")
print(f"Training log saved as '{log_callback.log_file}'")

