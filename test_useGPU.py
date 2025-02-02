# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 14:03:27 2025

@author: Hao
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tqdm import tqdm  # 用於顯示進度條

# 1. **檢查是否有可用的 GPU**  
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    # 設定 TensorFlow 使用 GPU
    tf.config.set_visible_devices(gpus[0], 'GPU')  # 指定使用第一個 GPU
    print("✅ TensorFlow is using GPU")
else:
    print("❌ No GPU detected. Running on CPU.")

# 2. **讀取模型 (.pb)**
export_path = "D:/AIdea/AOI_defect_classification/aoi/model/vgg_epoch200/"
model = tf.saved_model.load(f"""{export_path}/model/""")

# **取得推論函數 (signature)**
infer = model.signatures["serving_default"]

# 3. **讀取測試集資料**
csv_path = "test.csv"
image_folder = "test_images/"
test_data = pd.read_csv(csv_path)
filenames = test_data["ID"].values  # 確保 CSV 檔案中包含 'ID' 欄位

# 4. **批次處理 (Batch Inference)**
BATCH_SIZE = 32  # 你可以調整這個數字來控制每次推論的影像數量
predictions = []
image_paths = []

# 使用 tqdm 來顯示進度條
for i in tqdm(range(0, len(filenames), BATCH_SIZE), desc="Predicting", unit="batch"):
    batch_filenames = filenames[i : i + BATCH_SIZE]
    batch_images = []

    for file in batch_filenames:
        img_path = os.path.join(image_folder, file)
        if not os.path.exists(img_path):
            print(f"❌ Warning: {img_path} not found!")
            continue

        img = load_img(img_path, target_size=(224, 224))  # VGG16 需要 224x224
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)  # 預處理
        batch_images.append(img_array)

    if not batch_images:  # 避免空陣列報錯
        continue

    batch_images = np.array(batch_images, dtype=np.float32)
    batch_images_tensor = tf.convert_to_tensor(batch_images)

    # 進行批次推論
    batch_preds = infer(batch_images_tensor)[list(infer.structured_outputs.keys())[0]].numpy()
    batch_classes = np.argmax(batch_preds, axis=1)

    predictions.extend(batch_classes)
    image_paths.extend(batch_filenames)

# 5. **儲存預測結果**
output_df = pd.DataFrame({"ID": image_paths, "Label": predictions})
output_df.to_csv(f"""{export_path}prediction_results.csv""", index=False)

print("✅ 測試集預測完成，結果已儲存為 'prediction_results.csv'！")
