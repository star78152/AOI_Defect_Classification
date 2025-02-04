# AOI瑕疵分類

## 說明
此應用是來自AIdea題目「AOI 瑕疵分類」。 <br>
自動光學檢測（Automated Optical Inspection，AOI）是一種高速且高精度的光學影像檢測技術，利用機器視覺作為標準檢測方法，克服傳統人工使用光學儀器檢測的局限。其應用範圍廣泛，涵蓋高科技產業的研發與製造品管，以及國防、民生、醫療、環保與電力等領域。希望藉由 AOI 技術提升生產品質。針對所提供的 AOI 影像資料，來判讀瑕疵的分類，藉以提升透過數據科學來加強 AOI 判讀之效能。

## 評估標準
評估方式採用計算與實際值的相符正確率（Accuracy）。公式如下：參與本議題研究者在提供瑕疵預測類別後，系統後台將定期批次處理以計算分數，評估方式採用計算與實際值的相符正確率（Accuracy）。<br>
公式如下： <br>
Accuracy = Number of correct predictions / Number of total predictions

# 方法
## 架構
使用Python tensorflow進行AOI影像瑕疵分類。 <br>
透過遷移學習方式，沿用VGG16架構進行神經網路訓練，僅使用Epchs 200即可達到訓練Accuracy
0.997、Validation Accuracy 0.9921效果。
| Epoch | Train Accuracy | Validation Accuracy |
|-------|--------------|---------------------|
| 196   | 0.9921       | 0.9881              |
| 197   | 0.9921       | 0.9881              |
| 198   | 0.9881       | 0.9941              |
| 199   | 0.9921       | 0.9921              |
| 200   | <span style="color:red">0.997</span> | <span style="color:red">0.9921</span> |

在Test資料集也可達到0.9903的成績。


## Images

![This is an alt text.](/model/training_history.png "This is a sample image.")
