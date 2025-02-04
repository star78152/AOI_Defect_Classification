# AOI瑕疵分類

## 說明
此應用是來自AIdea題目「AOI 瑕疵分類」。 <br>
自動光學檢測（Automated Optical Inspection，AOI）是一種高速且高精度的光學影像檢測技術，利用機器視覺作為標準檢測方法，克服傳統人工使用光學儀器檢測的局限。其應用範圍廣泛，涵蓋高科技產業的研發與製造品管，以及國防、民生、醫療、環保與電力等領域。希望藉由 AOI 技術提升生產品質。針對所提供的 AOI 影像資料，來判讀瑕疵的分類，藉以提升透過數據科學來加強 AOI 判讀之效能。

## 評估標準
評估方式採用計算與實際值的相符正確率（Accuracy）。公式如下：參與本議題研究者在提供瑕疵預測類別後，系統後台將定期批次處理以計算分數，評估方式採用計算與實際值的相符正確率（Accuracy）。公式如下： <br>
$$
\test{Accuracy} = frac{\test{Number of correct predictions}}{\test{Number of total predictionsNumber of total} predictions}
$$

# 方法
## 架構
使用Python tensorflow進行AOI影像瑕疵分類。 <br>
