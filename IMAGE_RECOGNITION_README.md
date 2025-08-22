# 影像辨識模組使用指南

## 概述

這是一個基於 TensorFlow 和 MobileNet 的影像辨識模組，使用遷移學習技術來訓練自定義分類器。模組設計為可重用的組件，可以輕鬆整合到 FastAPI 後端中。

## 主要特性

- **遷移學習**: 使用預訓練的 MobileNet 模型作為特徵提取器
- **模組化設計**: 清晰的類別結構和接口設計
- **靈活配置**: 支援自定義類別和訓練參數
- **多格式支援**: 支援多種圖片格式（JPG, PNG, BMP, TIFF）
- **模型持久化**: 自動保存和載入訓練好的模型
- **異步支援**: 支援異步操作，適合 Web 應用

## 文件結構

```
├── image_recognition.py          # 主要影像辨識模組
├── image_recognition_config.py   # 配置文件
├── test_image_recognition.py     # 測試腳本
├── IMAGE_RECOGNITION_README.md   # 使用說明
├── CV-data/                      # 訓練數據目錄
│   └── fish/
│       ├── arowana/
│       │   ├── train/            # 訓練圖片
│       │   └── valid/            # 驗證圖片
│       └── tilapia/
│           ├── train/
│           └── valid/
└── models/                       # 模型保存目錄
    └── image_recognition/
        ├── classifier_model.h5   # 訓練好的分類器
        └── class_names.json      # 類別名稱
```

## 安裝依賴

```bash
pip install tensorflow opencv-python pillow numpy
```

## 快速開始

### 1. 基本使用

```python
import asyncio
from image_recognition import initialize_image_recognition_model

async def main():
    # 初始化模型
    model = await initialize_image_recognition_model()
    
    if model:
        # 獲取模型信息
        info = model.get_model_info()
        print(f"模型狀態: {info}")
        
        # 創建分類器（2個類別）
        model.create_classifier_model(2)
        
        # 載入訓練數據
        training_config = {
            "train_dataset": [
                {
                    "name": "吳郭魚",
                    "images": ["/吳郭魚/1.jpg", "/吳郭魚/2.jpg","/吳郭魚/3.jpg","/吳郭魚/4.jpg","/吳郭魚/5.jpg","/吳郭魚/6.jpg","/吳郭魚/7.jpg","/吳郭魚/8.jpg","/吳郭魚/9.jpg","/吳郭魚/10.jpg"]
                },
                {
                    "name": "銀龍魚", 
                    "images": ["/銀龍魚/1.jpg", "/銀龍魚/2.jpg","/銀龍魚/3.jpg","/銀龍魚/4.jpg","/銀龍魚/5.jpg","/銀龍魚/6.jpg","/銀龍魚/7.jpg","/銀龍魚/8.jpg","/銀龍魚/9.jpg","/銀龍魚/10.jpg"]
                }
            ]
        }
        
        model.load_training_data(training_config)
        
        # 訓練模型
        result = await model.train_model()
        print(f"訓練結果: {result}")
        
        # 預測圖片
        prediction = model.predict_image("path/to/test.jpg")
        print(f"預測結果: {prediction}")

# 運行
asyncio.run(main())
```

### 2. 使用配置文件

```python
from image_recognition_config import create_training_data_config, get_model_info

# 創建訓練數據配置
training_config = create_training_data_config()

# 獲取模型信息
model_info = get_model_info()
```

## API 接口

### ImageRecognitionModel 類別

#### 初始化
```python
model = ImageRecognitionModel(model_save_path="models/custom")
```

#### 主要方法

##### `load_mobilenet()`
載入預訓練的 MobileNet 模型
```python
success = await model.load_mobilenet()
```

##### `create_classifier_model(num_classes)`
創建分類器模型
```python
success = model.create_classifier_model(2)
```

##### `load_training_data(data_config)`
載入訓練數據
```python
data_config = {
    "classes": [
        {
            "name": "class1",
            "images": ["path/to/img1.jpg", "path/to/img2.jpg"]
        }
    ]
}
success = model.load_training_data(data_config)
```

##### `train_model()`
訓練模型
```python
result = await model.train_model()
# 返回: {"success": True, "final_accuracy": 0.95, ...}
```

##### `predict_image(image_path)`
預測單張圖片
```python
result = model.predict_image("path/to/image.jpg")
# 返回: {"success": True, "predicted_class": "arowana", "confidence": 0.95, ...}
```

##### `predict_base64_image(base64_data)`
預測 base64 編碼的圖片
```python
result = model.predict_base64_image(base64_string)
```

##### `get_model_info()`
獲取模型信息
```python
info = model.get_model_info()
```

##### `reset_model()`
重置模型狀態
```python
model.reset_model()
```

## 配置選項

### 模型參數
```python
IMAGE_RECOGNITION_CONFIG = {
    "model_params": {
        "input_width": 224,        # 輸入圖片寬度
        "input_height": 224,       # 輸入圖片高度
        "color_channels": 3,       # 顏色通道數
        "epochs": 10,              # 訓練輪數
        "batch_size": 5,           # 批次大小
        "learning_rate": 0.001,    # 學習率
        "validation_split": 0.2    # 驗證集比例
    }
}
```

### 數據增強
```python
"data_augmentation": {
    "rotation_range": 20,         # 旋轉範圍
    "width_shift_range": 0.2,     # 寬度偏移範圍
    "height_shift_range": 0.2,    # 高度偏移範圍
    "horizontal_flip": True,      # 水平翻轉
    "zoom_range": 0.2             # 縮放範圍
}
```

## 數據格式

### 訓練數據配置
```json
{
  "classes": [
    {
      "name": "arowana",
      "images": [
        "CV-data/fish/arowana/train/img1.jpg",
        "CV-data/fish/arowana/train/img2.jpg"
      ]
    },
    {
      "name": "tilapia",
      "images": [
        "CV-data/fish/tilapia/train/img1.jpg",
        "CV-data/fish/tilapia/train/img2.jpg"
      ]
    }
  ]
}
```

### 預測結果格式
```json
{
  "success": true,
  "predicted_class": "arowana",
  "confidence": 0.95,
  "class_probabilities": {
    "arowana": 0.95,
    "tilapia": 0.05
  },
  "all_predictions": [0.95, 0.05]
}
```

## 整合到 FastAPI

### 1. 在 main.py 中導入
```python
from image_recognition import initialize_image_recognition_model, image_recognition_model
```

### 2. 啟動時初始化
```python
@app.on_event("startup")
async def startup_event():
    # 初始化影像辨識模型
    await initialize_image_recognition_model()
```

### 3. 創建 API 端點
```python
@app.post("/api/image-recognition/train")
async def train_image_recognition_model(request: TrainingRequest):
    """訓練影像辨識模型"""
    if not image_recognition_model:
        raise HTTPException(status_code=500, detail="模型未初始化")
    
    # 載入訓練數據
    success = image_recognition_model.load_training_data(request.data_config)
    if not success:
        raise HTTPException(status_code=400, detail="訓練數據載入失敗")
    
    # 創建分類器
    image_recognition_model.create_classifier_model(len(request.data_config["classes"]))
    
    # 訓練模型
    result = await image_recognition_model.train_model()
    return result

@app.post("/api/image-recognition/predict")
async def predict_image(request: PredictionRequest):
    """預測圖片"""
    if not image_recognition_model or not image_recognition_model.is_trained:
        raise HTTPException(status_code=500, detail="模型未訓練")
    
    result = image_recognition_model.predict_image(request.image_path)
    return result
```

## 測試

### 運行測試腳本
```bash
python test_image_recognition.py
```

### 測試內容
1. 模型初始化
2. 分類器創建
3. 訓練數據載入
4. 模型訓練
5. 圖片預測
6. 配置函數

## 性能優化

### 1. 批次處理
- 使用適當的 `batch_size`
- 根據 GPU 記憶體調整

### 2. 數據預處理
- 圖片預先調整到正確尺寸
- 使用數據增強提高泛化能力

### 3. 模型保存
- 定期保存檢查點
- 使用模型量化減少檔案大小

## 故障排除

### 常見問題

1. **MobileNet 載入失敗**
   - 檢查網路連接
   - 確認 TensorFlow 版本

2. **記憶體不足**
   - 減少 `batch_size`
   - 減少圖片尺寸

3. **訓練準確率低**
   - 增加訓練數據
   - 調整學習率
   - 使用數據增強

4. **預測結果不準確**
   - 檢查訓練數據質量
   - 確認預測圖片格式
   - 重新訓練模型

## 擴展功能

### 1. 支援更多模型
- EfficientNet
- ResNet
- Vision Transformer

### 2. 多標籤分類
- 支援一張圖片多個標籤
- 修改損失函數

### 3. 實時預測
- 整合到影片串流
- 批次預測優化

## 授權

本模組基於 MIT 授權條款發布。

## 貢獻

歡迎提交 Issue 和 Pull Request 來改進這個模組。
