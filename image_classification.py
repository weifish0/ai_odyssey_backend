import os
import json
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import cv2
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)
TRAINING_DATA_BASE_PATH = "CV-data/fish"

# 確保啟用 eager execution，避免張量在非 eager 環境下調用 .numpy() 失敗
try:
    tf.config.run_functions_eagerly(True)
except Exception:
    pass

class ImageClassificationModel:
    """影像辨識模型類別，使用 MobileNet 進行遷移學習"""
    
    # 修改 __init__
    def __init__(self, user_name: str, base_model: tf.keras.Model, model_base_path: str = "models"):
        """
        初始化影像辨識模型
        
        Args:
            user_name: 使用者名稱
            base_model: 全域共用的預訓練基礎模型 (MobileNet)
            model_base_path: 所有模型保存的根路徑
        """
        if base_model is None:
            raise ValueError("必須提供一個已載入的 base_model (MobileNet)！")
            
        self.user_name = user_name
        # 每個使用者的模型都保存在以其 user_name 命名的子目錄中
        self.model_save_path = Path(model_base_path) / self.user_name
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # 模型參數 (與之前相同)
        self.MOBILE_NET_INPUT_WIDTH = 224
        self.MOBILE_NET_INPUT_HEIGHT = 224
        self.COLOR_CHANNEL = 3
        self.EPOCHS_NUM = 10
        self.BATCH_SIZE = 10
        
        # 模型組件
        # `mobilenet` 現在從外部傳入，是全域共享的
        self.mobilenet = base_model 
        self.classifier_model = None
        self.class_names = ["吳郭魚", "銀龍魚"] # 因為固定，可以直接寫死
        
        # 訓練數據 (與之前相同)
        self.training_data_inputs = []
        self.training_data_outputs = []
        self.examples_count = []
        
        # 模型狀態
        self.is_trained = False
        # 紀錄訓練圖片路徑（用於產生驗證集）
        self.training_image_set = set()
        
        # 初始化時就創建好分類器結構
        # 因為你的類別是固定的 (吳郭魚, 銀龍魚)，所以 num_classes 永遠是 2
        self.create_classifier_model(num_classes=2)
        
        # 嘗試載入此使用者之前保存的模型
        self.load_saved_model()
    
    def create_classifier_model(self, num_classes: int) -> bool:
        """
        創建分類器模型 (這個方法現在可以在 __init__ 中直接調用)
        """
        try:
            logger.info(f"創建分類器模型，類別數: {num_classes}")
            
            # 創建分類器模型
            self.classifier_model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(576,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            # 編譯模型
            loss_function = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
            self.classifier_model.compile(
                optimizer='adam',
                loss=loss_function,
                metrics=['accuracy']
            )
            
            logger.info("分類器模型創建成功！")
            return True
            
        except Exception as e:
            logger.error(f"創建分類器模型失敗: {e}")
            return False
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        預處理圖片
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            np.ndarray: 預處理後的圖片特徵向量
        """
        try:
            # 讀取圖片
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"無法讀取圖片: {image_path}")
                return None
            
            # 轉換為 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 調整大小
            img = cv2.resize(img, (self.MOBILE_NET_INPUT_WIDTH, self.MOBILE_NET_INPUT_HEIGHT))
            
            # 正規化
            img = img.astype(np.float32) / 255.0
            
            # 擴展維度
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"圖片預處理失敗 {image_path}: {e}")
            return None
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """
        使用 MobileNet 提取圖片特徵
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            np.ndarray: 特徵向量
        """
        try:
            if self.mobilenet is None:
                logger.error("MobileNet 模型未載入")
                return None
            
            # 預處理圖片
            img = self.preprocess_image(image_path)
            if img is None:
                return None
            
            # 提取特徵
            features = self.mobilenet.predict(img, verbose=0)
            return features.flatten()
            
        except Exception as e:
            logger.error(f"特徵提取失敗 {image_path}: {e}")
            return None
    
    def load_training_data(self, data_config: Dict[str, Any]) -> bool:
        """
        載入訓練數據
        
        Args:
            data_config: 數據配置，格式如下：
{"train_dataset": [
        {
			"name": "銀龍魚",
			"images": [
				"/銀龍魚/1.jpg",
				"/銀龍魚/2.jpg",
			]
		},
		{
			"name": "吳郭魚",
			"images": [
				"/吳郭魚/1.jpg",
				"/吳郭魚/2.jpg",
			]
		}
	]}

            
        Returns:
            bool: 是否成功載入
        """
        try:
            logger.info("開始載入訓練數據...")
            
            # 清空現有數據
            self.training_data_inputs = []
            self.training_data_outputs = []
            self.examples_count = []
            self.class_names = []
            self.training_image_set = set()
            
            # 載入類別名稱
            for class_info in data_config["train_dataset"]:
                self.class_names.append(class_info["name"])
                self.examples_count.append(0)
            
            TRAINING_DATA_BASE_PATH = "CV-data/fish"
            
            # 載入每個類別的圖片
            for class_idx, class_info in enumerate(data_config["train_dataset"]):
                logger.info(f"載入類別 {class_info['name']} 的圖片...")
                
                for image_path in class_info["images"]:
                    image_path = TRAINING_DATA_BASE_PATH + image_path
                    if not os.path.exists(image_path):
                        logger.warning(f"圖片不存在: {image_path}")
                        continue
                    
                    # 提取特徵
                    features = self.extract_features(image_path)
                    if features is not None:
                        self.training_data_inputs.append(features)
                        self.training_data_outputs.append(class_idx)
                        self.examples_count[class_idx] += 1
                        # 紀錄此張圖片作為訓練集
                        self.training_image_set.add(image_path)
            
            # 檢查數據
            total_samples = len(self.training_data_inputs)
            if total_samples == 0:
                logger.error("沒有載入到任何訓練數據")
                return False
            
            logger.info(f"成功載入 {total_samples} 個訓練樣本")
            for i, class_name in enumerate(self.class_names):
                logger.info(f"  {class_name}: {self.examples_count[i]} 張圖片")
            
            return True
            
        except Exception as e:
            logger.error(f"載入訓練數據失敗: {e}")
            return False
    
    async def train_model(self) -> Dict[str, Any]:
        """
        訓練模型
        
        Returns:
            Dict[str, Any]: 訓練結果
        """
        try:
            if not self.training_data_inputs:
                raise ValueError("沒有訓練數據")
            
            if self.classifier_model is None:
                raise ValueError("分類器模型未創建")
            
            logger.info("開始訓練模型...")
            
            # 準備訓練數據
            X = np.array(self.training_data_inputs)
            y = tf.keras.utils.to_categorical(self.training_data_outputs, len(self.class_names))
            
            # 打亂數據
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # 構建驗證集：目錄中的所有圖片扣掉訓練清單
            val_inputs: list[np.ndarray] = []
            val_outputs: list[int] = []
            # 收集驗證集路徑，以便回傳
            val_images_by_class: Dict[str, List[str]] = {name: [] for name in self.class_names}

            for class_idx, class_name in enumerate(self.class_names):
                class_dir = os.path.join(TRAINING_DATA_BASE_PATH, class_name)
                if not os.path.isdir(class_dir):
                    logger.warning(f"類別資料夾不存在: {class_dir}")
                    continue
                try:
                    for fname in sorted(os.listdir(class_dir)):
                        if not fname.lower().endswith('.jpg'):
                            continue
                        full_path = os.path.join(class_dir, fname)
                        if full_path in self.training_image_set:
                            continue
                        features = self.extract_features(full_path)
                        if features is not None:
                            val_inputs.append(features)
                            val_outputs.append(class_idx)
                            # 回傳相對於資料根目錄的路徑，例如 "/吳郭魚/3.jpg"
                            val_images_by_class[class_name].append(f"/{class_name}/{fname}")
                except Exception as e:
                    logger.warning(f"掃描驗證集時發生錯誤於 {class_dir}: {e}")

            # 在訓練前，使用全新的 optimizer 重新編譯模型，避免舊 optimizer 的變數集合不一致
            current_num_classes = len(self.class_names)
            loss_function = 'binary_crossentropy' if current_num_classes == 2 else 'categorical_crossentropy'
            self.classifier_model.compile(
                optimizer=tf.keras.optimizers.Adam(),
                loss=loss_function,
                metrics=['accuracy']
            )

            use_explicit_val = len(val_inputs) > 0
            if use_explicit_val:
                X_val = np.array(val_inputs)
                y_val = tf.keras.utils.to_categorical(val_outputs, len(self.class_names))
                logger.info(f"驗證集樣本數: {len(X_val)}")
                history = self.classifier_model.fit(
                    X, y,
                    batch_size=self.BATCH_SIZE,
                    epochs=self.EPOCHS_NUM,
                    validation_data=(X_val, y_val),
                    verbose=1
                )
            else:
                logger.warning("未找到驗證集樣本，改用 validation_split=0.2")
                history = self.classifier_model.fit(
                    X, y,
                    batch_size=self.BATCH_SIZE,
                    epochs=self.EPOCHS_NUM,
                    validation_split=0.2,
                    verbose=1
                )
            
            # 保存模型
            model_path = self.model_save_path / "classifier_model.h5"
            self.classifier_model.save(str(model_path))
            
            # 保存類別名稱
            class_names_path = self.model_save_path / "class_names.json"
            with open(class_names_path, 'w', encoding='utf-8') as f:
                json.dump(self.class_names, f, ensure_ascii=False, indent=2)
            
            self.is_trained = True
            logger.info("模型訓練完成並保存！")
            
            # 安全地轉換歷史指標為 float，避免非 eager 環境出錯
            def to_safe_float(value):
                try:
                    if isinstance(value, (list, tuple)):
                        value = value[-1] if value else None
                    if value is None:
                        return None
                    # numpy/py float/int
                    try:
                        return float(value)
                    except Exception:
                        pass
                    # EagerTensor
                    if hasattr(value, 'numpy'):
                        try:
                            return float(value.numpy())
                        except Exception:
                            return None
                    return None
                except Exception:
                    return None

            final_acc = to_safe_float(history.history.get('accuracy'))
            final_loss = to_safe_float(history.history.get('loss'))
            val_acc = to_safe_float(history.history.get('val_accuracy'))
            val_loss = to_safe_float(history.history.get('val_loss'))

            return {
                "success": True,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "final_accuracy": final_acc,
                "final_loss": final_loss,
                "model_path": str(model_path),
                "class_names": self.class_names,
                "num_train_samples": int(len(X)),
                "num_val_samples": int(len(val_inputs)) if 'val_inputs' in locals() else 0,
                "val_dataset": [
                    {"name": name, "images": val_images_by_class.get(name, [])}
                    for name in self.class_names
                ]
            }
            
        except Exception as e:
            logger.error(f"模型訓練失敗: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        預測單張圖片
        
        Args:
            image_path: 圖片路徑
            
        Returns:
            Dict[str, Any]: 預測結果
        """
        try:
            if not self.is_trained or self.classifier_model is None:
                raise ValueError("模型未訓練")
            
            # 從前端傳入的路徑（例如 "/吳郭魚/5.jpg"）推斷期望類別
            expected_class = None
            try:
                # 只取第一層資料夾名稱作為類別
                # "/吳郭魚/5.jpg" -> "吳郭魚"
                stripped = image_path.strip("/")
                parts = stripped.split("/")
                if len(parts) >= 2:
                    expected_class = parts[0]
            except Exception:
                expected_class = None

            image_path = TRAINING_DATA_BASE_PATH + image_path
            
            # 提取特徵
            features = self.extract_features(image_path)
            if features is None:
                return {"success": False, "error": "無法提取圖片特徵"}
            
            # 預測
            features_expanded = np.expand_dims(features, axis=0)
            predictions = self.classifier_model.predict(features_expanded, verbose=0)
            
            # 獲取預測結果
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            is_correct = (predicted_class == expected_class) if expected_class else False
            
            # 獲取所有類別的機率
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "all_predictions": predictions[0].tolist(),
                "is_correct": bool(is_correct)
            }
            
        except Exception as e:
            logger.error(f"圖片預測失敗: {e}")
            return {"success": False, "error": str(e)}
    
    def predict_base64_image(self, base64_data: str) -> Dict[str, Any]:
        """
        預測 base64 編碼的圖片
        
        Args:
            base64_data: base64 編碼的圖片數據
            
        Returns:
            Dict[str, Any]: 預測結果
        """
        try:
            # 解碼 base64 圖片
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            
            # 轉換為 numpy 數組
            img_array = np.array(image)
            
            # 轉換為 RGB
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # 調整大小
            img_array = cv2.resize(img_array, (self.MOBILE_NET_INPUT_WIDTH, self.MOBILE_NET_INPUT_HEIGHT))
            
            # 正規化
            img_array = img_array.astype(np.float32) / 255.0
            
            # 擴展維度
            img_array = np.expand_dims(img_array, axis=0)
            
            # 提取特徵
            features = self.mobilenet.predict(img_array, verbose=0)
            features = features.flatten()
            
            # 預測
            features_expanded = np.expand_dims(features, axis=0)
            predictions = self.classifier_model.predict(features_expanded, verbose=0)
            
            # 獲取預測結果
            predicted_class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_idx])
            predicted_class = self.class_names[predicted_class_idx]
            
            # 獲取所有類別的機率
            class_probabilities = {}
            for i, class_name in enumerate(self.class_names):
                class_probabilities[class_name] = float(predictions[0][i])
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "all_predictions": predictions[0].tolist()
            }
            
        except Exception as e:
            logger.error(f"Base64 圖片預測失敗: {e}")
            return {"success": False, "error": str(e)}
    
    def load_saved_model(self) -> bool:
        """
        載入已保存的模型
        
        Returns:
            bool: 是否成功載入
        """
        try:
            model_path = self.model_save_path / "classifier_model.h5"
            class_names_path = self.model_save_path / "class_names.json"
            
            if not model_path.exists() or not class_names_path.exists():
                logger.warning(f"使用者 {self.user_name} 尚未訓練過模型，或模型文件不存在。")
                return False
            
            # 載入分類器模型
            self.classifier_model = tf.keras.models.load_model(str(model_path))
            
            # 載入類別名稱
            with open(class_names_path, 'r', encoding='utf-8') as f:
                self.class_names = json.load(f)
            
            self.is_trained = True
            logger.info(f"✅ 成功載入使用者 {self.user_name} 的已保存模型！")
            return True
            
        except Exception as e:
            logger.error(f"載入使用者 {self.user_name} 的模型失敗: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        獲取模型信息
        
        Returns:
            Dict[str, Any]: 模型信息
        """
        return {
            "is_mobilenet_loaded": self.mobilenet is not None,
            "is_classifier_created": self.classifier_model is not None,
            "is_trained": self.is_trained,
            "class_names": self.class_names,
            "training_samples": len(self.training_data_inputs),
            "examples_count": self.examples_count,
            "model_save_path": str(self.model_save_path)
        }
    
    def reset_model(self):
        """重置模型狀態"""
        self.training_data_inputs = []
        self.training_data_outputs = []
        self.examples_count = []
        self.class_names = []
        self.is_trained = False
        logger.info("模型狀態已重置")

# 全局模型實例
image_recognition_model = None

async def initialize_image_recognition_model():
    """初始化影像辨識模型"""
    global image_recognition_model
    
    try:
        image_recognition_model = ImageClassificationModel()
        
        # 載入 MobileNet
        success = await image_recognition_model.load_mobilenet()
        if not success:
            logger.error("MobileNet 載入失敗")
            return None
        
        # 嘗試載入已保存的模型
        image_recognition_model.load_saved_model()
        
        logger.info("影像辨識模型初始化完成")
        return image_recognition_model
        
    except Exception as e:
        logger.error(f"影像辨識模型初始化失敗: {e}")
        return None
