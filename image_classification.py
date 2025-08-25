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
        
        # 改進的模型參數
        self.MOBILE_NET_INPUT_WIDTH = 224
        self.MOBILE_NET_INPUT_HEIGHT = 224
        self.COLOR_CHANNEL = 3
        self.EPOCHS_NUM = 20  # 增加訓練輪數，配合早停機制
        self.BATCH_SIZE = 6   # 適當增加批次大小
        self.PATIENCE = 10    # 早停耐心值
        
        # 模型組件
        self.mobilenet = base_model 
        self.classifier_model = None
        self.class_names = ["吳郭魚", "銀龍魚"]
        
        # 訓練數據
        self.training_data_inputs = []
        self.training_data_outputs = []
        self.examples_count = []
        
        # 模型狀態
        self.is_trained = False
        self.training_image_set = set()
        
        # 獲取 MobileNet 的實際輸出特徵維度
        self.feature_dim = self._get_mobilenet_feature_dim()
        
        # 初始化時創建分類器結構
        self.create_classifier_model(num_classes=2)
        
        # 嘗試載入已保存的模型
        self.load_saved_model()
    
    def _get_mobilenet_feature_dim(self) -> int:
        """獲取 MobileNet 的實際輸出特徵維度"""
        try:
            # 創建一個測試輸入
            dummy_input = tf.zeros((1, self.MOBILE_NET_INPUT_WIDTH, self.MOBILE_NET_INPUT_HEIGHT, 3))
            features = self.mobilenet(dummy_input)
            feature_dim = features.shape[-1]
            logger.info(f"MobileNet 輸出特徵維度: {feature_dim}")
            return int(feature_dim)
        except Exception as e:
            logger.warning(f"無法獲取特徵維度，使用預設值: {e}")
            return 576  # 預設值
    
    def create_classifier_model(self, num_classes: int) -> bool:
        """
        創建改進的分類器模型
        """
        try:
            logger.info(f"創建分類器模型，類別數: {num_classes}, 特徵維度: {self.feature_dim}")
            
            # 改進的分類器架構
            self.classifier_model = tf.keras.Sequential([
                # 第一層：適應特徵維度
                tf.keras.layers.Dense(256, activation='relu', input_shape=(self.feature_dim,)),
                tf.keras.layers.BatchNormalization(),  # 添加批標準化
                tf.keras.layers.Dropout(0.3),          # 減少 dropout 率
                
                # 第二層：中等複雜度
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                
                # 第三層：較小複雜度
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                
                # 輸出層
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            # 使用改進的優化器設定
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,  # 降低學習率
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            )
            
            # 編譯模型
            loss_function = 'binary_crossentropy' if num_classes == 2 else 'categorical_crossentropy'
            self.classifier_model.compile(
                optimizer=optimizer,
                loss=loss_function,
                metrics=['accuracy']
            )
            
            logger.info("改進的分類器模型創建成功！")
            return True
            
        except Exception as e:
            logger.error(f"創建分類器模型失敗: {e}")
            return False
    
    def preprocess_image(self, image_path: str, augment: bool = False) -> Optional[np.ndarray]:
        """
        改進的圖片預處理，支援數據增強
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
            
            # 數據增強（僅在訓練時）
            if augment:
                img = self._augment_image(img)
            
            # 正規化
            img = img.astype(np.float32) / 255.0
            
            # 擴展維度
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"圖片預處理失敗 {image_path}: {e}")
            return None
    
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """簡單的數據增強"""
        try:
            # 隨機水平翻轉
            if np.random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            # 隨機亮度調整
            if np.random.random() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # 隨機對比度調整
            if np.random.random() > 0.5:
                contrast = np.random.uniform(0.8, 1.2)
                img = np.clip(((img - 128) * contrast) + 128, 0, 255).astype(np.uint8)
            
            # 隨機旋轉（小角度）
            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                height, width = img.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, rotation_matrix, (width, height))
            
            return img
            
        except Exception as e:
            logger.warning(f"數據增強失敗: {e}")
            return img
    
    def extract_features(self, image_path: str, augment: bool = False) -> Optional[np.ndarray]:
        """
        使用 MobileNet 提取圖片特徵，支援數據增強
        """
        try:
            if self.mobilenet is None:
                logger.error("MobileNet 模型未載入")
                return None
            
            # 預處理圖片
            img = self.preprocess_image(image_path, augment=augment)
            if img is None:
                return None
            
            # 提取特徵
            features = self.mobilenet.predict(img, verbose=0)
            
            # 檢查特徵維度
            if features.shape[-1] != self.feature_dim:
                logger.warning(f"特徵維度不匹配: 期望 {self.feature_dim}, 實際 {features.shape[-1]}")
                # 如果維度不匹配，嘗試調整
                if len(features.shape) == 4:  # (1, H, W, C)
                    features = tf.keras.layers.GlobalAveragePooling2D()(features)
                elif len(features.shape) == 3:  # (1, H, C)
                    features = tf.keras.layers.GlobalAveragePooling1D()(features)
            
            # 確保特徵是一維的
            features = features.flatten()
            
            # 檢查最終維度
            if features.shape[0] != self.feature_dim:
                logger.warning(f"最終特徵維度不匹配: 期望 {self.feature_dim}, 實際 {features.shape[0]}")
                # 如果維度仍然不匹配，強制調整
                if features.shape[0] > self.feature_dim:
                    features = features[:self.feature_dim]
                else:
                    # 用零填充
                    padded_features = np.zeros(self.feature_dim)
                    padded_features[:features.shape[0]] = features
                    features = padded_features
            
            return features
            
        except Exception as e:
            logger.error(f"特徵提取失敗 {image_path}: {e}")
            return None
    
    def load_training_data(self, data_config: Dict[str, Any]) -> bool:
        """
        載入訓練數據，支援數據增強
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
            
            # 載入每個類別的圖片
            for class_idx, class_info in enumerate(data_config["train_dataset"]):
                logger.info(f"載入類別 {class_info['name']} 的圖片...")
                
                for image_path in class_info["images"]:
                    image_path = TRAINING_DATA_BASE_PATH + image_path
                    if not os.path.exists(image_path):
                        logger.warning(f"圖片不存在: {image_path}")
                        continue
                    
                    # 提取原始特徵
                    features = self.extract_features(image_path, augment=False)
                    if features is not None:
                        self.training_data_inputs.append(features)
                        self.training_data_outputs.append(class_idx)
                        self.examples_count[class_idx] += 1
                        self.training_image_set.add(image_path)
                        
                        # 數據增強：為每張圖片生成增強版本
                        if self.examples_count[class_idx] <= 10:  # 只對前10張圖片進行增強
                            for _ in range(2):  # 每張圖片生成2個增強版本
                                augmented_features = self.extract_features(image_path, augment=True)
                                if augmented_features is not None:
                                    self.training_data_inputs.append(augmented_features)
                                    self.training_data_outputs.append(class_idx)
                                    self.examples_count[class_idx] += 1
            
            # 檢查數據
            total_samples = len(self.training_data_inputs)
            if total_samples == 0:
                logger.error("沒有載入到任何訓練數據")
                return False
            
            logger.info(f"成功載入 {total_samples} 個訓練樣本（包含增強數據）")
            for i, class_name in enumerate(self.class_names):
                logger.info(f"  {class_name}: {self.examples_count[i]} 張圖片")
            
            return True
            
        except Exception as e:
            logger.error(f"載入訓練數據失敗: {e}")
            return False
    
    async def train_model(self) -> Dict[str, Any]:
        """
        改進的模型訓練方法
        """
        try:
            if not self.training_data_inputs:
                raise ValueError("沒有訓練數據")
            
            if self.classifier_model is None:
                raise ValueError("分類器模型未創建")
            
            logger.info("開始訓練改進的模型...")
            
            # 準備訓練數據
            X = np.array(self.training_data_inputs)
            y = tf.keras.utils.to_categorical(self.training_data_outputs, len(self.class_names))
            
            # 數據標準化
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1  # 避免除零
            X = (X - X_mean) / X_std
            
            # 保存標準化參數
            self.normalization_params = {
                'mean': X_mean.tolist(),
                'std': X_std.tolist()
            }
            
            # 打亂數據
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # 構建驗證集
            val_inputs: list[np.ndarray] = []
            val_outputs: list[int] = []
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
                        features = self.extract_features(full_path, augment=False)
                        if features is not None:
                            # 應用相同的標準化
                            features_normalized = (features - X_mean) / X_std
                            val_inputs.append(features_normalized)
                            val_outputs.append(class_idx)
                            val_images_by_class[class_name].append(f"/{class_name}/{fname}")
                except Exception as e:
                    logger.warning(f"掃描驗證集時發生錯誤於 {class_dir}: {e}")

            # 重新編譯模型，使用改進的優化器設定
            current_num_classes = len(self.class_names)
            loss_function = 'binary_crossentropy' if current_num_classes == 2 else 'categorical_crossentropy'
            
            # 使用學習率調度
            initial_lr = 0.001
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
            
            # 早停機制
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
            
            # 模型檢查點
            checkpoint_path = self.model_save_path / "best_model.h5"
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            
            self.classifier_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                loss=loss_function,
                metrics=['accuracy']
            )

            # 訓練模型
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
                    callbacks=[early_stopping, lr_scheduler, model_checkpoint],
                    verbose=1
                )
            else:
                logger.warning("未找到驗證集樣本，改用 validation_split=0.2")
                history = self.classifier_model.fit(
                    X, y,
                    batch_size=self.BATCH_SIZE,
                    epochs=self.EPOCHS_NUM,
                    validation_split=0.2,
                    callbacks=[early_stopping, lr_scheduler, model_checkpoint],
                    verbose=1
                )
            
            # 保存最終模型
            final_model_path = self.model_save_path / "classifier_model.h5"
            self.classifier_model.save(str(final_model_path))
            
            # 保存類別名稱和標準化參數
            class_names_path = self.model_save_path / "class_names.json"
            with open(class_names_path, 'w', encoding='utf-8') as f:
                json.dump(self.class_names, f, ensure_ascii=False, indent=2)
            
            normalization_path = self.model_save_path / "normalization_params.json"
            with open(normalization_path, 'w', encoding='utf-8') as f:
                json.dump(self.normalization_params, f, ensure_ascii=False, indent=2)
            
            self.is_trained = True
            logger.info("改進的模型訓練完成並保存！")
            
            # 獲取訓練結果
            def to_safe_float(value):
                try:
                    if isinstance(value, (list, tuple)):
                        value = value[-1] if value else None
                    if value is None:
                        return None
                    try:
                        return float(value)
                    except Exception:
                        pass
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
            
            # 獲取最佳驗證準確率
            best_val_acc = max(history.history.get('val_accuracy', [0]))
            best_epoch = np.argmax(history.history.get('val_accuracy', [0])) + 1

            return {
                "success": True,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "final_accuracy": final_acc,
                "final_loss": final_loss,
                "best_val_accuracy": float(best_val_acc),
                "best_epoch": int(best_epoch),
                "total_epochs": len(history.history.get('loss', [])),
                "model_path": str(final_model_path),
                "class_names": self.class_names,
                "num_train_samples": int(len(X)),
                "num_val_samples": int(len(val_inputs)) if 'val_inputs' in locals() else 0,
                "val_dataset": [
                    {"name": name, "images": val_images_by_class.get(name, [])}
                    for name in self.class_names
                ],
                "training_improvements": {
                    "feature_dimension": self.feature_dim,
                    "data_augmentation": True,
                    "early_stopping": True,
                    "learning_rate_scheduling": True,
                    "batch_normalization": True
                }
            }
            
        except Exception as e:
            logger.error(f"模型訓練失敗: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def predict_image(self, image_path: str) -> Dict[str, Any]:
        """
        預測單張圖片，支援標準化處理
        """
        try:
            if not self.is_trained or self.classifier_model is None:
                raise ValueError("模型未訓練")
            
            # 從前端傳入的路徑推斷期望類別
            expected_class = None
            try:
                stripped = image_path.strip("/")
                parts = stripped.split("/")
                if len(parts) >= 2:
                    expected_class = parts[0]
            except Exception:
                expected_class = None

            image_path = TRAINING_DATA_BASE_PATH + image_path
            
            # 提取特徵
            features = self.extract_features(image_path, augment=False)
            if features is None:
                return {"success": False, "error": "無法提取圖片特徵"}
            
            # 應用標準化（如果可用）
            if hasattr(self, 'normalization_params') and self.normalization_params:
                features = (features - np.array(self.normalization_params['mean'])) / np.array(self.normalization_params['std'])
            
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
            
            # 計算預測的確定性
            max_prob = max(predictions[0])
            second_max_prob = sorted(predictions[0])[-2]
            prediction_certainty = max_prob - second_max_prob
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "all_predictions": predictions[0].tolist(),
                "is_correct": bool(is_correct),
                "prediction_certainty": float(prediction_certainty),
                "expected_class": expected_class
            }
            
        except Exception as e:
            logger.error(f"圖片預測失敗: {e}")
            return {"success": False, "error": str(e)}
    
    def predict_base64_image(self, base64_data: str) -> Dict[str, Any]:
        """
        預測 base64 編碼的圖片，支援標準化處理
        """
        try:
            if not self.is_trained or self.classifier_model is None:
                raise ValueError("模型未訓練")
            
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
            
            # 處理特徵維度
            if features.shape[-1] != self.feature_dim:
                if len(features.shape) == 4:
                    features = tf.keras.layers.GlobalAveragePooling2D()(features)
                elif len(features.shape) == 3:
                    features = tf.keras.layers.GlobalAveragePooling1D()(features)
            
            features = features.flatten()
            
            # 檢查並調整特徵維度
            if features.shape[0] != self.feature_dim:
                if features.shape[0] > self.feature_dim:
                    features = features[:self.feature_dim]
                else:
                    padded_features = np.zeros(self.feature_dim)
                    padded_features[:features.shape[0]] = features
                    features = padded_features
            
            # 應用標準化（如果可用）
            if hasattr(self, 'normalization_params') and self.normalization_params:
                features = (features - np.array(self.normalization_params['mean'])) / np.array(self.normalization_params['std'])
            
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
            
            # 計算預測的確定性
            max_prob = max(predictions[0])
            second_max_prob = sorted(predictions[0])[-2]
            prediction_certainty = max_prob - second_max_prob
            
            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "all_predictions": predictions[0].tolist(),
                "prediction_certainty": float(prediction_certainty)
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
            
            # 嘗試載入標準化參數
            normalization_path = self.model_save_path / "normalization_params.json"
            if normalization_path.exists():
                with open(normalization_path, 'r', encoding='utf-8') as f:
                    self.normalization_params = json.load(f)
                logger.info(f"✅ 成功載入使用者 {self.user_name} 的標準化參數！")
            else:
                logger.warning(f"使用者 {self.user_name} 的標準化參數文件不存在，將使用預設值。")

            self.is_trained = True
            logger.info(f"✅ 成功載入使用者 {self.user_name} 的已保存模型！")
            return True
            
        except Exception as e:
            logger.error(f"載入使用者 {self.user_name} 的模型失敗: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        獲取改進的模型信息
        """
        info = {
            "is_mobilenet_loaded": self.mobilenet is not None,
            "is_classifier_created": self.classifier_model is not None,
            "is_trained": self.is_trained,
            "class_names": self.class_names,
            "training_samples": len(self.training_data_inputs),
            "examples_count": self.examples_count,
            "model_save_path": str(self.model_save_path),
            "feature_dimension": self.feature_dim,
            "model_architecture": {
                "input_shape": (self.feature_dim,),
                "layers": []
            }
        }
        
        # 添加模型架構信息
        if self.classifier_model is not None:
            for layer in self.classifier_model.layers:
                layer_info = {
                    "type": layer.__class__.__name__,
                    "output_shape": layer.output_shape,
                    "trainable": layer.trainable
                }
                if hasattr(layer, 'units'):
                    layer_info["units"] = layer.units
                if hasattr(layer, 'activation'):
                    layer_info["activation"] = str(layer.activation)
                if hasattr(layer, 'rate'):
                    layer_info["dropout_rate"] = layer.rate
                info["model_architecture"]["layers"].append(layer_info)
        
        # 添加標準化參數信息
        if hasattr(self, 'normalization_params') and self.normalization_params:
            info["normalization_params"] = {
                "mean_shape": len(self.normalization_params['mean']),
                "std_shape": len(self.normalization_params['std']),
                "std_min": min(self.normalization_params['std']),
                "std_max": max(self.normalization_params['std'])
            }
        
        # 添加改進特性信息
        info["improvements"] = {
            "data_augmentation": True,
            "early_stopping": True,
            "learning_rate_scheduling": True,
            "batch_normalization": True,
            "feature_dimension_adaptation": True,
            "data_normalization": hasattr(self, 'normalization_params') and self.normalization_params is not None
        }
        
        return info
    
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
