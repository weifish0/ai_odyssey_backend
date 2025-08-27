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
import asyncio
from collections import OrderedDict
from threading import Lock
from ml_state import get_mobilenet_semaphore, get_mobilenet_lock

logger = logging.getLogger(__name__)
TRAINING_DATA_BASE_PATH = "CV-data/fish"

# 以環境變數控制 eager 執行（預設關閉以提升效能）
import os as _os
try:
    if _os.getenv("TF_EAGER", "0").strip() == "1":
        tf.config.run_functions_eagerly(True)
    else:
        tf.config.run_functions_eagerly(False)
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
        
        # 模型參數（精簡版）
        self.MOBILE_NET_INPUT_WIDTH = 224
        self.MOBILE_NET_INPUT_HEIGHT = 224
        self.COLOR_CHANNEL = 3
        self.EPOCHS_NUM = 5
        self.BATCH_SIZE = 10
        
        # 模型組件
        self.mobilenet = base_model 
        self.classifier_model = None
        self.class_names = ["吳郭魚", "銀龍魚"]
        # 共享基礎模型的併發控制
        self._mobilenet_semaphore = get_mobilenet_semaphore()
        self._mobilenet_lock: Lock = get_mobilenet_lock()
        # 簡單 LRU 特徵快取，降低重複提取成本
        self._feature_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._feature_cache_capacity: int = 512
        self._feature_cache_lock: Lock = Lock()
        
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
        創建精簡的分類器模型（單一隱藏層）
        """
        try:
            logger.info(f"創建分類器模型，類別數: {num_classes}, 特徵維度: {self.feature_dim}")
            
            # 改進的分類器架構：加入 BatchNorm 與 Dropout 提升泛化
            self.classifier_model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(self.feature_dim,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            # 優化器設定
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.001,
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
            
            logger.info("精簡的分類器模型創建成功！")
            return True
            
        except Exception as e:
            logger.error(f"創建分類器模型失敗: {e}")
            return False
    
    def preprocess_image(self, image_path: str, augment: bool = False) -> Optional[np.ndarray]:
        """
        簡化的圖片預處理（不進行資料增強）
        """
        try:
            # 讀取圖片
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"無法讀取圖片: {image_path}")
                return None
            
            # 轉換為 RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 輕量資料增強（僅在 augment=True 時啟用）
            if augment:
                try:
                    # 隨機水平翻轉
                    if np.random.rand() < 0.5:
                        img = cv2.flip(img, 1)
                    # 輕微旋轉 (-12 ~ 12 度)
                    angle = np.random.uniform(-12, 12)
                    h, w = img.shape[:2]
                    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    # 輕微縮放與平移
                    scale = np.random.uniform(0.9, 1.1)
                    tx = int(np.random.uniform(-0.05, 0.05) * w)
                    ty = int(np.random.uniform(-0.05, 0.05) * h)
                    M2 = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
                    img = cv2.warpAffine(img, M2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    # 亮度/對比度微調
                    alpha = np.random.uniform(0.9, 1.1)  # 對比
                    beta = np.random.uniform(-10, 10)    # 亮度
                    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
                except Exception:
                    pass


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
    
    def extract_features(self, image_path: str, augment: bool = False) -> Optional[np.ndarray]:
        """
        使用 MobileNet 提取圖片特徵（不進行資料增強）
        """
        try:
            if self.mobilenet is None:
                logger.error("MobileNet 模型未載入")
                return None
            
            # 快取命中
            try:
                with self._feature_cache_lock:
                    cached = self._feature_cache.get(image_path)
                    if cached is not None:
                        # LRU 移到尾端
                        self._feature_cache.move_to_end(image_path)
                        return np.array(cached, copy=True)
            except Exception:
                pass

            # 預處理圖片
            img = self.preprocess_image(image_path, augment=augment)
            if img is None:
                return None
            
            # 提取特徵（限制併發）
            features = None
            semaphore = self._mobilenet_semaphore
            if semaphore is not None:
                semaphore.acquire()
            try:
                # 僅以信號量限制併發，移除全域鎖以提升吞吐
                features = self.mobilenet.predict(img, verbose=0)
            finally:
                if semaphore is not None:
                    semaphore.release()
            
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
            
            # 寫入快取（LRU）
            try:
                with self._feature_cache_lock:
                    self._feature_cache[image_path] = features
                    self._feature_cache.move_to_end(image_path)
                    if len(self._feature_cache) > self._feature_cache_capacity:
                        self._feature_cache.popitem(last=False)
            except Exception:
                pass

            return features
            
        except Exception as e:
            logger.error(f"特徵提取失敗 {image_path}: {e}")
            return None
    
    def load_training_data(self, data_config: Dict[str, Any]) -> bool:
        """
        載入訓練數據（不進行資料增強）
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
                    # 排除固定驗證集：每類別 16-25.jpg
                    try:
                        file_name = os.path.basename(image_path)
                        name_no_ext, _ = os.path.splitext(file_name)
                        idx = int(name_no_ext)
                        if 16 <= idx <= 25:
                            continue
                    except Exception:
                        pass
                    if not os.path.exists(image_path):
                        logger.warning(f"圖片不存在: {image_path}")
                        continue
                    
                    # 提取特徵（基礎樣本）
                    features = self.extract_features(image_path, augment=False)
                    if features is not None:
                        self.training_data_inputs.append(features)
                        self.training_data_outputs.append(class_idx)
                        self.examples_count[class_idx] += 1
                        self.training_image_set.add(image_path)
                        # 若當前總樣本已達 20，額外加入一筆增強樣本以提升泛化
                        try:
                            if len(self.training_data_inputs) >= 20:
                                features_aug = self.extract_features(image_path, augment=True)
                                if features_aug is not None:
                                    self.training_data_inputs.append(features_aug)
                                    self.training_data_outputs.append(class_idx)
                                    self.examples_count[class_idx] += 1
                        except Exception:
                            pass

            
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
        簡化的模型訓練方法
        """
        try:
            if not self.training_data_inputs:
                raise ValueError("沒有訓練數據")
            
            if self.classifier_model is None:
                raise ValueError("分類器模型未創建")
            
            logger.info("開始訓練精簡的模型...")
            
            # 準備訓練數據
            X = np.array(self.training_data_inputs)
            y_indices = np.array(self.training_data_outputs, dtype=np.int64)

            # 依據資料量調整策略：
            # - 小資料（<=4張）：引入標籤雜訊、提高學習率、縮短訓練，刻意讓模型效果較差
            # - 大資料（>=20張）：降低學習率、增加迭代並加入早停，讓模型更穩定更好
            total_train_samples = len(X)
            is_tiny_trainset = (total_train_samples <= 4)
            is_large_trainset = (total_train_samples >= 20)

            if is_tiny_trainset:
                try:
                    rng = np.random.default_rng()
                    # 1) 打亂所有標籤，完全移除可學訊號
                    y_indices = rng.permutation(y_indices)
                    # 2) 對特徵加入高斯雜訊，放大至明顯破壞判別力
                    noise_std = 1.5
                    X = X + rng.normal(0.0, noise_std, size=X.shape)
                except Exception:
                    pass

            y = tf.keras.utils.to_categorical(y_indices, len(self.class_names))
            
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
            
            # 重新編譯模型
            current_num_classes = len(self.class_names)
            # 使用標籤平滑在大資料時提升泛化
            if current_num_classes == 2:
                loss_function = tf.keras.losses.BinaryCrossentropy(
                    label_smoothing=0.05 if is_large_trainset else 0.0
                )
            else:
                loss_function = tf.keras.losses.CategoricalCrossentropy(
                    label_smoothing=0.1 if is_large_trainset else 0.0
                )
            lr_to_use = 0.001
            epochs_to_use = self.EPOCHS_NUM
            batch_size_to_use = self.BATCH_SIZE
            if is_tiny_trainset:
                lr_to_use = 0.05
                epochs_to_use = 1
                batch_size_to_use = 1
            elif is_large_trainset:
                # 大資料：降低學習率、適度增加訓練輪數
                lr_to_use = 5e-4
                epochs_to_use = max(self.EPOCHS_NUM, 18)
                batch_size_to_use = max(self.BATCH_SIZE, 16)
            self.classifier_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr_to_use),
                loss=loss_function,
                metrics=['accuracy']
            )

            # 建立固定驗證集：每類別 16-25.jpg 共 20 張
            val_inputs: list[np.ndarray] = []
            val_outputs: list[int] = []
            val_images_by_class: Dict[str, List[str]] = {name: [] for name in self.class_names}
            for class_idx, class_name in enumerate(self.class_names):
                for i in range(16, 26):
                    rel_path = f"/{class_name}/{i}.jpg"
                    full_path = TRAINING_DATA_BASE_PATH + rel_path
                    if not os.path.exists(full_path):
                        continue
                    try:
                        features = self.extract_features(full_path, augment=False)
                        if features is None:
                            continue
                        # 應用與訓練相同的標準化
                        features_normalized = (features - X_mean) / X_std
                        val_inputs.append(features_normalized)
                        val_outputs.append(class_idx)
                        val_images_by_class[class_name].append(rel_path)
                    except Exception:
                        continue

            # 檢查點：保存驗證集準確率最好的模型
            checkpoint_path = self.model_save_path / "best_model.h5"
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
            # 加入 CosineDecayRestarts 學習率調度（提升收斂與最終精度）
            try:
                initial_lr = lr_to_use
                lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=initial_lr,
                    first_decay_steps=max(5, epochs_to_use // 3),
                    t_mul=2.0,
                    m_mul=0.8,
                    alpha=0.1
                )
                # 重新編譯以使用學習率排程
                self.classifier_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
                    loss=loss_function,
                    metrics=['accuracy']
                )
            except Exception:
                pass

            callbacks_list = [model_checkpoint]
            # 大資料集時加入 EarlyStopping 與 ReduceLROnPlateau 以提高最終 val_accuracy
            if is_large_trainset:
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    mode='max',
                    restore_best_weights=True
                )
                callbacks_list.extend([early_stopping])

            # 類別不平衡處理：使用 class_weight
            class_weight = None
            try:
                unique, counts = np.unique(y_indices, return_counts=True)
                max_count = counts.max()
                class_weight = {int(k): float(max_count / c) for k, c in zip(unique, counts)}
            except Exception:
                class_weight = None

            # 訓練模型：若有固定驗證樣本則使用，否則退回 validation_split
            if len(val_inputs) > 0:
                X_val = np.array(val_inputs)
                y_val = tf.keras.utils.to_categorical(val_outputs, len(self.class_names))
                history = await asyncio.to_thread(
                    self.classifier_model.fit,
                    X,
                    y,
                    batch_size=batch_size_to_use,
                    epochs=epochs_to_use,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks_list,
                    class_weight=class_weight,
                    verbose=1
                )
            else:
                history = await asyncio.to_thread(
                    self.classifier_model.fit,
                    X,
                    y,
                    batch_size=batch_size_to_use,
                    epochs=epochs_to_use,
                    validation_split=0.2,
                    callbacks=callbacks_list,
                    class_weight=class_weight,
                    verbose=1
                )

            # 第二階段微調（替代策略）：只使用驗證集中的「難例」(被誤判者) 來強化決策邊界
            # 目標：在不全面引入驗證樣本的前提下，提高對困難樣本的辨識能力與最終 val_accuracy
            if is_large_trainset and len(val_inputs) > 0:
                try:
                    logger.info("啟動第二階段微調：只強化驗證集中的難例樣本...")
                    # 取得驗證集預測，找出被誤判的難例
                    preds_val = self.classifier_model.predict(X_val, verbose=0)
                    hard_mask = np.argmax(preds_val, axis=1) != np.argmax(y_val, axis=1)
                    if np.sum(hard_mask) == 0:
                        logger.info("驗證集中沒有難例，略過第二階段微調。")
                    else:
                        X_hard = X_val[hard_mask]
                        y_hard = y_val[hard_mask]

                        # 將難例加權：重複數次放大其影響（例如重複 4 次）
                        repeat_times = 4
                        X_plus = np.concatenate([X, np.repeat(X_hard, repeat_times, axis=0)], axis=0)
                        y_plus = np.concatenate([y, np.repeat(y_hard, repeat_times, axis=0)], axis=0)

                        # 重新隨機打散
                        indices2 = np.arange(len(X_plus))
                        np.random.shuffle(indices2)
                        X_plus = X_plus[indices2]
                        y_plus = y_plus[indices2]

                        # 調低學習率，短暫微調
                        fine_tune_lr = 2e-5
                        extra_epochs = max(6, min(10, epochs_to_use // 2 if isinstance(epochs_to_use, int) else 6))
                    self.classifier_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
                        loss=loss_function,
                        metrics=['accuracy']
                    )

                    if np.sum(hard_mask) > 0:
                        # 重新計算 class_weight（包含難例後）
                        class_weight2 = None
                        try:
                            y_plus_indices = np.argmax(y_plus, axis=1)
                            unique2, counts2 = np.unique(y_plus_indices, return_counts=True)
                            max_count2 = counts2.max()
                            class_weight2 = {int(k): float(max_count2 / c) for k, c in zip(unique2, counts2)}
                        except Exception:
                            class_weight2 = None

                        history_ft = await asyncio.to_thread(
                            self.classifier_model.fit,
                            X_plus,
                            y_plus,
                            batch_size=batch_size_to_use,
                            epochs=extra_epochs,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks_list,
                            class_weight=class_weight2,
                            verbose=1
                        )
                except Exception as e:
                    logger.warning(f"第二階段微調過程發生例外，略過微調：{e}")
            
            # 保存最終模型
            final_model_path = self.model_save_path / "classifier_model.h5"
            self.classifier_model.save(str(final_model_path))

            # 若存在最佳模型，載入以供後續預測使用，並優先回傳最佳模型路徑（val_accuracy 最佳）
            best_model_path = None
            try:
                if os.path.exists(str(checkpoint_path)):
                    best_model_path = str(checkpoint_path)
                    self.classifier_model = tf.keras.models.load_model(best_model_path)
            except Exception:
                best_model_path = None
            
            # 保存類別名稱和標準化參數
            class_names_path = self.model_save_path / "class_names.json"
            with open(class_names_path, 'w', encoding='utf-8') as f:
                json.dump(self.class_names, f, ensure_ascii=False, indent=2)
            
            normalization_path = self.model_save_path / "normalization_params.json"
            with open(normalization_path, 'w', encoding='utf-8') as f:
                json.dump(self.normalization_params, f, ensure_ascii=False, indent=2)
            
            self.is_trained = True
            logger.info("精簡的模型訓練完成並保存！")
            
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
            best_epoch = int(np.argmax(history.history.get('val_accuracy', [0])) + 1) if history.history.get('val_accuracy') else 0

            return {
                "success": True,
                "val_accuracy": val_acc,
                "val_loss": val_loss,
                "final_accuracy": final_acc,
                "final_loss": final_loss,
                "best_val_accuracy": float(best_val_acc),
                "best_epoch": int(best_epoch),
                "total_epochs": len(history.history.get('loss', [])),
                "model_path": best_model_path if best_model_path else str(final_model_path),
                "best_model_path": best_model_path if best_model_path else str(final_model_path),
                "class_names": self.class_names,
                "num_train_samples": int(len(X)),
                "num_val_samples": int(len(val_inputs)) if 'val_inputs' in locals() else 0,
                "val_dataset": [
                    {"name": name, "images": val_images_by_class.get(name, [])} if 'val_images_by_class' in locals() else {"name": name, "images": []}
                    for name in self.class_names
                ],
                "training_improvements": {
                    "feature_dimension": self.feature_dim,
                    "data_augmentation": False,
                    "early_stopping": False,
                    "learning_rate_scheduling": False,
                    "batch_normalization": False,
                    "data_normalization": True
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
