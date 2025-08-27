import logging
import os
import tensorflow as tf
from threading import Lock, Semaphore

logger = logging.getLogger(__name__)

# 使用字典來管理全域狀態，避免模組間的全域變數引用問題
_global_state = {
    "GLOBAL_MOBILENET": None,
    "USER_MODELS": {},
    "MOBILENET_LOCK": Lock(),
    "MOBILENET_SEMAPHORE": None
}


def init_mobilenet() -> bool:
    """初始化 MobileNet 基礎模型，返回是否成功"""
    try:
        logger.info("正在載入 MobileNet V3 基礎模型...")
        mobilenet = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        mobilenet.trainable = False
        # 預熱模型
        dummy_input = tf.zeros((1, 224, 224, 3))
        _ = mobilenet(dummy_input)
        
        # 更新全域狀態
        _global_state["GLOBAL_MOBILENET"] = mobilenet
        # 初始化信號量：限制同時執行 mobilenet.predict 的併發數
        max_workers = max(2, (os.cpu_count() or 4) // 2)
        _global_state["MOBILENET_SEMAPHORE"] = Semaphore(max_workers)
        
        logger.info("✅ MobileNet V3 基礎模型載入成功並已設定為全域共用！")
        return True
    except Exception as e:
        logger.error(f"❌ 載入 MobileNet 基礎模型失敗: {e}")
        _global_state["GLOBAL_MOBILENET"] = None
        return False


def is_mobilenet_ready() -> bool:
    """檢查 MobileNet 模型是否已準備就緒"""
    return _global_state["GLOBAL_MOBILENET"] is not None


def get_mobilenet():
    """取得 MobileNet 模型實例"""
    return _global_state["GLOBAL_MOBILENET"]


def get_mobilenet_lock() -> Lock:
    """取得用於序列化 MobileNet 操作的鎖"""
    return _global_state["MOBILENET_LOCK"]


def get_mobilenet_semaphore() -> Semaphore:
    """取得限制 MobileNet 併發的信號量"""
    return _global_state["MOBILENET_SEMAPHORE"]


def get_user_models():
    """取得使用者模型字典"""
    return _global_state["USER_MODELS"]


def set_user_model(user_name: str, model):
    """設定使用者模型"""
    _global_state["USER_MODELS"][user_name] = model


def get_user_model(user_name: str):
    """取得使用者模型"""
    return _global_state["USER_MODELS"].get(user_name)


# 使用屬性來提供動態的全域變數引用
class GlobalState:
    @property
    def GLOBAL_MOBILENET(self):
        return _global_state["GLOBAL_MOBILENET"]
    
    @property
    def USER_MODELS(self):
        return _global_state["USER_MODELS"]

    @property
    def MOBILENET_LOCK(self):
        return _global_state["MOBILENET_LOCK"]

    @property
    def MOBILENET_SEMAPHORE(self):
        return _global_state["MOBILENET_SEMAPHORE"]


# 創建全域狀態實例
_global_state_instance = GlobalState()

# 為了向後相容，提供動態的全域變數引用
GLOBAL_MOBILENET = _global_state_instance.GLOBAL_MOBILENET
USER_MODELS = _global_state_instance.USER_MODELS
MOBILENET_LOCK = _global_state_instance.MOBILENET_LOCK
MOBILENET_SEMAPHORE = _global_state_instance.MOBILENET_SEMAPHORE


