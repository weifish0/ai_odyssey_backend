#!/usr/bin/env python3
"""
測試改進後的影像分類模型
"""

import asyncio
import json
import tensorflow as tf
from image_classification import ImageClassificationModel
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_improved_model():
    """測試改進後的模型"""
    
    print("🚀 開始測試改進後的影像分類模型...")
    
    try:
        # 1. 載入 MobileNet 基礎模型
        print("📱 載入 MobileNet V3 基礎模型...")
        mobilenet = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        mobilenet.trainable = False
        print("✅ MobileNet 載入成功")
        
        # 2. 創建改進的影像分類模型
        print("🔧 創建改進的影像分類模型...")
        model = ImageClassificationModel(
            user_name="test_user",
            base_model=mobilenet
        )
        print("✅ 模型創建成功")
        
        # 3. 顯示模型信息
        print("\n📊 模型信息:")
        model_info = model.get_model_info()
        print(json.dumps(model_info, indent=2, ensure_ascii=False))
        
        # 4. 準備測試數據
        print("\n📁 準備測試數據...")
        test_data_config = {
            "train_dataset": [
                {
                    "name": "吳郭魚",
                    "images": ["/吳郭魚/1.jpg", "/吳郭魚/2.jpg", "/吳郭魚/3.jpg", "/吳郭魚/4.jpg", "/吳郭魚/5.jpg"]
                },
                {
                    "name": "銀龍魚", 
                    "images": ["/銀龍魚/1.jpg", "/銀龍魚/2.jpg", "/銀龍魚/3.jpg", "/銀龍魚/4.jpg", "/銀龍魚/5.jpg"]
                }
            ]
        }
        
        # 5. 載入訓練數據
        print("📥 載入訓練數據...")
        success = model.load_training_data(test_data_config)
        if not success:
            print("❌ 載入訓練數據失敗")
            return
        
        print(f"✅ 成功載入 {len(model.training_data_inputs)} 個訓練樣本")
        
        # 6. 訓練模型
        print("\n🎯 開始訓練模型...")
        training_result = await model.train_model()
        
        if training_result.get("success"):
            print("✅ 模型訓練成功！")
            print(f"   最佳驗證準確率: {training_result.get('best_val_accuracy', 'N/A'):.2%}")
            print(f"   最佳輪數: {training_result.get('best_epoch', 'N/A')}")
            print(f"   總訓練輪數: {training_result.get('total_epochs', 'N/A')}")
            print(f"   最終驗證準確率: {training_result.get('val_accuracy', 'N/A'):.2%}")
            print(f"   最終驗證損失: {training_result.get('val_loss', 'N/A'):.4f}")
            
            # 顯示改進特性
            improvements = training_result.get('training_improvements', {})
            print("\n🔧 使用的改進技術:")
            for tech, enabled in improvements.items():
                status = "✅" if enabled else "❌"
                print(f"   {status} {tech}")
        else:
            print(f"❌ 模型訓練失敗: {training_result.get('error')}")
            return
        
        # 7. 測試預測
        print("\n🔮 測試預測功能...")
        test_image = "/吳郭魚/6.jpg"  # 使用未訓練的圖片
        prediction = model.predict_image(test_image)
        
        if prediction.get("success"):
            print("✅ 預測成功！")
            print(f"   預測類別: {prediction.get('predicted_class')}")
            print(f"   置信度: {prediction.get('confidence'):.2%}")
            print(f"   預測確定性: {prediction.get('prediction_certainty', 0):.4f}")
            print(f"   期望類別: {prediction.get('expected_class')}")
            print(f"   是否正確: {prediction.get('is_correct')}")
            
            # 顯示各類別機率
            print("\n📊 各類別機率:")
            for class_name, prob in prediction.get('class_probabilities', {}).items():
                print(f"   {class_name}: {prob:.2%}")
        else:
            print(f"❌ 預測失敗: {prediction.get('error')}")
        
        # 8. 最終模型信息
        print("\n📋 最終模型信息:")
        final_info = model.get_model_info()
        print(f"   特徵維度: {final_info.get('feature_dimension')}")
        print(f"   訓練樣本數: {final_info.get('training_samples')}")
        print(f"   是否已訓練: {final_info.get('is_trained')}")
        
        print("\n🎉 測試完成！")
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 設定 TensorFlow 環境
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 強制使用 CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # 運行測試
    asyncio.run(test_improved_model())
