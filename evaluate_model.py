#!/usr/bin/env python3
"""
評估影像分類模型訓練成果的腳本

流程：
1) 初始化全域 MobileNet 基礎模型
2) 分別用兩組訓練資料進行訓練
3) 使用吳郭魚/銀龍魚 16-25.jpg 做測試，計算準確率

注意：
- 資料路徑基於 image_classification.py 中的 TRAINING_DATA_BASE_PATH
- 測試圖片以 "/吳郭魚/16.jpg" 這種相對路徑傳入（程式會自動拼接）
"""

import json
import os
import sys
from typing import Dict, Any, List, Tuple

from ml_state import init_mobilenet, get_mobilenet
from image_classification import ImageClassificationModel, TRAINING_DATA_BASE_PATH


TRAIN_SET_1: Dict[str, Any] = {
    "train_dataset": [
        {
            "name": "吳郭魚",
            "images": ["/吳郭魚/1.jpg", "/吳郭魚/2.jpg"],
        },
        {
            "name": "銀龍魚",
            "images": [
                "/銀龍魚/1.jpg",
                "/銀龍魚/2.jpg",
                "/銀龍魚/3.jpg",
                "/銀龍魚/4.jpg",
                "/銀龍魚/5.jpg",
                "/銀龍魚/6.jpg",
                "/銀龍魚/7.jpg",
                "/銀龍魚/8.jpg",
                "/銀龍魚/9.jpg",
                "/銀龍魚/10.jpg",
                "/銀龍魚/11.jpg",
                "/銀龍魚/12.jpg",
                "/銀龍魚/13.jpg",
                "/銀龍魚/14.jpg",
                "/吳郭魚/15.jpg",
            ],
        },
    ]
}


TRAIN_SET_2: Dict[str, Any] = {
    "train_dataset": [
        {
            "name": "吳郭魚",
            "images": [
                "/吳郭魚/1.jpg",
                "/吳郭魚/2.jpg",
                "/吳郭魚/3.jpg",
                "/吳郭魚/4.jpg",
                "/吳郭魚/5.jpg",
                "/吳郭魚/6.jpg",
                "/吳郭魚/7.jpg",
                "/吳郭魚/8.jpg",
                "/吳郭魚/9.jpg",
                "/吳郭魚/10.jpg",
                "/吳郭魚/11.jpg",
                "/吳郭魚/12.jpg",
                "/吳郭魚/13.jpg",
                "/吳郭魚/14.jpg",
                "/吳郭魚/15.jpg",
            ],
        },
        {
            "name": "銀龍魚",
            "images": [
                "/銀龍魚/1.jpg",
                "/銀龍魚/2.jpg",
                "/銀龍魚/3.jpg",
                "/銀龍魚/4.jpg",
                "/銀龍魚/5.jpg",
                "/銀龍魚/6.jpg",
                "/銀龍魚/7.jpg",
                "/銀龍魚/8.jpg",
                "/銀龍魚/9.jpg",
                "/銀龍魚/10.jpg",
                "/銀龍魚/11.jpg",
                "/銀龍魚/12.jpg",
                "/銀龍魚/13.jpg",
                "/銀龍魚/14.jpg",
                "/吳郭魚/15.jpg",
            ],
        },
    ]
}


def build_test_set() -> List[Tuple[str, str]]:
    """建立測試資料: (相對路徑, 期望類別) 列表"""
    test_pairs: List[Tuple[str, str]] = []
    for i in range(16, 26):
        test_pairs.append((f"/吳郭魚/{i}.jpg", "吳郭魚"))
        test_pairs.append((f"/銀龍魚/{i}.jpg", "銀龍魚"))
    return test_pairs


def evaluate_model(model: ImageClassificationModel, test_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """逐張測試並統計準確率"""
    total = 0
    correct = 0
    details: List[Dict[str, Any]] = []

    for rel_path, expected in test_pairs:
        total += 1
        result = model.predict_image(rel_path)
        ok = bool(result.get("success")) and (result.get("predicted_class") == expected)
        if ok:
            correct += 1
        details.append({
            "image": rel_path,
            "expected": expected,
            "predicted": result.get("predicted_class"),
            "confidence": result.get("confidence"),
            "success": bool(result.get("success")),
            "error": result.get("error"),
        })

    accuracy = correct / total if total > 0 else 0.0
    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "details": details,
    }


def run_experiment(exp_name: str, train_config: Dict[str, Any]) -> Dict[str, Any]:
    print(f"=== 開始實驗: {exp_name} ===")
    # 取得已初始化之全域 MobileNet
    base_model = get_mobilenet()
    if base_model is None:
        raise RuntimeError("MobileNet 尚未初始化")

    # 每次實驗建立新的使用者模型（避免互相影響）
    model = ImageClassificationModel(user_name=f"eval_{exp_name}", base_model=base_model)
    ok = model.load_training_data(train_config)
    if not ok:
        raise RuntimeError("載入訓練數據失敗")

    train_result = model.train_model()
    if not train_result.get("success"):
        raise RuntimeError(f"訓練失敗: {train_result.get('error')}")

    # 測試集評估
    test_pairs = build_test_set()
    eval_result = evaluate_model(model, test_pairs)

    summary = {
        "experiment": exp_name,
        "train_result": {
            "final_accuracy": train_result.get("final_accuracy"),
            "val_accuracy": train_result.get("val_accuracy"),
            "best_val_accuracy": train_result.get("best_val_accuracy"),
            "best_epoch": train_result.get("best_epoch"),
        },
        "eval_result": {
            "total": eval_result["total"],
            "correct": eval_result["correct"],
            "accuracy": eval_result["accuracy"],
        },
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return {"summary": summary, "details": eval_result["details"]}


def main():
    # 初始化 MobileNet
    print("初始化 MobileNet...")
    if not init_mobilenet():
        print("❌ MobileNet 初始化失敗")
        sys.exit(1)
    else:
        print("✅ MobileNet 初始化成功")

    results = {}
    # 實驗 1
    results["set1"] = run_experiment("set1", TRAIN_SET_1)
    # 實驗 2
    results["set2"] = run_experiment("set2", TRAIN_SET_2)

    # 儲存結果
    out_path = os.path.join("models", "eval_results.json")
    os.makedirs("models", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"評估結果已輸出: {out_path}")


if __name__ == "__main__":
    main()


