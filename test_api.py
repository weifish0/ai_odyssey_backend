#!/usr/bin/env python3
"""
AI Odyssey Backend API 測試腳本
"""

import requests
import json
import time

# API 基礎 URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """測試健康檢查"""
    print("🔍 測試健康檢查...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def test_register():
    """測試使用者註冊"""
    print("📝 測試使用者註冊...")
    data = {
        "username": "testuser",
        "password": "testpass123"
    }
    response = requests.post(f"{BASE_URL}/auth/register", json=data)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()
    return response.status_code == 201

def test_login():
    """測試使用者登入"""
    print("🔑 測試使用者登入...")
    data = {
        "username": "testuser",
        "password": "testpass123"
    }
    response = requests.post(f"{BASE_URL}/auth/login", json=data)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()
    
    if response.status_code == 200:
        return response.json()["data"]["access_token"]
    return None

def test_get_user_info(token):
    """測試獲取使用者資訊"""
    print("👤 測試獲取使用者資訊...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/users/me", headers=headers)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def test_ask_pet(token):
    """測試詢問 AI 寵物"""
    print("🐟 測試詢問 AI 寵物...")
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "question": "銀龍魚和吳郭魚有什麼外觀特徵？"
    }
    response = requests.post(f"{BASE_URL}/module2/ask-pet", json=data, headers=headers)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def test_training_images(token):
    """測試獲取訓練圖片"""
    print("🖼️ 測試獲取訓練圖片...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{BASE_URL}/module2/training-images", headers=headers)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def test_submit_labels(token):
    """測試提交標註結果"""
    print("🏷️ 測試提交標註結果...")
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "labels": [
            {"image_id": "img_fish_001", "classification": "arowana"},
            {"image_id": "img_fish_002", "classification": "tilapia"}
        ]
    }
    response = requests.post(f"{BASE_URL}/module2/submit-labels", json=data, headers=headers)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def test_identify_fish(token):
    """測試 AI 辨識"""
    print("🎣 測試 AI 辨識...")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{BASE_URL}/module2/identify-fish", headers=headers)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def test_generate_recipe_text(token):
    """測試生成食譜描述"""
    print("👨‍🍳 測試生成食譜描述...")
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "prompt": "為一位熱愛星空的國王，設計一道看起來像迷你銀河的甜點"
    }
    response = requests.post(f"{BASE_URL}/module3/generate-recipe-text", json=data, headers=headers)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def test_generate_recipe_image(token):
    """測試生成食譜圖片"""
    print("🎨 測試生成食譜圖片...")
    headers = {"Authorization": f"Bearer {token}"}
    data = {
        "prompt": "一道名為「星夜琉璃脆」的甜點，深黑色的圓潤糕點"
    }
    response = requests.post(f"{BASE_URL}/module3/generate-recipe-image", json=data, headers=headers)
    print(f"狀態碼: {response.status_code}")
    print(f"回應: {response.json()}")
    print()

def main():
    """主測試函數"""
    print("🚀 開始測試 AI Odyssey Backend API")
    print("=" * 50)
    
    # 測試健康檢查
    test_health_check()
    
    # 測試註冊
    if test_register():
        # 測試登入
        token = test_login()
        if token:
            # 測試需要認證的端點
            test_get_user_info(token)
            test_ask_pet(token)
            test_training_images(token)
            test_submit_labels(token)
            test_identify_fish(token)
            test_generate_recipe_text(token)
            test_generate_recipe_image(token)
        else:
            print("❌ 登入失敗，無法測試需要認證的端點")
    else:
        print("❌ 註冊失敗，無法繼續測試")
    
    print("✅ 測試完成！")

if __name__ == "__main__":
    main() 