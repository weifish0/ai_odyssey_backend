#!/usr/bin/env python3
"""
資料庫遷移腳本
將現有的模擬資料庫使用者遷移到 SQLite
"""

import bcrypt
import logging
from database import db_manager

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_existing_users():
    """遷移現有的模擬資料庫使用者"""
    
    # 現有的模擬資料庫使用者（從 main.py 複製過來）
    existing_users = {
        "will": {
            "id": "123456789",
            "username": "will",
            "hashed_password": "mypassword",  # 這是明文密碼，需要加密
            "money": 300000
        },
        "9n": {
            "id": "999999999",
            "username": "9n",
            "hashed_password": "9nhaha1234",  # 這是明文密碼，需要加密
            "money": 300000
        }
    }
    
    try:
        logger.info("開始遷移使用者資料...")
        
        # 為每個使用者加密密碼並遷移
        for username, user_data in existing_users.items():
            try:
                # 加密密碼
                hashed_password = bcrypt.hashpw(
                    user_data['hashed_password'].encode('utf-8'), 
                    bcrypt.gensalt()
                )
                
                # 更新使用者資料
                user_data['hashed_password'] = hashed_password.decode('utf-8')
                
                logger.info(f"準備遷移使用者: {username}")
                
            except Exception as e:
                logger.error(f"加密使用者 {username} 密碼失敗: {e}")
                continue
        
        # 執行遷移
        db_manager.migrate_existing_users(existing_users)
        
        logger.info("✅ 使用者遷移完成！")
        
        # 驗證遷移結果
        logger.info("驗證遷移結果...")
        for username in existing_users.keys():
            user = db_manager.get_user_by_username(username)
            if user:
                logger.info(f"✅ 使用者 {username} 遷移成功，ID: {user['id']}")
            else:
                logger.error(f"❌ 使用者 {username} 遷移失敗")
        
    except Exception as e:
        logger.error(f"遷移過程發生錯誤: {e}")
        raise

def create_test_user():
    """創建測試使用者"""
    try:
        logger.info("創建測試使用者...")
        
        # 創建一個新的測試使用者
        test_user = db_manager.create_user(
            username="test_user",
            password="test123",
            initial_money=1000
        )
        
        logger.info(f"✅ 測試使用者創建成功: {test_user}")
        
        # 驗證登入
        verified_user = db_manager.verify_user("test_user", "test123")
        if verified_user:
            logger.info(f"✅ 測試使用者登入驗證成功: {verified_user}")
        else:
            logger.error("❌ 測試使用者登入驗證失敗")
            
    except Exception as e:
        logger.error(f"創建測試使用者失敗: {e}")

def show_database_status():
    """顯示資料庫狀態"""
    try:
        logger.info("=== 資料庫狀態 ===")
        
        # 檢查資料庫連線
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # 檢查表格
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"資料庫表格: {[table[0] for table in tables]}")
        
        # 檢查使用者數量
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        logger.info(f"使用者總數: {user_count}")
        
        # 檢查會話數量
        cursor.execute("SELECT COUNT(*) FROM user_sessions")
        session_count = cursor.fetchone()[0]
        logger.info(f"會話總數: {session_count}")
        
        # 檢查活動記錄數量
        cursor.execute("SELECT COUNT(*) FROM user_activities")
        activity_count = cursor.fetchone()[0]
        logger.info(f"活動記錄總數: {activity_count}")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"檢查資料庫狀態失敗: {e}")

if __name__ == "__main__":
    try:
        logger.info("🚀 開始執行資料庫遷移...")
        
        # 顯示初始狀態
        show_database_status()
        
        # 遷移現有使用者
        migrate_existing_users()
        
        # 創建測試使用者
        create_test_user()
        
        # 顯示最終狀態
        show_database_status()
        
        logger.info("🎉 資料庫遷移完成！")
        
    except Exception as e:
        logger.error(f"❌ 遷移失敗: {e}")
        exit(1)
