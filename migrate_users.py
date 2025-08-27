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
            # 你可以用 password_plain 或 hashed_password 放明文（將自動加密）
            "hashed_password": "mypassword",
        },
        "9n": {
            "id": "999999999",
            "username": "9n",
            "hashed_password": "9nhaha1234",
        }
    }
    
    try:
        logger.info("開始遷移使用者資料...")
        
        # 為每個使用者加密密碼並遷移（相容 password_plain 或未加密的 hashed_password）
        for username, user_data in existing_users.items():
            try:
                raw_pw = user_data.get('password_plain') or user_data.get('hashed_password')
                if raw_pw is None:
                    raise ValueError("缺少 password_plain/hashed_password 欄位")

                # 若不是 bcrypt 格式（$2 開頭），視為明文並加密
                if not str(raw_pw).startswith("$2"):
                    hashed_password = bcrypt.hashpw(
                        str(raw_pw).encode('utf-8'),
                        bcrypt.gensalt()
                    ).decode('utf-8')
                else:
                    hashed_password = str(raw_pw)

                # 更新為最終要寫入資料庫的雜湊
                user_data['hashed_password'] = hashed_password
                
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
        
        # 顯示最終狀態
        show_database_status()
        
        logger.info("🎉 資料庫遷移完成！")
        
    except Exception as e:
        logger.error(f"❌ 遷移失敗: {e}")
        exit(1)
