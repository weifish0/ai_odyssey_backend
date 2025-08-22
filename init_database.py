#!/usr/bin/env python3
"""
資料庫初始化腳本
用於首次設置資料庫
"""

import logging
from database import db_manager

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函數"""
    try:
        logger.info("🚀 開始初始化資料庫...")
        
        # 資料庫管理器會在初始化時自動創建表格
        logger.info("✅ 資料庫表格創建完成")
        
        # 顯示資料庫狀態
        conn = db_manager.get_connection()
        cursor = conn.cursor()
        
        # 檢查表格
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"📋 資料庫表格: {[table[0] for table in tables]}")
        
        conn.close()
        
        logger.info("🎉 資料庫初始化完成！")
        logger.info("💡 接下來可以執行 migrate_users.py 來遷移現有使用者")
        
    except Exception as e:
        logger.error(f"❌ 資料庫初始化失敗: {e}")
        exit(1)

if __name__ == "__main__":
    main()
