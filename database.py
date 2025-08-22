import sqlite3
import bcrypt
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone, timedelta
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "ai_odyssey.db"):
        """初始化資料庫管理器"""
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """取得資料庫連線"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 讓查詢結果可以像字典一樣存取
            return conn
        except Exception as e:
            logger.error(f"資料庫連線失敗: {e}")
            raise
    
    def init_database(self):
        """初始化資料庫表格"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 創建使用者表格
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    money INTEGER DEFAULT 500,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
            
            # 創建使用者會話表格（用於追蹤登入狀態）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_valid BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 創建使用者活動記錄表格
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    activity_type TEXT NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # 創建 token 黑名單表格
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_blacklist (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT UNIQUE NOT NULL,
                    blacklisted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reason TEXT
                )
            ''')
            
            # 為 token 欄位創建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_token_blacklist_token 
                ON token_blacklist(token)
            ''')
            
            conn.commit()
            logger.info("資料庫表格初始化完成")
            
        except Exception as e:
            logger.error(f"資料庫初始化失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_user(self, username: str, password: str, initial_money: int = 500) -> Dict[str, Any]:
        """創建新使用者"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 檢查使用者名稱是否已存在
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                raise ValueError("使用者名稱已被註冊")
            
            # 加密密碼
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # 插入新使用者
            cursor.execute('''
                INSERT INTO users (username, hashed_password, money, created_at)
                VALUES (?, ?, ?, ?)
            ''', (username, hashed_password.decode('utf-8'), initial_money, datetime.now(timezone.utc)))
            
            user_id = cursor.lastrowid
            
            # 記錄活動
            cursor.execute('''
                INSERT INTO user_activities (user_id, activity_type, description)
                VALUES (?, ?, ?)
            ''', (user_id, "REGISTER", f"使用者 {username} 註冊成功"))
            
            conn.commit()
            
            logger.info(f"使用者 {username} 註冊成功，ID: {user_id}")
            
            return {
                "id": user_id,
                "username": username,
                "money": initial_money,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"創建使用者失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def verify_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """驗證使用者登入"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 查詢使用者
            cursor.execute('''
                SELECT id, username, hashed_password, money, is_active
                FROM users WHERE username = ?
            ''', (username,))
            
            user = cursor.fetchone()
            if not user:
                return None
            
            # 檢查帳號是否被停用
            if not user['is_active']:
                raise ValueError("帳號已被停用")
            
            # 驗證密碼
            if bcrypt.checkpw(password.encode('utf-8'), user['hashed_password'].encode('utf-8')):
                # 更新最後登入時間
                cursor.execute('''
                    UPDATE users SET last_login = ? WHERE id = ?
                ''', (datetime.now(timezone.utc), user['id']))
                
                # 記錄登入活動
                cursor.execute('''
                    INSERT INTO user_activities (user_id, activity_type, description)
                    VALUES (?, ?, ?)
                ''', (user['id'], "LOGIN", f"使用者 {username} 登入成功"))
                
                conn.commit()
                
                return {
                    "id": user['id'],
                    "username": user['username'],
                    "money": user['money']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"驗證使用者失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """根據 ID 取得使用者資訊"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, money, created_at, last_login, is_active
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            if user:
                return {
                    "id": user['id'],
                    "username": user['username'],
                    "money": user['money'],
                    "created_at": user['created_at'],
                    "last_login": user['last_login'],
                    "is_active": bool(user['is_active'])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"取得使用者資訊失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """根據使用者名稱取得使用者資訊"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, username, money, created_at, last_login, is_active
                FROM users WHERE username = ?
            ''', (username,))
            
            user = cursor.fetchone()
            if user:
                return {
                    "id": user['id'],
                    "username": user['username'],
                    "money": user['money'],
                    "created_at": user['created_at'],
                    "last_login": user['last_login'],
                    "is_active": bool(user['is_active'])
                }
            
            return None
            
        except Exception as e:
            logger.error(f"取得使用者資訊失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def update_user_money(self, user_id: int, new_amount: int) -> bool:
        """更新使用者金幣數量"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE users SET money = ? WHERE id = ?
            ''', (new_amount, user_id))
            
            if cursor.rowcount > 0:
                # 記錄活動
                cursor.execute('''
                    INSERT INTO user_activities (user_id, activity_type, description)
                    VALUES (?, ?, ?)
                ''', (user_id, "MONEY_UPDATE", f"金幣更新為 {new_amount}"))
                
                conn.commit()
                logger.info(f"使用者 {user_id} 金幣更新為 {new_amount}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"更新使用者金幣失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_session(self, user_id: int, session_token: str, expires_at: datetime) -> bool:
        """創建使用者會話"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 先將舊的會話設為無效
            cursor.execute('''
                UPDATE user_sessions SET is_valid = 0 WHERE user_id = ?
            ''', (user_id,))
            
            # 創建新會話
            cursor.execute('''
                INSERT INTO user_sessions (user_id, session_token, expires_at)
                VALUES (?, ?, ?)
            ''', (user_id, session_token, expires_at))
            
            conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"創建會話失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """驗證會話是否有效"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT us.user_id, us.expires_at, u.username, u.is_active
                FROM user_sessions us
                JOIN users u ON us.user_id = u.id
                WHERE us.session_token = ? AND us.is_valid = 1
            ''', (session_token,))
            
            session = cursor.fetchone()
            if session:
                # 檢查會話是否過期
                expires_at = datetime.fromisoformat(session['expires_at'])
                if datetime.now(timezone.utc) < expires_at and session['is_active']:
                    return {
                        "user_id": session['user_id'],
                        "username": session['username']
                    }
                else:
                    # 會話過期，設為無效
                    cursor.execute('''
                        UPDATE user_sessions SET is_valid = 0 WHERE session_token = ?
                    ''', (session_token,))
                    conn.commit()
            
            return None
            
        except Exception as e:
            logger.error(f"驗證會話失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def invalidate_session(self, session_token: str) -> bool:
        """使會話無效（登出）"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 將會話標記為無效
            cursor.execute('''
                UPDATE user_sessions SET is_valid = 0 WHERE session_token = ?
            ''', (session_token,))
            
            # 將 token 加入黑名單
            cursor.execute('''
                INSERT OR REPLACE INTO token_blacklist (token, blacklisted_at, reason)
                VALUES (?, ?, ?)
            ''', (session_token, datetime.now(timezone.utc), "使用者登出"))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Token {session_token[:20]}... 已被撤銷")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"使會話無效失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def is_token_blacklisted(self, token: str) -> bool:
        """檢查 token 是否在黑名單中"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT COUNT(*) FROM token_blacklist WHERE token = ?
            ''', (token,))
            
            count = cursor.fetchone()[0]
            return count > 0
            
        except Exception as e:
            logger.error(f"檢查 token 黑名單失敗: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def is_session_valid(self, token: str) -> bool:
        """檢查會話是否仍然有效"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT us.expires_at, us.is_valid, u.is_active
                FROM user_sessions us
                JOIN users u ON us.user_id = u.id
                WHERE us.session_token = ? AND us.is_valid = 1
            ''', (token,))
            
            session = cursor.fetchone()
            if session:
                # 檢查會話是否過期
                expires_at = datetime.fromisoformat(session['expires_at'])
                if datetime.now(timezone.utc) < expires_at and session['is_active']:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"檢查會話有效性失敗: {e}")
            return False
        finally:
            if conn:
                conn.close()
    
    def cleanup_expired_tokens(self):
        """清理過期的 token 和會話"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 清理過期的會話
            cursor.execute('''
                UPDATE user_sessions 
                SET is_valid = 0 
                WHERE expires_at < ? AND is_valid = 1
            ''', (datetime.now(timezone.utc),))
            
            expired_sessions = cursor.rowcount
            
            # 清理過期的黑名單 token（保留最近 7 天的記錄）
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)
            cursor.execute('''
                DELETE FROM token_blacklist 
                WHERE blacklisted_at < ?
            ''', (week_ago,))
            
            expired_blacklist = cursor.rowcount
            
            conn.commit()
            
            if expired_sessions > 0 or expired_blacklist > 0:
                logger.info(f"清理完成：{expired_sessions} 個過期會話，{expired_blacklist} 個過期黑名單記錄")
            
        except Exception as e:
            logger.error(f"清理過期 token 失敗: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_user_statistics(self, user_id: int) -> Dict[str, Any]:
        """取得使用者統計資訊"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 取得基本資訊
            cursor.execute('''
                SELECT username, money, created_at, last_login
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            if not user:
                return {}
            
            # 取得活動統計
            cursor.execute('''
                SELECT activity_type, COUNT(*) as count
                FROM user_activities 
                WHERE user_id = ?
                GROUP BY activity_type
            ''', (user_id,))
            
            activities = dict(cursor.fetchall())
            
            # 取得會話統計
            cursor.execute('''
                SELECT COUNT(*) as total_sessions,
                       COUNT(CASE WHEN is_valid = 1 THEN 1 END) as active_sessions
                FROM user_sessions 
                WHERE user_id = ?
            ''', (user_id,))
            
            sessions = cursor.fetchone()
            
            return {
                "username": user['username'],
                "money": user['money'],
                "created_at": user['created_at'],
                "last_login": user['last_login'],
                "activities": activities,
                "sessions": {
                    "total": sessions['total_sessions'],
                    "active": sessions['active_sessions']
                }
            }
            
        except Exception as e:
            logger.error(f"取得使用者統計資訊失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def migrate_existing_users(self, existing_users: Dict[str, Any]):
        """遷移現有的模擬資料庫使用者到 SQLite"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            migrated_count = 0
            
            for username, user_data in existing_users.items():
                # 檢查使用者是否已存在
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                if cursor.fetchone():
                    logger.info(f"使用者 {username} 已存在，跳過遷移")
                    continue
                
                # 創建新使用者
                cursor.execute('''
                    INSERT INTO users (username, hashed_password, money, created_at)
                    VALUES (?, ?, ?, ?)
                ''', (username, user_data['hashed_password'], user_data['money'], datetime.now(timezone.utc)))
                
                migrated_count += 1
                logger.info(f"成功遷移使用者 {username}")
            
            conn.commit()
            logger.info(f"成功遷移 {migrated_count} 個使用者")
            
        except Exception as e:
            logger.error(f"遷移使用者失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()

# 全域資料庫管理器實例
db_manager = DatabaseManager()
