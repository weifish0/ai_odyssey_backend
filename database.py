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
            try:
                # 啟用 WAL 與合理同步等參數以改善並發寫入
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA busy_timeout=3000;")
            except Exception:
                pass
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
                    -- 分數欄位（v2 之後新增）
                    score_level1 INTEGER DEFAULT 0,
                    score_level2 INTEGER DEFAULT 0,
                    score_level3 INTEGER DEFAULT 0,
                    score_total  INTEGER DEFAULT 0,
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
            
            # 檢查並補齊缺少的欄位（針對舊版資料庫進行輕量遷移）
            cursor.execute("PRAGMA table_info(users)")
            existing_cols = {row[1] for row in cursor.fetchall()}
            required_cols = {
                "score_level1": "ALTER TABLE users ADD COLUMN score_level1 INTEGER DEFAULT 0",
                "score_level2": "ALTER TABLE users ADD COLUMN score_level2 INTEGER DEFAULT 0",
                "score_level3": "ALTER TABLE users ADD COLUMN score_level3 INTEGER DEFAULT 0",
                "score_total":  "ALTER TABLE users ADD COLUMN score_total  INTEGER DEFAULT 0",
            }
            for col, ddl in required_cols.items():
                if col not in existing_cols:
                    try:
                        cursor.execute(ddl)
                        logger.info(f"已新增缺少欄位: {col}")
                    except Exception as e:
                        logger.warning(f"新增欄位 {col} 失敗或已存在: {e}")

            conn.commit()
            logger.info("資料庫表格初始化完成")
            
        except Exception as e:
            logger.error(f"資料庫初始化失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def create_user(self, username: str, password: str) -> Dict[str, Any]:
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
                INSERT INTO users (username, hashed_password, score_level1, score_level2, score_level3, score_total, created_at)
                VALUES (?, ?, 0, 0, 0, 0, ?)
            ''', (username, hashed_password.decode('utf-8'), datetime.now(timezone.utc)))
            
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
                # 提供新舊欄位相容
                "score_level1": 0,
                "score_level2": 0,
                "score_level3": 0,
                "score_total": 0,
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
                SELECT id, username, hashed_password,
                       score_level1, score_level2, score_level3, score_total,
                       is_active
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
                
                # 回傳時同樣提供相容欄位
                return {
                    "id": user['id'],
                    "username": user['username'],
                    "score_level1": user["score_level1"],
                    "score_level2": user["score_level2"],
                    "score_level3": user["score_level3"],
                    "score_total": user["score_total"],
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
                SELECT id, username, score_level1, score_level2, score_level3, score_total,
                       created_at, last_login, is_active
                FROM users WHERE id = ?
            ''', (user_id,))
            
            user = cursor.fetchone()
            if user:
                return {
                    "id": user['id'],
                    "username": user['username'],
                    "score_level1": user['score_level1'],
                    "score_level2": user['score_level2'],
                    "score_level3": user['score_level3'],
                    "score_total": user['score_total'],
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
                SELECT id, username, score_level1, score_level2, score_level3, score_total,
                       created_at, last_login, is_active
                FROM users WHERE username = ?
            ''', (username,))
            
            user = cursor.fetchone()
            if user:
                return {
                    "id": user['id'],
                    "username": user['username'],
                    "score_level1": user['score_level1'],
                    "score_level2": user['score_level2'],
                    "score_level3": user['score_level3'],
                    "score_total": user['score_total'],
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
    

    def update_user_score(self, username: str, new_score: int, level: int | None = None) -> bool:
        """更新使用者分數
        - 若提供 level (1/2/3)，更新對應關卡分數並重算總分
        - 否則更新總分 score_total
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # 取得使用者 id 與現有分數
            cursor.execute('SELECT id, score_level1, score_level2, score_level3 FROM users WHERE username = ?', (username,))
            row = cursor.fetchone()
            if not row:
                return False
            user_id = row['id']

            if level in (1, 2, 3):
                col = f"score_level{level}"
                cursor.execute(f'UPDATE users SET {col} = ? WHERE id = ?', (new_score, user_id))
                # 重新讀取三關分數計算總分
                l1 = new_score if level == 1 else row['score_level1']
                l2 = new_score if level == 2 else row['score_level2']
                l3 = new_score if level == 3 else row['score_level3']
                total = int(l1) + int(l2) + int(l3)
                cursor.execute('UPDATE users SET score_total = ? WHERE id = ?', (total, user_id))
                activity_desc = f"關卡{level}分數更新為 {new_score}，總分 {total}"
            else:
                # 直接更新總分
                cursor.execute('UPDATE users SET score_total = ? WHERE id = ?', (new_score, user_id))
                activity_desc = f"總分更新為 {new_score}"

            # 記錄活動
            cursor.execute('''
                INSERT INTO user_activities (user_id, activity_type, description)
                VALUES (?, ?, ?)
            ''', (user_id, "SCORE_UPDATE", activity_desc))

            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"更新使用者分數失敗: {e}")
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
                SELECT username, created_at, last_login
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
                    INSERT INTO users (username, hashed_password, created_at)
                    VALUES (?, ?, ?)
                ''', (username, user_data['hashed_password'], datetime.now(timezone.utc)))
                
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

    def get_all_users_with_scores(self) -> list[Dict[str, Any]]:
        """取得所有使用者與分數資訊（用於分數看板）"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, username, score_level1, score_level2, score_level3, score_total, created_at, last_login
                FROM users
                ORDER BY score_total DESC, username ASC
            ''')

            rows = cursor.fetchall()
            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "username": row["username"],
                    "score_level1": row["score_level1"],
                    "score_level2": row["score_level2"],
                    "score_level3": row["score_level3"],
                    "score_total": row["score_total"],
                    "created_at": row["created_at"],
                    "last_login": row["last_login"],
                })
            return result
        except Exception as e:
            logger.error(f"取得使用者分數清單失敗: {e}")
            raise
        finally:
            if conn:
                conn.close()

# 全域資料庫管理器實例
db_manager = DatabaseManager()
