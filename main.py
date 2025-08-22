from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import os
import uuid
import jwt
import bcrypt
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import logging
import random
from openai import OpenAI
import hashlib
import aiofiles
from fastapi.staticfiles import StaticFiles
import json
from google import genai
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import tensorflow as tf
from image_classification import ImageClassificationModel

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 影像辨識 ---
# 1. 用於存放預先載入的 MobileNet 基礎模型
GLOBAL_MOBILENET = None
# 2. 用於緩存不同用戶的已訓練模型，避免重複從硬碟讀取
USER_MODELS = {} 

# --- API 生命週期事件 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在應用程式啟動時執行的程式碼
    global GLOBAL_MOBILENET
    logger.info("伺服器啟動，開始載入 MobileNet V3 基礎模型...")
    try:
        GLOBAL_MOBILENET = tf.keras.applications.MobileNetV3Small(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        GLOBAL_MOBILENET.trainable = False
        # 預熱模型
        dummy_input = tf.zeros((1, 224, 224, 3))
        _ = GLOBAL_MOBILENET(dummy_input)
        logger.info("✅ MobileNet V3 基礎模型載入成功並已設定為全域共用！")
    except Exception as e:
        logger.error(f"❌ 載入 MobileNet 基礎模型失敗: {e}")
        GLOBAL_MOBILENET = None
    
    # 啟動定期清理任務
    import asyncio
    cleanup_task = asyncio.create_task(cleanup_expired_tokens_task())
    logger.info("✅ 定期清理過期 token 任務已啟動")
    
    yield
    
    # 在應用程式關閉時執行的程式碼
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    logger.info("伺服器正在關閉...")

app = FastAPI(
    title="AI Odyssey Backend API",
    description="AI 學習遊戲平台後端 API",
    version="2.0.0",
    lifespan=lifespan
)

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 安全設定
security = HTTPBearer()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 300

# 導入 SQLite 資料庫管理器
from database import db_manager

# 初始化資料庫（在應用程式啟動時會自動創建表格）
logger.info("正在初始化 SQLite 資料庫...")

# NCHC API 設定
NCHC_API_BASE_URL = "https://portal.genai.nchc.org.tw/api/v1"
API_KEY = os.getenv("NCHC_API_KEY")

if not API_KEY:
    logger.warning("NCHC_API_KEY 環境變數未設定")

# OpenAI API 設定
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY 環境變數未設定")
else:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Google Gemini API 設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY 環境變數未設定")
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

# 載入 system prompts
def load_system_prompts():
    """載入 system prompts 從 JSON 檔案"""
    try:
        with open("system_prompts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("prompts", [])
    except FileNotFoundError:
        logger.warning("system_prompts.json 檔案未找到，使用預設 prompt")
        return []
    except json.JSONDecodeError:
        logger.error("system_prompts.json 檔案格式錯誤")
        return []

# 載入 prompts
system_prompts = load_system_prompts()

# 確保 static/images 資料夾存在
STATIC_IMAGES_DIR = "static/images"
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

# 定期清理過期 token 的任務
async def cleanup_expired_tokens_task():
    """定期清理過期的 token 和會話"""
    import asyncio
    while True:
        try:
            await asyncio.sleep(3600)  # 每小時執行一次
            db_manager.cleanup_expired_tokens()
        except Exception as e:
            logger.error(f"清理過期 token 任務失敗: {e}")

# 設定靜態檔案服務 (在資料夾創建之後)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Pydantic 模型 - 認證相關
class UserRegister(BaseModel):
    username: str = Field(..., description="使用者名稱")
    password: str = Field(..., description="密碼")

class UserLogin(BaseModel):
    username: str = Field(..., description="使用者名稱")
    password: str = Field(..., description="密碼")

class User(BaseModel):
    id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000")
    username: str = Field(..., example="探險家小明")
    money: int = Field(500, example=500, description="遊戲金幣")

# Pydantic 模型 - LLM 對話相關
class LLMChatRequest(BaseModel):
    question: str = Field(..., 
                          example="什麼是影像辨識")
    system_prompt: str = Field(default=system_prompts[0]["content"] if system_prompts else "請輸入系統提示詞", 
                              description="系統提示詞，如果不填寫將使用預設的智識庫設定")

class LLMChatResponse(BaseModel):
    answer: str = Field(..., example="嘿，探險家！影像辨識就像是讓電腦學會『看』東西的超能力！就像你教小朋友認識動物一樣，我們教電腦認識圖片中的內容。想像一下，如果電腦能認出照片裡是一隻貓、一朵花，或者一個蘋果，那不是很酷嗎？🤖✨")


# Pydantic 模型 - 模組一：國王的厭食症
class GenerateRecipeTextRequest(BaseModel):
    prompt: str = Field(..., description="食譜生成提示詞")

class GenerateRecipeTextResponse(BaseModel):
    recipe_text: str = Field(..., example="✨ 魔法森林的彩虹水果沙拉 ✨\n\n這道菜就像森林精靈的魔法盛宴！我們需要：\n\n🍎 紅蘋果 - 切成小星星形狀\n🍊 橘子 - 剝成小月牙\n🍇 葡萄 - 像珍珠一樣閃亮\n🥝 奇異果 - 切成小圓片\n\n調味魔法：\n🍯 蜂蜜 - 森林的甜蜜精華\n🍋 檸檬汁 - 清新的魔法\n🌿 薄荷葉 - 森林的香氣\n\n做法：\n1. 將所有水果洗淨，切成可愛的形狀\n2. 輕輕混合，不要破壞水果的完整性\n3. 淋上蜂蜜和檸檬汁的魔法組合\n4. 最後點綴新鮮的薄荷葉\n\n這道沙拉不僅美味，還能讓國王想起森林的快樂時光！🌈")

class GenerateRecipeImageRequest(BaseModel):
    prompt: str = Field(..., example="宮保雞丁",description="圖片生成提示詞")

class GenerateRecipeImageResponse(BaseModel):
    image_url: str = Field(..., example="https://ai-odyssey-backend-rbzz.onrender.com/static/images/d1b094d315f4.png")
    prompt: str = Field(..., example="大便")
    enhanced_prompt: str = Field(..., example="Beautiful, appetizing food photography: 大便. High quality, professional food image, safe for all audiences, no inappropriate content.")
    model_used: str = Field(..., example="dall-e-3")
    image_size: str = Field(..., example="1024x1024")
    generation_time: str = Field(..., example="2024-01-15T10:30:45.123456")
    generation_date: str = Field(..., example="2024-01-15")
    generation_timestamp: float = Field(..., example=1705311045.123456)
    file_info: Dict[str, Any] = Field(..., example={
        "original_filename": "d1b094d315f4.png",
        "file_path": "static/images/d1b094d315f4.png",
        "full_url": "https://ai-odyssey-backend-rbzz.onrender.com/static/images/d1b094d315f4.png"
    })
    
class GenerateCustomImageResponse(BaseModel):
    image_url: str = Field(..., example="https://ai-odyssey-backend-rbzz.onrender.com/static/images/d1b094d315f4.png")
    prompt: str = Field(..., example="李久恩穿著紅色褲褲，手拿十字架，冷傲退基佬，寫實畫風，高清")
    model_used: str = Field(..., example="dall-e-3")
    image_size: str = Field(..., example="1024x1024")
    generation_time: str = Field(..., example="2024-01-15T10:30:45.123456")
    generation_date: str = Field(..., example="2024-01-15")
    generation_timestamp: float = Field(..., example=1705311045.123456)
    file_info: Dict[str, Any] = Field(..., example={
        "original_filename": "d1b094d315f4.png",
        "file_path": "static/images/d1b094d315f4.png",
        "full_url": "https://ai-odyssey-backend-rbzz.onrender.com/static/images/d1b094d315f4.png"
    })
    
class GenerateCustomImageRequest(BaseModel):
    prompt: str = Field(..., example="李久恩穿著紅色褲褲，手拿十字架，冷傲退基佬，寫實畫風，高清", description="客製化圖片生成")
    

# Pydantic 模型 - 食物圖片分析相關
class FoodImageAnalysisRequest(BaseModel):
    image_hash: str = Field(..., description="圖片12位數hash", example="a0f7cfb81fbf.png")
    dish_expect: Optional[str] = Field(default="我期待這是一道色香味俱全、營養均衡、擺盤精美的美食", description="對這道菜的期待描述，如果不填寫將使用預設期待")
    

class FoodImageAnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Gemini 對食物的詳細評語")
    score: int = Field(..., description="食物評分 (0-100分)", ge=0 , le=100)
    model_used: str = Field(..., description="使用的模型")
    image_path: str = Field(..., description="分析的圖片路徑")
    prompt: str = Field(..., description="使用的提示詞")
    dish_expect: str = Field(..., description="顧客對這道菜的期待描述")
    full_response_text: str = Field(..., description="Gemini 的完整回應文字")


# Pydantic 模型 - 模組二：池塘裡面銀龍魚和吳郭魚的辨識
class ImageInfo(BaseModel):
    name: str
    images: List[str]

class TrainingRequest(BaseModel):
    train_dataset: List[ImageInfo]
    
class PredictionRequest(BaseModel):
    image_path: str = Field(..., description="要進行預測的圖片在伺服器上的相對路徑")


# Pydantic 模型 - AI English Writing Teacher 專案
class ImageData(BaseModel):
    data: str = Field(..., description="base64編碼的圖片數據")
    mimeType: str = Field(..., description="圖片MIME類型")

class OCRRequest(BaseModel):
    images: List[ImageData] = Field(..., description="圖片列表")
    system_prompt: str = Field(..., description="系統提示詞")

class TopicOCRResponse(BaseModel):
    topic: str = Field(..., description="作文的主題或問題")
    requirements: str = Field(..., description="題目要求")
    score: str = Field(..., description="分數或佔比")

class EssayOCRResponse(BaseModel):
    recognized_text: str = Field(..., description="辨識出的完整英文作文文字")

# 作文批改相關
class TopicInfo(BaseModel):
    topic: str = Field(..., description="作文題目")
    requirements: str = Field(..., description="題目要求")
    score: int = Field(..., description="分數")

class CorrectionRequest(BaseModel):
    essay_text: str = Field(..., description="學生的英文作文內容")
    topic_info: TopicInfo = Field(..., description="題目資訊")
    system_prompt: str = Field(..., description="系統提示詞")

class ScoreDetail(BaseModel):
    score: float = Field(..., description="分數")
    comment: str = Field(..., description="評語")

class FeedbackItem(BaseModel):
    type: str = Field(..., description="錯誤類型")
    original: str = Field(..., description="原始錯誤")
    suggestion: str = Field(..., description="建議修正")
    explanation: str = Field(..., description="解釋說明")

class CorrectionResponse(BaseModel):
    scores: Dict[str, ScoreDetail] = Field(..., description="各項評分")
    total_score: float = Field(..., description="總分")
    detailed_feedback: List[FeedbackItem] = Field(..., description="詳細回饋")
    overall_comment: str = Field(..., description="整體評語")
    rewritten_essay: str = Field(..., description="改寫後的範文")

# 講義生成相關
class HandoutRequest(BaseModel):
    correction_data: Dict[str, List[FeedbackItem]] = Field(..., description="批改資料")
    system_prompt: str = Field(..., description="系統提示詞")

class HandoutResponse(BaseModel):
    handout_content: str = Field(..., description="生成的講義內容")

# 練習題生成相關
class PracticeRequest(BaseModel):
    correction_data: Dict[str, List[FeedbackItem]] = Field(..., description="批改資料")
    num_questions: int = Field(..., description="題目數量")
    num_versions: int = Field(..., description="版本數量")
    system_prompt: str = Field(..., description="系統提示詞")

class PracticeQuestion(BaseModel):
    question: str = Field(..., description="題目內容")
    options: List[str] = Field(..., description="選項")

class PracticeAnswer(BaseModel):
    answer: str = Field(..., description="答案")
    explanation: str = Field(..., description="答案解釋")

class PracticeVersion(BaseModel):
    title: str = Field(..., description="版本標題")
    questions: List[PracticeQuestion] = Field(..., description="題目列表")
    answers: List[PracticeAnswer] = Field(..., description="答案列表")

class PracticeResponse(BaseModel):
    versions: List[PracticeVersion] = Field(..., description="練習題版本")

# 通用回應格式
class AIWritingTeacherResponse(BaseModel):
    success: bool = Field(..., description="是否成功")
    data: Optional[Dict[str, Any]] = Field(None, description="回應資料")
    error: Optional[Dict[str, Any]] = Field(None, description="錯誤資訊")

class SuccessResponse(BaseModel):
    status: str = "success"
    data: Dict[str, Any]

class ErrorResponse(BaseModel):
    status: str = "error"
    message: str

# 工具函數
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        
        # 先檢查 token 是否在黑名單中
        if db_manager.is_token_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token 已被撤銷",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="無效的認證憑證",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 從資料庫驗證使用者
        user = db_manager.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="使用者不存在",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # 檢查會話是否仍然有效
        if not db_manager.is_session_valid(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="會話已過期或已被登出",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return username
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="無效的認證憑證",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_api_key():
    """取得 API Key"""
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API Key 未設定，請設定 NCHC_API_KEY 環境變數"
        )
    return API_KEY

async def get_openai_client():
    """取得 OpenAI 客戶端"""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OpenAI API Key 未設定，請設定 OPENAI_API_KEY 環境變數"
        )
    return openai_client


async def download_and_save_image(image_url: str, prompt: str) -> str:
    """下載圖片並保存到本地，返回本地檔案路徑"""
    try:
        # 生成隨機 hash 值
        timestamp = str(datetime.now(timezone.utc).timestamp())
        random_salt = str(random.randint(1000, 9999))
        hash_input = f"{prompt}{timestamp}{random_salt}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        # 檔案名稱
        filename = f"{file_hash}.png"
        filepath = os.path.join(STATIC_IMAGES_DIR, filename)
        
        # 下載圖片
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            
            # 保存圖片到本地
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(response.content)
        
        logger.info(f"圖片已保存到: {filepath}")
        return filename
        
    except Exception as e:
        logger.error(f"下載圖片失敗: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"下載圖片失敗: {str(e)}"
        )

# API 端點

@app.get("/")
async def root():
    """根路徑"""
    return {
        "message": "AI Odyssey Backend API",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy", "api_key_configured": bool(API_KEY)}

# 1. 帳號與認證 (Authentication)

@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """使用者註冊"""
    try:
        # 使用資料庫管理器創建使用者
        user = db_manager.create_user(
            username=user_data.username,
            password=user_data.password,
            initial_money=0
        )
        
        return {
            "message": "註冊成功！",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "money": user["money"]
            }
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"註冊失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="註冊過程中發生錯誤，請稍後再試。"
        )

@app.post("/auth/login")
async def login(user_data: UserLogin):
    """使用者登入"""
    try:
        # 使用資料庫管理器驗證使用者
        user = db_manager.verify_user(user_data.username, user_data.password)
        
        if not user:
            # 為了安全性，不區分帳號不存在和密碼錯誤
            # 但提供更友善的錯誤訊息
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登入失敗：請檢查您的帳號名稱和密碼是否正確。如果忘記密碼，請聯繫管理員。",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # 建立 JWT token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_data.username}, expires_delta=access_token_expires
        )
        
        # 創建資料庫會話記錄
        expires_at = datetime.now(timezone.utc) + access_token_expires
        db_manager.create_session(user["id"], access_token, expires_at)
        
        # 記錄成功登入
        logger.info(f"使用者 {user_data.username} 登入成功")
        
        return {
            "message": "登入成功！",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "money": user["money"]
            }
        }
        
    except ValueError as e:
        # 處理特定的業務邏輯錯誤（如帳號被停用）
        error_message = str(e)
        if "帳號已被停用" in error_message:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="帳號已被停用，請聯繫管理員。"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_message
            )
    except HTTPException:
        # 重新拋出 HTTP 異常
        raise
    except Exception as e:
        # 記錄詳細錯誤資訊
        logger.error(f"登入過程中發生未預期錯誤: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登入過程中發生系統錯誤，請稍後再試。如果問題持續發生，請聯繫技術支援。"
        )

@app.get("/users/me", response_model=SuccessResponse)
async def get_current_user(username: str = Depends(verify_token)):
    """獲取當前使用者資訊"""
    try:
        user = db_manager.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="使用者不存在"
            )
        
        return SuccessResponse(data={
            "user": {
                "id": user["id"],
                "username": user["username"],
                "money": user["money"],
                "created_at": user["created_at"],
                "last_login": user["last_login"]
            }
        })
        
    except Exception as e:
        logger.error(f"獲取使用者資訊失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="獲取使用者資訊失敗"
        )

@app.post("/auth/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """使用者登出"""
    try:
        # 使會話無效
        token = credentials.credentials
        
        # 先檢查 token 是否有效
        if not db_manager.is_session_valid(token):
            return {"message": "會話已過期或不存在"}
        
        # 撤銷 token
        if db_manager.invalidate_session(token):
            logger.info(f"使用者登出成功，token: {token[:20]}...")
            return {"message": "登出成功！"}
        else:
            logger.warning(f"登出時 token 無效: {token[:20]}...")
            return {"message": "會話已過期或不存在"}
            
    except Exception as e:
        logger.error(f"登出失敗: {e}")
        # 即使失敗也要嘗試撤銷 token
        try:
            token = credentials.credentials
            db_manager.invalidate_session(token)
        except:
            pass
        return {"message": "登出成功！"}  # 即使失敗也回傳成功，避免前端錯誤

@app.post("/llmchat", response_model=LLMChatResponse)
async def chat_with_llm(
    request: LLMChatRequest,
    username: str = Depends(verify_token),      
    api_key: str = Depends(get_api_key)
):
    """與 LLM 進行對話，使用 gpt-oss-120b 模型，temp 0.4，max_tokens 2000，使用system_prompt作為系統提示詞（如果不填寫將使用預設的智識庫設定），使用question作為使用者問題"""
    try:
        # 建立訊息列表
        messages = [
            {
                "role": "system",
                "content": request.system_prompt
            },
            {
                "role": "user",
                "content": request.question
            }
        ]
        
        # 準備請求
        chat_request = {
            "model": "gpt-oss-120b",
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 2000
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NCHC_API_BASE_URL}/chat/completions",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json=chat_request,
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"NCHC API 錯誤: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"NCHC API 錯誤: {response.text}"
                )
            
            result = response.json()
            
            # 提取回應內容
            if result.get("choices") and len(result["choices"]) > 0:
                answer = result["choices"][0]["message"]["content"]
            else:
                answer = "抱歉，我無法回答這個問題。"
            
            return LLMChatResponse(answer=answer)
            
    except httpx.TimeoutException:
        logger.error("請求超時")
        raise HTTPException(status_code=408, detail="請求超時")
    except httpx.RequestError as e:
        logger.error(f"請求錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"請求錯誤: {str(e)}")
    except Exception as e:
        logger.error(f"未預期錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"未預期錯誤: {str(e)}")

# 3. 模組一：國王的厭食症 (Generative AI)

@app.post("/module1/generate-recipe-text", response_model=GenerateRecipeTextResponse)
async def generate_recipe_text(
    request: GenerateRecipeTextRequest,
    username: str = Depends(verify_token),
    api_key: str = Depends(get_api_key)
):
    """根據玩家輸入的提示詞 (Prompt)，生成菜色的文字描述。"""
    try:
        # 建立訊息列表
        messages = [
            {
                "role": "system",
                "content": "你是一個專業的廚師和美食作家。請根據使用者的要求，創造出富有創意和想像力的食譜描述。描述要生動有趣，符合童話故事的風格。"
            },
            {
                "role": "user",
                "content": request.prompt
            }
        ]
        
        # 準備請求
        chat_request = {
            "model": "gpt-oss-120b",
            "messages": messages,
            "temperature": 0.8,
            "max_tokens": 1500
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NCHC_API_BASE_URL}/chat/completions",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json=chat_request,
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"NCHC API 錯誤: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"NCHC API 錯誤: {response.text}"
                )
            
            result = response.json()
            
            # 提取回應內容
            if result.get("choices") and len(result["choices"]) > 0:
                recipe_text = result["choices"][0]["message"]["content"]
            else:
                recipe_text = "抱歉，我無法生成食譜描述。"
            
            return GenerateRecipeTextResponse(recipe_text=recipe_text)
            
    except httpx.TimeoutException:
        logger.error("請求超時")
        raise HTTPException(status_code=408, detail="請求超時")
    except httpx.RequestError as e:
        logger.error(f"請求錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"請求錯誤: {str(e)}")
    except Exception as e:
        logger.error(f"未預期錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"未預期錯誤: {str(e)}")


@app.post("/generate-custom-image", response_model=GenerateCustomImageResponse)
async def generate_custom_image(
    request: GenerateCustomImageRequest,
    http_request: Request,
    username: str = Depends(verify_token),
    client: OpenAI = Depends(get_openai_client)
):
    """完全客製化圖片生成，使用 DALL-E 3 模型"""
    try:      
        # 使用 DALL-E 生成圖片
        response = client.images.generate(
            model="dall-e-3",
            prompt=request.prompt,
            n=1,
            size="1024x1024"
        )
        
        # 直接取得圖片 URL
        if response.data and len(response.data) > 0:
            image_url = response.data[0].url
        else:
            raise HTTPException(
                status_code=500,
                detail="圖片生成失敗"
            )
        
        # 下載並保存圖片到本地
        filename = await download_and_save_image(image_url, request.prompt)
        
        # 返回本地圖片 URL
        base_url = str(http_request.base_url).rstrip('/')
        local_image_url = f"{base_url}/static/images/{filename}"
        
        # 記錄生成時間
        generation_time = datetime.now(timezone.utc)
        
        return GenerateCustomImageResponse(
            image_url=local_image_url,
            prompt=request.prompt,
            model_used="dall-e-3",
            image_size="1024x1024",
            generation_time=generation_time.isoformat(),
            generation_date=generation_time.strftime("%Y-%m-%d"),
            generation_timestamp=generation_time.timestamp(),
            file_info={
                "original_filename": filename,
                "file_path": f"static/images/{filename}",
                "full_url": local_image_url
            }
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"圖片生成錯誤: {error_msg}")
        
        # 處理特定的 OpenAI 錯誤
        if "content_policy_violation" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="提示詞內容違反安全政策，請使用更適當的描述。"
            )
        elif "safety_system" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="提示詞被安全系統拒絕，請避免使用可能不當的詞彙。"
            )
        elif "400" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="請求格式錯誤，請檢查提示詞內容。建議：使用清晰、具體的食物描述。"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"圖片生成錯誤: {error_msg}"
            )


@app.post("/module1/generate-recipe-image", response_model=GenerateRecipeImageResponse)
async def generate_recipe_image(
    request: GenerateRecipeImageRequest,
    http_request: Request,
    username: str = Depends(verify_token),
    client: OpenAI = Depends(get_openai_client)
):
    """將食譜文字描述傳給後端，生成對應的菜色圖片。"""
    try:
        # 優化提示詞，增加安全性和具體性
        enhanced_prompt = f"Beautiful, appetizing food photography: {request.prompt}. High quality, professional food image."
        
        # 使用 DALL-E 生成圖片
        response = client.images.generate(
            model="dall-e-3",
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024"
        )
        
        # 直接取得圖片 URL
        if response.data and len(response.data) > 0:
            image_url = response.data[0].url
        else:
            raise HTTPException(
                status_code=500,
                detail="圖片生成失敗"
            )
        
        # 下載並保存圖片到本地
        filename = await download_and_save_image(image_url, request.prompt)
        
        # 返回本地圖片 URL
        base_url = str(http_request.base_url).rstrip('/')
        local_image_url = f"{base_url}/static/images/{filename}"
        
        # 記錄生成時間
        generation_time = datetime.now(timezone.utc)
        
        return GenerateRecipeImageResponse(
            image_url=local_image_url,
            prompt=request.prompt,
            enhanced_prompt=enhanced_prompt,
            model_used="dall-e-3",
            image_size="1024x1024",
            generation_time=generation_time.isoformat(),
            generation_date=generation_time.strftime("%Y-%m-%d"),
            generation_timestamp=generation_time.timestamp(),
            file_info={
                "original_filename": filename,
                "file_path": f"static/images/{filename}",
                "full_url": local_image_url
            }
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"圖片生成錯誤: {error_msg}")
        
        # 處理特定的 OpenAI 錯誤
        if "content_policy_violation" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="提示詞內容違反安全政策，請使用更適當的描述。建議：描述具體的食材、烹飪方式、菜色外觀等。"
            )
        elif "safety_system" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="提示詞被安全系統拒絕，請避免使用可能不當的詞彙。建議：專注於食物的正面描述。"
            )
        elif "400" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="請求格式錯誤，請檢查提示詞內容。建議：使用清晰、具體的食物描述。"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"圖片生成錯誤: {error_msg}"
            )

@app.post("/module1/analyze-food-image", response_model=FoodImageAnalysisResponse)
async def analyze_food_image(
    request: FoodImageAnalysisRequest,
    username: str = Depends(verify_token)
):
    """使用 Gemini 分析食物圖片並給出評語"""
    try:
        # 檢查圖片是否存在
        image_path = os.path.join(STATIC_IMAGES_DIR, f"{request.image_hash}")
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail=f"找不到圖片: {image_path}"
            )
        
        # 讀取圖片文件
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 使用 Gemini 分析圖片
        from google.genai import types
        
        # 動態生成 prompt，讓 AI 以嚴格餐廳廚師的身份來評分
        chef_prompt = f"""你是一位經驗豐富、要求嚴格的米其林星級餐廳主廚。請以專業廚師的眼光，嚴格分析這張食物圖片。
對食物的期待：{request.dish_expect}
請按照以下固定格式回覆：
SCORE: [0-100分]
(評分標準：0-59分=不合格，60-70分=及格，71-85分=良好，86-95分=優秀，96-100分=完美)

ANALYSIS:
[請以嚴格廚師的角度，詳細評語(不要超過150字，不要條列式回答，不要使用markdown格式語法如"**")，包括：
1. 食物名稱和類型識別
2. 外觀、顏色、擺盤的專業評估
3. 食材選擇分析
4. 與顧客期待的對比分析
5. 改進建議和專業點評]

請確保回覆格式完全按照上述模板，SCORE 必須是 0-100 的整數分數。以嚴格的專業標準來評分。"""
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                chef_prompt
            ]
        )
        
        # 解析 AI 回覆，提取分數和評語
        response_text = response.text
        
        # 嘗試從回覆中提取分數和評語
        score = 50  # 預設分數
        analysis = response_text  # 預設使用完整回覆
        
        try:
            # 尋找 SCORE: 標記
            if "SCORE:" in response_text:
                score_line = response_text.split("SCORE:")[1].split("\n")[0].strip()
                # 提取數字
                import re
                score_match = re.search(r'\d+', score_line)
                if score_match:
                    score = int(score_match.group())
                    # 確保分數在 0-100 範圍內
                    score = max(0, min(100, score))
            
            # 尋找 ANALYSIS: 標記
            if "ANALYSIS:" in response_text:
                analysis_parts = response_text.split("ANALYSIS:")
                if len(analysis_parts) > 1:
                    analysis = analysis_parts[1].strip()
                else:
                    analysis = response_text
            else:
                # 如果沒有找到標記，嘗試智能分割
                lines = response_text.split('\n')
                # 跳過可能包含分數的前幾行，取後面的內容作為評語
                analysis_lines = []
                for line in lines:
                    if not line.strip().startswith('SCORE:') and line.strip():
                        analysis_lines.append(line)
                if analysis_lines:
                    analysis = '\n'.join(analysis_lines).strip()
                    
        except Exception as e:
            logger.warning(f"解析 AI 回覆時出現問題: {e}，使用預設值")
            score = 50
            analysis = response_text
        
        return FoodImageAnalysisResponse(
            analysis=analysis,
            score=score,
            model_used="gemini-2.5-flash",
            full_response_text=response_text,
            image_path=image_path,
            prompt=chef_prompt,
            dish_expect=request.dish_expect
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"圖片文件不存在: {image_path}"
        )
    except Exception as e:
        logger.error(f"圖片分析錯誤: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"圖片分析錯誤: {str(e)}"
        )


# 2. 模組二：池塘裡面銀龍魚和吳郭魚的辨識
@app.post("/module2/train/{user_name}")
async def train_user_model(
    user_name: str, 
    request: TrainingRequest,
    current_user: str = Depends(verify_token)
):
    """訓練玩家的模型，回傳訓練結果"""
    # 驗證使用者只能訓練自己的模型
    if user_name != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只能訓練自己的模型"
        )
    
    if GLOBAL_MOBILENET is None:
        return {"success": False, "error": "基礎模型尚未準備就緒，請稍後再試。"}

    # 1. 為此使用者取得或創建一個模型管理器實例
    #    這裡使用快取 USER_MODELS 來避免重複創建對象
    if user_name not in USER_MODELS:
        USER_MODELS[user_name] = ImageClassificationModel(user_name=user_name, base_model=GLOBAL_MOBILENET)
    
    model_manager = USER_MODELS[user_name]
    
    # 2. 載入訓練數據
    data_config = request.model_dump()
    if not model_manager.load_training_data(data_config):
        return {"success": False, "error": "載入訓練數據失敗，請檢查圖片路徑。"}
        
    # 3. 執行訓練
    result = await model_manager.train_model()
    
    # 訓練完成後，更新快取中的模型狀態
    USER_MODELS[user_name] = model_manager
    
    return result

@app.post("/module2/predict/{user_name}")
async def predict_with_user_model(
    user_name: str, 
    request: PredictionRequest,
    current_user: str = Depends(verify_token)
):
    """使用玩家的模型預測魚的種類，回傳預測結果"""
    # 驗證使用者只能使用自己的模型
    if user_name != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只能使用自己的模型"
        )
    
    if GLOBAL_MOBILENET is None:
        return {"success": False, "error": "基礎模型尚未準備就緒，請稍後再試。"}

    # 1. 取得此使用者的模型管理器
    #    如果不在快取中，就創建一個新的。__init__ 會自動嘗試從硬碟載入已保存的模型
    if user_name not in USER_MODELS:
        USER_MODELS[user_name] = ImageClassificationModel(user_name=user_name, base_model=GLOBAL_MOBILENET)

    model_manager = USER_MODELS[user_name]

    # 2. 檢查模型是否已訓練
    if not model_manager.is_trained:
        return {"success": False, "error": f"使用者 {user_name} 的模型尚未訓練，請先調用訓練 API。"}
    
    # 3. 執行預測
    prediction = model_manager.predict_image(request.image_path)
    return prediction
   

# ============================================================================
# AI English Writing Teacher 專案 API 端點
# ============================================================================

# 1. 題目圖片辨識接口
@app.post("/api/ocr/topic", response_model=AIWritingTeacherResponse)
async def ocr_topic(request: OCRRequest):
    """題目圖片辨識接口"""
    try:
        # 檢查是否有圖片
        if not request.images:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INVALID_REQUEST",
                    "message": "請提供圖片",
                    "details": "images 欄位不能為空"
                }
            )
        
        # 使用 Gemini 進行圖片辨識
        from google.genai import types
        
        # 配置 Gemini 客戶端
        gemini_api_key = os.getenv("AI_ENGLISH_WRITING_TEACHER")
        if not gemini_api_key:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "Gemini API Key 未設定",
                    "details": "請設定 AI_ENGLISH_WRITING_TEACHER 環境變數"
                }
            )
        
        client = genai.Client(api_key=gemini_api_key)
        
        # 處理第一張圖片（題目辨識通常只需要一張）
        image_data = request.images[0]
        
        # 將 base64 轉換為 bytes
        import base64
        try:
            image_bytes = base64.b64decode(image_data.data)
        except Exception as e:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INVALID_IMAGE_FORMAT",
                    "message": "圖片格式錯誤",
                    "details": f"base64 解碼失敗: {str(e)}"
                }
            )
        
        # 使用 Gemini 進行圖片辨識
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image_data.mimeType,
                ),
                request.system_prompt
            ]
        )
        
        # 解析回應（假設 AI 會返回 JSON 格式）
        response_text = response.text
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            if json_end != -1:
                json_str = response_text[json_start:json_end].strip()
                result = json.loads(json_str)
            else:
                result = {"topic": "無法解析", "requirements": "無法解析", "score": "無法解析"}
        else:
            # 如果沒有 JSON 標記，嘗試直接解析
            result = json.loads(response_text)

        return AIWritingTeacherResponse(
            success=True,
            data=result,
            error=None
        )
        
    except json.JSONDecodeError:
        # 如果 JSON 解析失敗，使用預設格式
        result = {
            "topic": "圖片辨識完成，但格式解析失敗",
            "requirements": "請檢查 AI 回應格式",
            "score": "無法確定"
        }
        return AIWritingTeacherResponse(success=True, data=result, error=None)
    except Exception as e:
        logger.error(f"題目圖片辨識失敗: {e}")
        return AIWritingTeacherResponse(
            success=False,
            data=None,
            error={
                "code": "OCR_FAILED",
                "message": "圖片辨識失敗",
                "details": str(e)
            }
        )

# 2. 手寫作文辨識接口
@app.post("/api/ocr/essay", response_model=AIWritingTeacherResponse)
async def ocr_essay(request: OCRRequest):
    """手寫作文辨識接口"""
    try:
        # 檢查是否有圖片
        if not request.images:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INVALID_REQUEST",
                    "message": "請提供圖片",
                    "details": "images 欄位不能為空"
                }
            )
        
        # 使用 Gemini 進行圖片辨識
        from google.genai import types
        
        # 配置 Gemini 客戶端
        gemini_api_key = os.getenv("AI_ENGLISH_WRITING_TEACHER")
        if not gemini_api_key:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "Gemini API Key 未設定",
                    "details": "請設定 AI_ENGLISH_WRITING_TEACHER 環境變數"
                }
            )
        
        client = genai.Client(api_key=gemini_api_key)
        
        # 處理所有圖片（手寫作文可能需要多張圖片）
        all_texts = []
        
        for image_data in request.images:
            # 將 base64 轉換為 bytes
            import base64
            try:
                image_bytes = base64.b64decode(image_data.data)
            except Exception as e:
                return AIWritingTeacherResponse(
                    success=False,
                    data=None,
                    error={
                        "code": "INVALID_IMAGE_FORMAT",
                        "message": "圖片格式錯誤",
                        "details": f"base64 解碼失敗: {str(e)}"
                    }
                )
            
            # 使用 Gemini 進行圖片辨識
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=image_data.mimeType,
                    ),
                    request.system_prompt
                ]
            )
            
            all_texts.append(response.text)
        
        # 合併所有辨識出的文字
        combined_text = " ".join(all_texts)
        
        return AIWritingTeacherResponse(
            success=True,
            data={"recognized_text": combined_text},
            error=None
        )
        
    except Exception as e:
        logger.error(f"手寫作文辨識失敗: {e}")
        return AIWritingTeacherResponse(
            success=False,
            data=None,
            error={
                "code": "OCR_FAILED",
                "message": "手寫作文辨識失敗",
                "details": str(e)
            }
        )

# 3. 作文批改接口
@app.post("/api/ai/correction", response_model=AIWritingTeacherResponse)
async def correct_essay(request: CorrectionRequest):
    """作文批改接口"""
    try:
        # 使用 gpt-oss-120b 進行作文批改
        nchc_api_key = os.getenv("NCHC_API_KEY")
        if not nchc_api_key:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "NCHC API Key 未設定",
                    "details": "請設定 NCHC_API_KEY 環境變數"
                }
            )
        
        # 準備請求
        messages = [
            {
                "role": "system",
                "content": request.system_prompt
            },
            {
                "role": "user",
                "content": f"題目資訊：{request.topic_info.topic}\n要求：{request.topic_info.requirements}\n分數：{request.topic_info.score}\n\n學生作文：{request.essay_text}"
            }
        ]
        
        chat_request = {
            "model": "gpt-oss-120b",
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 20000
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NCHC_API_BASE_URL}/chat/completions",
                headers={
                    "x-api-key": nchc_api_key,
                    "Content-Type": "application/json"
                },
                json=chat_request,
                timeout=60.0
            )
            
            if response.status_code != 200:
                return AIWritingTeacherResponse(
                    success=False,
                    data=None,
                    error={
                        "code": "AI_GENERATION_FAILED",
                        "message": "AI 生成失敗",
                        "details": f"NCHC API 錯誤: {response.text}"
                    }
                )
            
            result = response.json()
            
            # 提取回應內容
            if result.get("choices") and len(result["choices"]) > 0:
                ai_response = result["choices"][0]["message"]["content"]
            else:
                return AIWritingTeacherResponse(
                    success=False,
                    data=None,
                    error={
                        "code": "AI_GENERATION_FAILED",
                        "message": "AI 回應格式錯誤",
                        "details": "無法提取 AI 回應內容"
                    }
                )
            
            # 嘗試解析 AI 回應（假設會返回結構化數據）
            try:
                import json
                if "```json" in ai_response:
                    json_start = ai_response.find("```json") + 7
                    json_end = ai_response.find("```", json_start)
                    if json_end != -1:
                        json_str = ai_response[json_start:json_end].strip()
                        correction_data = json.loads(json_str)
                    else:
                        correction_data = {"error": "無法解析 JSON 格式"}
                else:
                    correction_data = {"raw_response": ai_response}
                
                return AIWritingTeacherResponse(
                    success=True,
                    data=correction_data,
                    error=None
                )
                
            except json.JSONDecodeError:
                return AIWritingTeacherResponse(
                    success=True,
                    data={"raw_response": ai_response},
                    error=None
                )
                
    except Exception as e:
        logger.error(f"作文批改失敗: {e}")
        return AIWritingTeacherResponse(
            success=False,
            data=None,
            error={
                "code": "AI_GENERATION_FAILED",
                "message": "作文批改失敗",
                "details": str(e)
            }
        )

# 4. 講義生成接口
@app.post("/api/ai/handout", response_model=AIWritingTeacherResponse)
async def generate_handout(request: HandoutRequest):
    """講義生成接口"""
    try:
        # 使用 gpt-oss-120b 生成講義
        nchc_api_key = os.getenv("NCHC_API_KEY")
        if not nchc_api_key:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "NCHC API Key 未設定",
                    "details": "請設定 NCHC_API_KEY 環境變數"
                }
            )
        
        # 準備請求
        messages = [
            {
                "role": "system",
                "content": request.system_prompt
            },
            {
                "role": "user",
                "content": f"批改資料：{json.dumps(jsonable_encoder(request.correction_data), ensure_ascii=False)}"
            }
        ]
        
        chat_request = {
            "model": "gpt-oss-120b",
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 20000
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NCHC_API_BASE_URL}/chat/completions",
                headers={
                    "x-api-key": nchc_api_key,
                    "Content-Type": "application/json"
                },
                json=chat_request,
                timeout=60.0
            )
            
            if response.status_code != 200:
                return AIWritingTeacherResponse(
                    success=False,
                    data=None,
                    error={
                        "code": "AI_GENERATION_FAILED",
                        "message": "講義生成失敗",
                        "details": f"NCHC API 錯誤: {response.text}"
                    }
                )
            
            result = response.json()
            
            # 提取回應內容
            if result.get("choices") and len(result["choices"]) > 0:
                handout_content = result["choices"][0]["message"]["content"]
            else:
                return AIWritingTeacherResponse(
                    success=False,
                    data=None,
                    error={
                        "code": "AI_GENERATION_FAILED",
                        "message": "講義生成失敗",
                        "details": "無法提取 AI 回應內容"
                    }
                )
            
            return AIWritingTeacherResponse(
                success=True,
                data={"handout_content": handout_content},
                error=None
            )
                
    except Exception as e:
        logger.error(f"講義生成失敗: {e}")
        return AIWritingTeacherResponse(
            success=False,
            data=None,
            error={
                "code": "AI_GENERATION_FAILED",
                "message": "講義生成失敗",
                "details": str(e)
            }
        )

# 5. 練習題生成接口
@app.post("/api/ai/practice", response_model=AIWritingTeacherResponse)
async def generate_practice_questions(request: PracticeRequest):
    """練習題生成接口"""
    try:
        # 使用 gpt-oss-120b 生成練習題
        nchc_api_key = os.getenv("NCHC_API_KEY")
        if not nchc_api_key:
            return AIWritingTeacherResponse(
                success=False,
                data=None,
                error={
                    "code": "INTERNAL_ERROR",
                    "message": "NCHC API Key 未設定",
                    "details": "請設定 NCHC_API_KEY 環境變數"
                }
            )
        
        # 準備請求
        messages = [
            {
                "role": "system",
                "content": request.system_prompt
            },
            {
                "role": "user",
                "content": f"批改資料：{json.dumps(jsonable_encoder(request.correction_data), ensure_ascii=False)}\n題目數量：{request.num_questions}\n版本數量：{request.num_versions}"
            }
        ]
        
        chat_request = {
            "model": "gpt-oss-120b",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 20000
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NCHC_API_BASE_URL}/chat/completions",
                headers={
                    "x-api-key": nchc_api_key,
                    "Content-Type": "application/json"
                },
                json=chat_request,
                timeout=60.0
            )
            
            if response.status_code != 200:
                return AIWritingTeacherResponse(
                    success=False,
                    data=None,
                    error={
                        "code": "AI_GENERATION_FAILED",
                        "message": "練習題生成失敗",
                        "details": f"NCHC API 錯誤: {response.text}"
                    }
                )
            
            result = response.json()
            
            # 提取回應內容
            if result.get("choices") and len(result["choices"]) > 0:
                practice_content = result["choices"][0]["message"]["content"]
            else:
                return AIWritingTeacherResponse(
                    success=False,
                    data=None,
                    error={
                        "code": "AI_GENERATION_FAILED",
                        "message": "練習題生成失敗",
                        "details": "無法提取 AI 回應內容"
                    }
                )
            
            # 嘗試解析 AI 回應
            if "```json" in practice_content:
                json_start = practice_content.find("```json") + 7
                json_end = practice_content.find("```", json_start)
                if json_end != -1:
                    json_str = practice_content[json_start:json_end].strip()
                    practice_data = json.loads(json_str)
                else:
                    practice_data = {"raw_response": practice_content}
            else:
                practice_data = {"raw_response": practice_content}
            
            return AIWritingTeacherResponse(
                success=True,
                data=practice_data,
                error=None
            )
                
    except json.JSONDecodeError:
        return AIWritingTeacherResponse(
            success=True,
            data={"raw_response": "AI 回應的 JSON 解析失敗"},
            error=None
        )
    except Exception as e:
        logger.error(f"練習題生成失敗: {e}")
        return AIWritingTeacherResponse(
            success=False,
            data=None,
            error={
                "code": "AI_GENERATION_FAILED",
                "message": "練習題生成失敗",
                "details": str(e)
            }
        )

@app.get("/system-prompts")
async def get_system_prompts():
    """取得可用的 system prompts 列表"""
    return {
        "prompts": system_prompts
    }

@app.get("/users/{username}/statistics")
async def get_user_statistics(username: str, current_user: str = Depends(verify_token)):
    """獲取使用者統計資訊（需要登入）"""
    try:
        # 檢查是否為當前使用者或管理員
        if username != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="只能查看自己的統計資訊"
            )
        
        user = db_manager.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="使用者不存在"
            )
        
        statistics = db_manager.get_user_statistics(user["id"])
        return SuccessResponse(data=statistics)
        
    except Exception as e:
        logger.error(f"獲取使用者統計資訊失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="獲取統計資訊失敗"
        )

@app.get("/models")
async def list_models():
    """取得可用模型列表 - 向後相容"""
    return {
        "models": [
            "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "gpt-oss-120b",
            "gemini-2.5-flash",
            "dall-e-3",
            "gemini-embedding-001"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
