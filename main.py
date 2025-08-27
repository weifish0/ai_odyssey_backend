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
from core.config import NCHC_API_BASE_URL, OPENAI_API_KEY, GEMINI_API_KEY
from core.deps import verify_token
from ml_state import init_mobilenet
from routers import misc as misc_router, auth as auth_router
from routers import module1 as module1_router
from routers import module2 as module2_router
from routers import rag as rag_router
from routers import ai_teacher as ai_teacher_router
from routers import food_analysis as food_analysis_router
from routers import users as users_router
from routers import llm as llm_router

# 載入環境變數
load_dotenv()

# 強制 TensorFlow 使用 CPU，避免 CUDA 錯誤
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 減少 TensorFlow 警告訊息

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 影像辨識 --- 由 ml_state 管理模型狀態

# --- API 生命週期事件 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在應用程式啟動時執行的程式碼
    
    # 配置 TensorFlow 以減少警告和錯誤
    logger.info("正在配置 TensorFlow 環境...")
    try:
        # 設定 TensorFlow 日誌級別
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
        # 檢查可用的設備
        devices = tf.config.list_physical_devices()
        logger.info(f"可用的 TensorFlow 設備: {[device.device_type for device in devices]}")
        
        # 如果有 GPU 但想要強制使用 CPU
        if tf.config.list_physical_devices('GPU'):
            logger.info("檢測到 GPU，但強制使用 CPU 模式")
            tf.config.set_visible_devices([], 'GPU')
        
        logger.info("✅ TensorFlow 環境配置完成")
    except Exception as e:
        logger.warning(f"TensorFlow 環境配置警告: {e}")
    
    logger.info("伺服器啟動，開始載入 MobileNet V3 基礎模型...")
    success = init_mobilenet()
    if success:
        logger.info("✅ MobileNet 模型初始化成功")
    else:
        logger.error("❌ MobileNet 模型初始化失敗")
    
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

# 導入 SQLite 資料庫管理器
from database import db_manager

# 導入 RAG 查詢引擎
from query_engine import load_vector_database, search

# 初始化資料庫（在應用程式啟動時會自動創建表格）
logger.info("正在初始化 SQLite 資料庫...")

API_KEY = os.getenv("NCHC_API_KEY")
if not API_KEY:
    logger.warning("NCHC_API_KEY 環境變數未設定")

# OpenAI API 設定
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY 環境變數未設定")

# Google Gemini API 設定
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY 環境變數未設定")
else:
    client = genai.Client(api_key=GEMINI_API_KEY)

# 載入 prompts 改由 core.config 提供
from core.config import SYSTEM_PROMPTS as system_prompts

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

# 掛載 Routers
app.include_router(misc_router.router)
app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(module1_router.router)
app.include_router(module2_router.router)
app.include_router(llm_router.router)
app.include_router(rag_router.router)
app.include_router(food_analysis_router.router)
app.include_router(ai_teacher_router.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
