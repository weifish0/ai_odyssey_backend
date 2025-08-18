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
from datetime import datetime, timedelta
from dotenv import load_dotenv
import logging
import random
from openai import OpenAI
import hashlib
import aiofiles
from fastapi.staticfiles import StaticFiles
import json
from google import genai

# 載入環境變數
load_dotenv()

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Odyssey Backend API",
    description="AI 學習遊戲平台後端 API",
    version="1.0.0"
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
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# 模擬資料庫 (實際應用中應該使用真實資料庫)
users_db = {}
fish_images_db = [
    {"image_id": "img_fish_001", "url": "https://cdn.your-game.com/fishes/fish_a.png"},
    {"image_id": "img_fish_002", "url": "https://cdn.your-game.com/fishes/fish_b.png"},
    {"image_id": "img_fish_003", "url": "https://cdn.your-game.com/fishes/fish_c.png"},
    {"image_id": "img_fish_004", "url": "https://cdn.your-game.com/fishes/fish_d.png"},
    {"image_id": "img_fish_005", "url": "https://cdn.your-game.com/fishes/fish_e.png"},
]

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

class ImageLabel(BaseModel):
    image_id: str = Field(..., example="img_fish_001", description="圖片 ID")
    classification: str = Field(..., example="arowana", description="分類結果")

class SubmitLabelsRequest(BaseModel):
    labels: List[ImageLabel] = Field(..., example=[
        {"image_id": "img_fish_001", "classification": "arowana"},
        {"image_id": "img_fish_002", "classification": "tilapia"},
        {"image_id": "img_fish_003", "classification": "arowana"}
    ], description="標註結果列表")

class SubmitLabelsResponse(BaseModel):
    message: str = Field(..., example="魔法魚鉤學習完成！它現在能更好地分辨魚了！")
    accuracy: float = Field(..., example=0.95, description="訓練準確率，範圍 0.8-0.98")

class IdentifyFishResponse(BaseModel):
    fish_type: str = Field(..., example="arowana", description="魚類類型：arowana(銀龍魚) 或 tilapia(吳郭魚)")
    image_url: str = Field(..., example="https://cdn.your-game.com/fishes/arowana_1.png")
    decision: str = Field(..., example="keep", description="決定：keep(保留) 或 release(放生)")
    value_gained: int = Field(..., example=1000, description="獲得的價值")
    message: str = Field(..., example="是銀龍魚！AI 魔法魚鉤決定留下牠！")

# Pydantic 模型 - 模組三：國王的厭食症
class GenerateRecipeTextRequest(BaseModel):
    prompt: str = Field(..., description="食譜生成提示詞")

class GenerateRecipeTextResponse(BaseModel):
    recipe_text: str = Field(..., example="✨ 魔法森林的彩虹水果沙拉 ✨\n\n這道菜就像森林精靈的魔法盛宴！我們需要：\n\n🍎 紅蘋果 - 切成小星星形狀\n🍊 橘子 - 剝成小月牙\n🍇 葡萄 - 像珍珠一樣閃亮\n🥝 奇異果 - 切成小圓片\n\n調味魔法：\n🍯 蜂蜜 - 森林的甜蜜精華\n🍋 檸檬汁 - 清新的魔法\n🌿 薄荷葉 - 森林的香氣\n\n做法：\n1. 將所有水果洗淨，切成可愛的形狀\n2. 輕輕混合，不要破壞水果的完整性\n3. 淋上蜂蜜和檸檬汁的魔法組合\n4. 最後點綴新鮮的薄荷葉\n\n這道沙拉不僅美味，還能讓國王想起森林的快樂時光！🌈")

class GenerateRecipeImageRequest(BaseModel):
    prompt: str = Field(..., description="圖片生成提示詞")

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

# Pydantic 模型 - 食物圖片分析相關
class FoodImageAnalysisRequest(BaseModel):
    image_hash: str = Field(..., description="圖片6位數hash", example="abc123")
    prompt: Optional[str] = Field(default="請分析這張食物圖片，給出詳細的評語，包括：1. 食物名稱和類型 2. 外觀描述顏色與味道評估 3. 營養價值評估 4. 整體評價", description="分析提示詞，如果不填寫將使用預設的食物分析提示詞")

class FoodImageAnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Gemini 對食物的評語")
    model_used: str = Field(..., description="使用的模型")
    image_hash: str = Field(..., description="分析的圖片hash")
    prompt: str = Field(..., description="使用的提示詞")


# 通用回應格式
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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="無效的認證憑證",
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
        timestamp = str(datetime.now().timestamp())
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
        "version": "1.0.0",
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
    if user_data.username in users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="使用者名稱已被註冊。"
        )
    
    # 加密密碼
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
    
    # 建立使用者
    user_id = str(uuid.uuid4())
    users_db[user_data.username] = {
        "id": user_id,
        "username": user_data.username,
        "hashed_password": hashed_password,
        "money": 500
    }
    
    return {
        "status": "success",
        "data": {
            "message": "註冊成功！",
            "user": {
                "id": user_id,
                "username": user_data.username
            }
        }
    }

@app.post("/auth/login")
async def login(user_data: UserLogin):
    """使用者登入"""
    if user_data.username not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="帳號或密碼錯誤。"
        )
    
    user = users_db[user_data.username]
    if not bcrypt.checkpw(user_data.password.encode('utf-8'), user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="帳號或密碼錯誤。"
        )
    
    # 建立 JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_data.username}, expires_delta=access_token_expires
    )
    
    return {
        "status": "success",
        "data": {
            "message": "登入成功！",
            "access_token": access_token,
            "token_type": "bearer"
        }
    }

@app.get("/users/me", response_model=SuccessResponse)
async def get_current_user(username: str = Depends(verify_token)):
    """獲取當前使用者資訊"""
    user = users_db[username]
    return SuccessResponse(data={
        "user": {
            "id": user["id"],
            "username": user["username"],
            "money": user["money"]
        }
    })

@app.post("/llama4", response_model=LLMChatResponse)
# TODO: 需要驗證 token
# username: str = Depends(verify_token),
async def chat_with_llm(
    request: LLMChatRequest,      
    api_key: str = Depends(get_api_key)
):
    """與 LLM 進行對話，使用llama4模型，temp 0.4，max_tokens 1000，使用system_prompt作為系統提示詞（如果不填寫將使用預設的智識庫設定），使用question作為使用者問題"""
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
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 1000
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

@app.get("/module2/training-images", response_model=SuccessResponse)
async def get_training_images(username: str = Depends(verify_token)):
    """獲取待分類的魚類圖片"""
    # 隨機選擇 3-5 張圖片
    num_images = random.randint(3, 5)
    selected_images = random.sample(fish_images_db, min(num_images, len(fish_images_db)))
    
    return SuccessResponse(data={
        "images": selected_images
    })

@app.post("/module2/submit-labels", response_model=SubmitLabelsResponse)
async def submit_labels(
    request: SubmitLabelsRequest,
    username: str = Depends(verify_token)
):
    """提交標註結果"""
    # 模擬訓練準確率 (80-98%)
    accuracy = random.uniform(0.8, 0.98)
    
    return SubmitLabelsResponse(
        message="魔法魚鉤學習完成！它現在能更好地分辨魚了！",
        accuracy=round(accuracy, 2)
    )

@app.post("/module2/identify-fish", response_model=IdentifyFishResponse)
async def identify_fish(username: str = Depends(verify_token)):
    """進行 AI 辨識"""
    # 隨機決定釣到的魚類
    fish_types = ["arowana", "tilapia"]
    fish_type = random.choice(fish_types)
    
    if fish_type == "arowana":
        # 銀龍魚 - 保留
        return IdentifyFishResponse(
            fish_type="arowana",
            image_url="https://cdn.your-game.com/fishes/arowana_1.png",
            decision="keep",
            value_gained=1000,
            message="是銀龍魚！AI 魔法魚鉤決定留下牠！"
        )
    else:
        # 吳郭魚 - 放生
        return IdentifyFishResponse(
            fish_type="tilapia",
            image_url="https://cdn.your-game.com/fishes/tilapia_1.png",
            decision="release",
            value_gained=0,
            message="是吳郭魚！AI 魔法魚鉤將牠放回去了。"
        )

# 3. 模組三：國王的厭食症 (Generative AI)

# TODO: 需要驗證 token
# username: str = Depends(verify_token),
@app.post("/module3/generate-recipe-text", response_model=GenerateRecipeTextResponse)
async def generate_recipe_text(
    request: GenerateRecipeTextRequest,
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
            "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
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


# TODO: 需要驗證 token
# username: str = Depends(verify_token),
@app.post("/module3/generate-recipe-image", response_model=GenerateRecipeImageResponse)
async def generate_recipe_image(
    request: GenerateRecipeImageRequest,
    http_request: Request,
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
        generation_time = datetime.now()
        
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

@app.post("/module3/analyze-food-image", response_model=FoodImageAnalysisResponse)
async def analyze_food_image(
    request: FoodImageAnalysisRequest,
):
    """使用 Gemini 分析食物圖片並給出評語"""
    try:
        # 檢查圖片是否存在
        image_path = os.path.join(STATIC_IMAGES_DIR, f"{request.image_hash}.png")
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail=f"找不到圖片: {request.image_hash}.png"
            )
        
        # 讀取圖片文件
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        # 使用 Gemini 分析圖片
        from google.genai import types
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/png',
                ),
                request.prompt
            ]
        )
        
        return FoodImageAnalysisResponse(
            analysis=response.text,
            model_used="gemini-2.5-flash",
            image_hash=request.image_hash,
            prompt=request.prompt
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"圖片文件不存在: {request.image_hash}.png"
        )
    except Exception as e:
        logger.error(f"Gemini API 錯誤: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Gemini API 錯誤: {str(e)}"
        )


@app.get("/system-prompts")
async def get_system_prompts():
    """取得可用的 system prompts 列表"""
    return {
        "status": "success",
        "data": {
            "prompts": system_prompts
        }
    }

@app.get("/models")
async def list_models():
    """取得可用模型列表 - 向後相容"""
    return {
        "models": [
            {
                "id": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
                "description": "Llama-4-Maverick-17B-128E-Instruct-FP8"
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
