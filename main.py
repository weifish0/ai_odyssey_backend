from fastapi import FastAPI, HTTPException, Depends, status
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
    id: str
    username: str
    money: int = 500

# Pydantic 模型 - 模組二：魚池的財富密碼
class AskPetRequest(BaseModel):
    question: str = Field(..., description="向 AI 寵物提問")

class AskPetResponse(BaseModel):
    answer: str

class ImageLabel(BaseModel):
    image_id: str = Field(..., description="圖片 ID")
    classification: str = Field(..., description="分類結果")

class SubmitLabelsRequest(BaseModel):
    labels: List[ImageLabel] = Field(..., description="標註結果列表")

class SubmitLabelsResponse(BaseModel):
    message: str
    accuracy: float

class IdentifyFishResponse(BaseModel):
    fish_type: str
    image_url: str
    decision: str
    value_gained: int
    message: str

# Pydantic 模型 - 模組三：國王的厭食症
class GenerateRecipeTextRequest(BaseModel):
    prompt: str = Field(..., description="食譜生成提示詞")

class GenerateRecipeTextResponse(BaseModel):
    recipe_text: str

class GenerateRecipeImageRequest(BaseModel):
    prompt: str = Field(..., description="圖片生成提示詞")

class GenerateRecipeImageResponse(BaseModel):
    image_url: str

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

def validate_prompt_safety(prompt: str) -> bool:
    """驗證提示詞的安全性"""
    # 定義可能觸發安全系統的詞彙
    unsafe_words = [
        "blood", "gore", "violence", "weapon", "drug", "alcohol", "nude", "naked",
        "sexual", "explicit", "offensive", "hate", "discrimination", "political",
        "controversial", "inappropriate", "unsafe", "dangerous", "harmful"
    ]
    
    # 檢查是否包含不安全的詞彙
    prompt_lower = prompt.lower()
    for word in unsafe_words:
        if word in prompt_lower:
            return False
    
    return True

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

@app.get("/users/me")
async def get_current_user(username: str = Depends(verify_token)):
    """獲取當前使用者資訊"""
    user = users_db[username]
    return {
        "status": "success",
        "data": {
            "user": {
                "id": user["id"],
                "username": user["username"],
                "money": user["money"]
            }
        }
    }

# 2. 模組二：魚池的財富密碼 (Image Recognition)

@app.post("/module2/ask-pet")

# TODO: 需要驗證 token
# username: str = Depends(verify_token),
async def ask_pet(
    request: AskPetRequest,      
    api_key: str = Depends(get_api_key)
):
    """詢問 AI 寵物"""
    try:
        # 建立訊息列表
        messages = [
            {
                "role": "system",
                "content": "你是一個專門研究魚類的 AI 助手。請用中文繁體回答關於魚類特徵的問題，回答要詳細且準確。"
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
            "temperature": 0.7,
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
            
            return {
                "status": "success",
                "data": {
                    "answer": answer
                }
            }
            
    except httpx.TimeoutException:
        logger.error("請求超時")
        raise HTTPException(status_code=408, detail="請求超時")
    except httpx.RequestError as e:
        logger.error(f"請求錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"請求錯誤: {str(e)}")
    except Exception as e:
        logger.error(f"未預期錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"未預期錯誤: {str(e)}")

@app.get("/module2/training-images")
async def get_training_images(username: str = Depends(verify_token)):
    """獲取待分類的魚類圖片"""
    # 隨機選擇 3-5 張圖片
    num_images = random.randint(3, 5)
    selected_images = random.sample(fish_images_db, min(num_images, len(fish_images_db)))
    
    return {
        "status": "success",
        "data": {
            "images": selected_images
        }
    }

@app.post("/module2/submit-labels")
async def submit_labels(
    request: SubmitLabelsRequest,
    username: str = Depends(verify_token)
):
    """提交標註結果"""
    # 模擬訓練準確率 (80-98%)
    accuracy = random.uniform(0.8, 0.98)
    
    return {
        "status": "success",
        "data": {
            "message": "魔法魚鉤學習完成！它現在能更好地分辨魚了！",
            "accuracy": round(accuracy, 2)
        }
    }

@app.post("/module2/identify-fish")
async def identify_fish(username: str = Depends(verify_token)):
    """進行 AI 辨識"""
    # 隨機決定釣到的魚類
    fish_types = ["arowana", "tilapia"]
    fish_type = random.choice(fish_types)
    
    if fish_type == "arowana":
        # 銀龍魚 - 保留
        return {
            "status": "success",
            "data": {
                "fish_type": "arowana",
                "image_url": "https://cdn.your-game.com/fishes/arowana_1.png",
                "decision": "keep",
                "value_gained": 1000,
                "message": "是銀龍魚！AI 魔法魚鉤決定留下牠！"
            }
        }
    else:
        # 吳郭魚 - 放生
        return {
            "status": "success",
            "data": {
                "fish_type": "tilapia",
                "image_url": "https://cdn.your-game.com/fishes/tilapia_1.png",
                "decision": "release",
                "value_gained": 0,
                "message": "是吳郭魚！AI 魔法魚鉤將牠放回去了。"
            }
        }

# 3. 模組三：國王的厭食症 (Generative AI)

# TODO: 需要驗證 token
# username: str = Depends(verify_token),
@app.post("/module3/generate-recipe-text")
async def generate_recipe_text(
    request: GenerateRecipeTextRequest,
    api_key: str = Depends(get_api_key)
):
    """創造食譜描述"""
    try:
        # 建立訊息列表
        messages = [
            {
                "role": "system",
                "content": "你是一個專業的廚師和美食作家。請根據使用者的要求，創造出富有創意和想像力的食譜描述。描述要生動有趣，符合童話故事的風格。請使用中文繁體回答。"
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
            
            return {
                "status": "success",
                "data": {
                    "recipe_text": recipe_text
                }
            }
            
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
@app.post("/module3/generate-recipe-image")
async def generate_recipe_image(
    request: GenerateRecipeImageRequest,
    client: OpenAI = Depends(get_openai_client)
):
    """視覺化菜色"""
    try:
        # 驗證提示詞安全性
        if not validate_prompt_safety(request.prompt):
            raise HTTPException(
                status_code=400,
                detail="提示詞包含不安全的內容，請使用更適當的描述。建議：專注於食物的正面描述，如食材、烹飪方式、菜色外觀等。"
            )
        
        # 優化提示詞，增加安全性和具體性
        enhanced_prompt = f"Beautiful, appetizing food photography: {request.prompt}. High quality, professional food image, safe for all audiences, no inappropriate content."
        
        # 使用 DALL-E 生成圖片
        response = client.images.generate(
            model="dall-e-2",
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
        local_image_url = f"/static/images/{filename}"
        
        return {
            "status": "success",
            "data": {
                "image_url": local_image_url,
                "original_url": image_url,
                "filename": filename
            }
        }
        
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

# 保留原有的 NCHC API 端點 (向後相容)
@app.post("/chat/completions")
async def chat_completions(
    request: Dict[str, Any],
    api_key: str = Depends(get_api_key)
):
    """完整的 Chat Completions API 端點 - 向後相容"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NCHC_API_BASE_URL}/chat/completions",
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json=request,
                timeout=60.0
            )
            
            if response.status_code != 200:
                logger.error(f"NCHC API 錯誤: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"NCHC API 錯誤: {response.text}"
                )
            
            return response.json()
            
    except httpx.TimeoutException:
        logger.error("請求超時")
        raise HTTPException(status_code=408, detail="請求超時")
    except httpx.RequestError as e:
        logger.error(f"請求錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"請求錯誤: {str(e)}")
    except Exception as e:
        logger.error(f"未預期錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"未預期錯誤: {str(e)}")

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
