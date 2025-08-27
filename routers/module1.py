from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from core.deps import verify_token
from core.config import NCHC_API_BASE_URL
from core.utils import download_and_save_image

import httpx
from openai import OpenAI
import asyncio


class GenerateRecipeTextRequest(BaseModel):
    prompt: str = Field(..., description="食譜生成提示詞")


class GenerateRecipeTextResponse(BaseModel):
    recipe_text: str


class GenerateRecipeImageRequest(BaseModel):
    prompt: str = Field(..., description="圖片生成提示詞", example="宮保雞丁")


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


class GenerateCustomImageRequest(BaseModel):
    prompt: str = Field(..., example="李久恩穿著紅色褲褲，手拿十字架，冷傲退基佬，寫實畫風，高清", description="客製化圖片生成")
    


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


router = APIRouter(prefix="/module1", tags=["module1"])


@router.post("/generate-recipe-text", response_model=GenerateRecipeTextResponse)
async def generate_recipe_text(
    request: GenerateRecipeTextRequest,
    username: str = Depends(verify_token),
    api_key: str = Depends(lambda: __import__('os').getenv('NCHC_API_KEY')),
):
    try:
        messages = [
            {"role": "system", "content": "你是一個專業的廚師和美食作家。請根據使用者的要求，創造出富有創意和想像力的食譜描述。描述要生動有趣，符合童話故事的風格。"},
            {"role": "user", "content": request.prompt},
        ]

        chat_request = {"model": "gpt-oss-120b", "messages": messages, "temperature": 0.8, "max_tokens": 1500}

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{NCHC_API_BASE_URL}/chat/completions",
                headers={"x-api-key": api_key, "Content-Type": "application/json"},
                json=chat_request,
                timeout=60.0,
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"NCHC API 錯誤: {response.text}")

        result = response.json()
        recipe_text = result["choices"][0]["message"]["content"] if result.get("choices") else "抱歉，我無法生成食譜描述。"
        return GenerateRecipeTextResponse(recipe_text=recipe_text)
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="請求超時")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"未預期錯誤: {str(e)}")


@router.post("/generate-recipe-image", response_model=GenerateRecipeImageResponse)
async def generate_recipe_image(
    request: GenerateRecipeImageRequest,
    http_request: Request,
    username: str = Depends(verify_token),
    client: OpenAI = Depends(lambda: OpenAI(api_key=__import__('os').getenv('OPENAI_API_KEY'))),
):
    try:
        enhanced_prompt = f"Beautiful, appetizing food photography: {request.prompt}. High quality food image."
        response = await asyncio.to_thread(
            client.images.generate,
            model="dall-e-3",
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024",
        )

        if response.data and len(response.data) > 0:
            image_url = response.data[0].url
        else:
            raise HTTPException(status_code=500, detail="圖片生成失敗")

        filename = await download_and_save_image(image_url, request.prompt)
        base_url = str(http_request.base_url).rstrip('/')
        local_image_url = f"{base_url}/static/images/{filename}"
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
                "full_url": local_image_url,
            },
        )
    except Exception as e:
        error_msg = str(e)
        if "content_policy_violation" in error_msg:
            raise HTTPException(status_code=400, detail="提示詞內容違反安全政策，請使用更適當的描述。")
        elif "safety_system" in error_msg:
            raise HTTPException(status_code=400, detail="提示詞被安全系統拒絕，請避免使用可能不當的詞彙。")
        elif "400" in error_msg:
            raise HTTPException(status_code=400, detail="請求格式錯誤，請檢查提示詞內容。建議：使用清晰、具體的食物描述。")
        else:
            raise HTTPException(status_code=500, detail=f"圖片生成錯誤: {error_msg}")


@router.post("/generate-custom-image", response_model=GenerateCustomImageResponse)
async def generate_custom_image(
    request: GenerateCustomImageRequest,
    http_request: Request,
    username: str = Depends(verify_token),
    client: OpenAI = Depends(lambda: OpenAI(api_key=__import__('os').getenv('OPENAI_API_KEY'))),
):
    try:
        response = await asyncio.to_thread(
            client.images.generate,
            model="dall-e-3",
            prompt=request.prompt,
            n=1,
            size="1024x1024",
        )
        if response.data and len(response.data) > 0:
            image_url = response.data[0].url
        else:
            raise HTTPException(status_code=500, detail="圖片生成失敗")

        filename = await download_and_save_image(image_url, request.prompt)
        base_url = str(http_request.base_url).rstrip('/')
        local_image_url = f"{base_url}/static/images/{filename}"
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
                "full_url": local_image_url,
            },
        )
    except Exception as e:
        error_msg = str(e)
        if "content_policy_violation" in error_msg:
            raise HTTPException(status_code=400, detail="提示詞內容違反安全政策，請使用更適當的描述。")
        elif "safety_system" in error_msg:
            raise HTTPException(status_code=400, detail="提示詞被安全系統拒絕，請避免使用可能不當的詞彙。")
        elif "400" in error_msg:
            raise HTTPException(status_code=400, detail="請求格式錯誤，請檢查提示詞內容。建議：使用清晰、具體的食物描述。")
        else:
            raise HTTPException(status_code=500, detail=f"圖片生成錯誤: {error_msg}")


