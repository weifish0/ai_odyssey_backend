import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime

from core.deps import verify_token
from core.utils import STATIC_IMAGES_DIR
from google import genai
import asyncio


class FoodImageAnalysisRequest(BaseModel):
    image_hash: str = Field(..., description="圖片12位數hash")
    dish_expect: Optional[str] = Field(default="我期待這是一道色香味俱全、營養均衡、擺盤精美的美食")


class FoodImageAnalysisResponse(BaseModel):
    analysis: str = Field(..., description="Gemini 對食物的詳細評語")
    score: int = Field(..., description="食物評分 (0-100分)", ge=0 , le=100)
    model_used: str = Field(..., description="使用的模型")
    image_path: str = Field(..., description="分析的圖片路徑")
    prompt: str = Field(..., description="使用的提示詞")
    dish_expect: str = Field(..., description="顧客對這道菜的期待描述")
    full_response_text: str = Field(..., description="Gemini 的完整回應文字")


router = APIRouter(prefix="/module1", tags=["module1"])


@router.post("/analyze-food-image", response_model=FoodImageAnalysisResponse)
async def analyze_food_image(request: FoodImageAnalysisRequest, username: str = Depends(verify_token)):
    try:
        image_path = os.path.join(STATIC_IMAGES_DIR, f"{request.image_hash}")
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"找不到圖片: {image_path}")

        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        from google.genai import types
        chef_prompt = f"""你是一位經驗豐富、要求嚴格的餐廳主廚。請以專業廚師的眼光，嚴格分析這張食物圖片。
對食物的期待：{request.dish_expect}
請按照以下固定格式回覆：
SCORE: [0-100分]
(評分標準：0-59分=不合格，60-70分=及格，71-85分=良好，86-95分=優秀，96-100分=完美)

ANALYSIS:
[請以嚴格廚師的角度，詳細評語(不要超過100字，不要條列式回答，不要使用markdown格式語法如"**")，包括：
1. 食物名稱和類型識別
2. 外觀、顏色、擺盤的專業評估
3. 與顧客期待的對比分析
4. 改進建議和專業點評]

請確保回覆格式完全按照上述模板，SCORE 必須是 0-100 的整數分數。"""

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=[types.Part.from_bytes(data=image_bytes, mime_type='image/png'), chef_prompt],
        )
        response_text = response.text

        score = 50
        analysis = response_text
        try:
            if "SCORE:" in response_text:
                score_line = response_text.split("SCORE:")[1].split("\n")[0].strip()
                import re
                score_match = re.search(r'\d+', score_line)
                if score_match:
                    score = max(0, min(100, int(score_match.group())))
            if "ANALYSIS:" in response_text:
                parts = response_text.split("ANALYSIS:")
                analysis = parts[1].strip() if len(parts) > 1 else response_text
            else:
                lines = response_text.split('\n')
                analysis_lines = [l for l in lines if not l.strip().startswith('SCORE:') and l.strip()]
                if analysis_lines:
                    analysis = '\n'.join(analysis_lines).strip()
        except Exception:
            score = 50
            analysis = response_text

        return FoodImageAnalysisResponse(
            analysis=analysis,
            score=score,
            model_used="gemini-2.5-flash",
            full_response_text=response_text,
            image_path=image_path,
            prompt=chef_prompt,
            dish_expect=request.dish_expect,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"圖片文件不存在: {image_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"圖片分析錯誤: {str(e)}")


