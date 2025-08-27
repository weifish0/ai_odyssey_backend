from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
import httpx

from core.deps import verify_token
from core.config import NCHC_API_BASE_URL, NCHC_API_KEY, SYSTEM_PROMPTS


class LLMChatRequest(BaseModel):
    question: str = Field(..., example="什麼是影像辨識")
    system_prompt: str = Field(
        default=SYSTEM_PROMPTS[0]["content"] if SYSTEM_PROMPTS else "請輸入系統提示詞",
        description="系統提示詞"
    )


class LLMChatResponse(BaseModel):
    answer: str


router = APIRouter(tags=["llm"])


def _get_nchc_api_key() -> str:
    if not NCHC_API_KEY:
        raise HTTPException(status_code=500, detail="API Key 未設定，請設定 NCHC_API_KEY 環境變數")
    return NCHC_API_KEY


@router.post("/llmchat", response_model=LLMChatResponse)
async def chat_with_llm(request: LLMChatRequest, username: str = Depends(verify_token), api_key: str = Depends(_get_nchc_api_key)):
    """與 LLM 進行對話，使用 gpt-oss-120b 模型，temp 0.4，max_tokens 2000，使用system_prompt作為系統提示詞（如果不填寫將使用預設的知識機器人設定），使用question作為使用者問題"""
    try:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.question},
        ]
        chat_request = {"model": "gpt-oss-120b", "messages": messages, "temperature": 0.4, "max_tokens": 2000}
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
        answer = result["choices"][0]["message"]["content"] if result.get("choices") else "抱歉，我無法回答這個問題。"
        return LLMChatResponse(answer=answer)
    except httpx.TimeoutException:
        raise HTTPException(status_code=408, detail="請求超時")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"未預期錯誤: {str(e)}")


