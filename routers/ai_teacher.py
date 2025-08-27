import json
import os
from typing import Dict, Any, List

from fastapi import APIRouter
from pydantic import BaseModel, Field
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from google import genai


class ImageData(BaseModel):
    data: str = Field(..., description="base64編碼的圖片數據")
    mimeType: str = Field(..., description="圖片MIME類型")


class OCRRequest(BaseModel):
    images: List[ImageData]
    system_prompt: str


class AIWritingTeacherResponse(BaseModel):
    success: bool
    data: Dict[str, Any] | None
    error: Dict[str, Any] | None


class TopicInfo(BaseModel):
    topic: str
    requirements: str
    score: int


class CorrectionRequest(BaseModel):
    essay_text: str
    topic_info: TopicInfo
    system_prompt: str


class HandoutRequest(BaseModel):
    correction_data: Dict[str, Any]
    system_prompt: str


class PracticeRequest(BaseModel):
    correction_data: Dict[str, Any]
    num_questions: int
    num_versions: int
    system_prompt: str


router = APIRouter(prefix="/api", tags=["ai_teacher"])


@router.post("/ocr/topic", response_model=AIWritingTeacherResponse)
async def ocr_topic(request: OCRRequest):
    try:
        if not request.images:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INVALID_REQUEST", "message": "請提供圖片", "details": "images 欄位不能為空"})
        from google.genai import types
        gemini_api_key = os.getenv("AI_ENGLISH_WRITING_TEACHER")
        if not gemini_api_key:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INTERNAL_ERROR", "message": "Gemini API Key 未設定", "details": "請設定 AI_ENGLISH_WRITING_TEACHER 環境變數"})
        client = genai.Client(api_key=gemini_api_key)
        image_data = request.images[0]
        import base64
        try:
            image_bytes = base64.b64decode(image_data.data)
        except Exception as e:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INVALID_IMAGE_FORMAT", "message": "圖片格式錯誤", "details": f"base64 解碼失敗: {str(e)}"})
        response = client.models.generate_content(model="gemini-2.5-flash", contents=[types.Part.from_bytes(data=image_bytes, mime_type=image_data.mimeType), request.system_prompt])
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
            result = json.loads(response_text)
        return AIWritingTeacherResponse(success=True, data=result, error=None)
    except json.JSONDecodeError:
        result = {"topic": "圖片辨識完成，但格式解析失敗", "requirements": "請檢查 AI 回應格式", "score": "無法確定"}
        return AIWritingTeacherResponse(success=True, data=result, error=None)
    except Exception as e:
        return AIWritingTeacherResponse(success=False, data=None, error={"code": "OCR_FAILED", "message": "圖片辨識失敗", "details": str(e)})


@router.post("/ocr/essay", response_model=AIWritingTeacherResponse)
async def ocr_essay(request: OCRRequest):
    try:
        if not request.images:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INVALID_REQUEST", "message": "請提供圖片", "details": "images 欄位不能為空"})
        from google.genai import types
        gemini_api_key = os.getenv("AI_ENGLISH_WRITING_TEACHER")
        if not gemini_api_key:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INTERNAL_ERROR", "message": "Gemini API Key 未設定", "details": "請設定 AI_ENGLISH_WRITING_TEACHER 環境變數"})
        client = genai.Client(api_key=gemini_api_key)
        all_texts = []
        import base64
        for image_data in request.images:
            try:
                image_bytes = base64.b64decode(image_data.data)
            except Exception as e:
                return AIWritingTeacherResponse(success=False, data=None, error={"code": "INVALID_IMAGE_FORMAT", "message": "圖片格式錯誤", "details": f"base64 解碼失敗: {str(e)}"})
            response = client.models.generate_content(model="gemini-2.5-flash", contents=[types.Part.from_bytes(data=image_bytes, mime_type=image_data.mimeType), request.system_prompt])
            all_texts.append(response.text)
        combined_text = " ".join(all_texts)
        return AIWritingTeacherResponse(success=True, data={"recognized_text": combined_text}, error=None)
    except Exception as e:
        return AIWritingTeacherResponse(success=False, data=None, error={"code": "OCR_FAILED", "message": "手寫作文辨識失敗", "details": str(e)})


@router.post("/ai/correction", response_model=AIWritingTeacherResponse)
async def correct_essay(request: CorrectionRequest):
    try:
        import httpx
        nchc_api_key = os.getenv("NCHC_API_KEY")
        if not nchc_api_key:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INTERNAL_ERROR", "message": "NCHC API Key 未設定", "details": "請設定 NCHC_API_KEY 環境變數"})
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": f"題目資訊：{request.topic_info.topic}\n要求：{request.topic_info.requirements}\n分數：{request.topic_info.score}\n\n學生作文：{request.essay_text}"},
        ]
        chat_request = {"model": "gpt-oss-120b", "messages": messages, "temperature": 0.3, "max_tokens": 20000}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://portal.genai.nchc.org.tw/api/v1/chat/completions",
                headers={"x-api-key": nchc_api_key, "Content-Type": "application/json"},
                json=chat_request,
                timeout=60.0,
            )
        if response.status_code != 200:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "AI 生成失敗", "details": f"NCHC API 錯誤: {response.text}"})
        result = response.json()
        if result.get("choices") and len(result["choices"]) > 0:
            ai_response = result["choices"][0]["message"]["content"]
        else:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "AI 回應格式錯誤", "details": "無法提取 AI 回應內容"})
        try:
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
            return AIWritingTeacherResponse(success=True, data=correction_data, error=None)
        except json.JSONDecodeError:
            return AIWritingTeacherResponse(success=True, data={"raw_response": ai_response}, error=None)
    except Exception as e:
        return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "作文批改失敗", "details": str(e)})


@router.post("/ai/handout", response_model=AIWritingTeacherResponse)
async def generate_handout(request: HandoutRequest):
    try:
        import httpx
        nchc_api_key = os.getenv("NCHC_API_KEY")
        if not nchc_api_key:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INTERNAL_ERROR", "message": "NCHC API Key 未設定", "details": "請設定 NCHC_API_KEY 環境變數"})
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": f"批改資料：{json.dumps(jsonable_encoder(request.correction_data), ensure_ascii=False)}"},
        ]
        chat_request = {"model": "gpt-oss-120b", "messages": messages, "temperature": 0.4, "max_tokens": 20000}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://portal.genai.nchc.org.tw/api/v1/chat/completions",
                headers={"x-api-key": nchc_api_key, "Content-Type": "application/json"},
                json=chat_request,
                timeout=60.0,
            )
        if response.status_code != 200:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "講義生成失敗", "details": f"NCHC API 錯誤: {response.text}"})
        result = response.json()
        if result.get("choices") and len(result["choices"]) > 0:
            handout_content = result["choices"][0]["message"]["content"]
        else:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "講義生成失敗", "details": "無法提取 AI 回應內容"})
        return AIWritingTeacherResponse(success=True, data={"handout_content": handout_content}, error=None)
    except Exception as e:
        return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "講義生成失敗", "details": str(e)})


@router.post("/ai/practice", response_model=AIWritingTeacherResponse)
async def generate_practice_questions(request: PracticeRequest):
    try:
        import httpx
        nchc_api_key = os.getenv("NCHC_API_KEY")
        if not nchc_api_key:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "INTERNAL_ERROR", "message": "NCHC API Key 未設定", "details": "請設定 NCHC_API_KEY 環境變數"})
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": f"批改資料：{json.dumps(jsonable_encoder(request.correction_data), ensure_ascii=False)}\n題目數量：{request.num_questions}\n版本數量：{request.num_versions}"},
        ]
        chat_request = {"model": "gpt-oss-120b", "messages": messages, "temperature": 0.5, "max_tokens": 20000}
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://portal.genai.nchc.org.tw/api/v1/chat/completions",
                headers={"x-api-key": nchc_api_key, "Content-Type": "application/json"},
                json=chat_request,
                timeout=60.0,
            )
        if response.status_code != 200:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "練習題生成失敗", "details": f"NCHC API 錯誤: {response.text}"})
        result = response.json()
        if result.get("choices") and len(result["choices"]) > 0:
            practice_content = result["choices"][0]["message"]["content"]
        else:
            return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "練習題生成失敗", "details": "無法提取 AI 回應內容"})
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
        return AIWritingTeacherResponse(success=True, data=practice_data, error=None)
    except json.JSONDecodeError:
        return AIWritingTeacherResponse(success=True, data={"raw_response": "AI 回應的 JSON 解析失敗"}, error=None)
    except Exception as e:
        return AIWritingTeacherResponse(success=False, data=None, error={"code": "AI_GENERATION_FAILED", "message": "練習題生成失敗", "details": str(e)})


