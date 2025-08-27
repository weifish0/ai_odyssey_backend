from fastapi import APIRouter
from core.config import NCHC_API_KEY, SYSTEM_PROMPTS
from ml_state import is_mobilenet_ready, get_mobilenet

router = APIRouter()


@router.get("/")
async def root():
    return {
        "message": "AI Odyssey Backend API",
        "version": "2.0.0",
        "docs": "/docs"
    }


@router.get("/health")
async def health_check():
    return {"status": "healthy", "api_key_configured": bool(NCHC_API_KEY)}


@router.get("/system-prompts")
async def get_system_prompts():
    return {"prompts": SYSTEM_PROMPTS}


@router.get("/models")
async def list_models():
    return {
        "models": [
            "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "gpt-oss-120b",
            "gemini-2.5-flash",
            "dall-e-2",
            "gemini-embedding-001",
        ]
    }


@router.get("/model-status")
async def check_model_status():
    """檢查 MobileNet 模型狀態"""
    return {
        "mobilenet_ready": is_mobilenet_ready(),
        "mobilenet_instance": str(get_mobilenet()) if get_mobilenet() else None
    }


