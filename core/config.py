import os
import json
from dotenv import load_dotenv

load_dotenv()

# Security / JWT
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "300"))

# External APIs
NCHC_API_BASE_URL = os.getenv("NCHC_API_BASE_URL", "https://portal.genai.nchc.org.tw/api/v1")
NCHC_API_KEY = os.getenv("NCHC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")  # 備用 API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AI_ENGLISH_WRITING_TEACHER_KEY = os.getenv("AI_ENGLISH_WRITING_TEACHER")


def load_system_prompts():
    try:
        with open("system_prompts.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("prompts", [])
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


SYSTEM_PROMPTS = load_system_prompts()


