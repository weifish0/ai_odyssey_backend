import hashlib
import os
import random
from datetime import datetime, timezone

import aiofiles
import httpx

from core.config import SYSTEM_PROMPTS, NCHC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY

# 路徑由 main.py 確保目錄存在
STATIC_IMAGES_DIR = "static/images"


async def download_and_save_image(image_url: str, prompt: str) -> str:
    try:
        timestamp = str(datetime.now(timezone.utc).timestamp())
        random_salt = str(random.randint(1000, 9999))
        hash_input = f"{prompt}{timestamp}{random_salt}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        filename = f"{file_hash}.png"
        filepath = os.path.join(STATIC_IMAGES_DIR, filename)

        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(response.content)
        return filename
    except Exception as exc:
        raise RuntimeError(f"下載圖片失敗: {exc}")


