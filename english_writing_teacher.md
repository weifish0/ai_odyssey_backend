## gemini 超過限制可能遇到問題
ERROR:main:題目圖片辨識失敗: 503 UNAVAILABLE. {'error': {'code': 503, 'message': 'The model is overloaded. Please try again later.', 'status': 'UNAVAILABLE'}}

是服務端暫時過載。HTTP 503 UNAVAILABLE 代表 Gemini 模型目前忙碌或暫時不可用，跟你程式邏輯無關。通常過幾秒～幾十秒重試就會恢復。

建議處理方式（擇一或併用）:
- 加入指數退避重試
- 降低併發與圖片大小
- 暫時切到其他可用模型（如 gemini-1.5-flash）
- 把錯誤透傳為 503，回覆前端「稍後重試」

可直接加一個重試小工具，替換你現在對 Gemini 的呼叫：

```python
import asyncio
from functools import partial
from fastapi import HTTPException

async def gemini_generate_with_retry(client, contents, model="gemini-2.5-flash", retries=3, base_delay=1.5):
    fn = partial(client.models.generate_content, model=model, contents=contents)
    last_err = None
    for i in range(retries):
        try:
            return await asyncio.to_thread(fn)
        except Exception as e:
            msg = str(e)
            if "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg:
                last_err = e
                await asyncio.sleep(base_delay * (2 ** i))
                continue
            raise
    raise HTTPException(status_code=503, detail="Gemini 暫時過載，請稍後再試")
```

在 `/api/ocr/topic`、`/api/ocr/essay` 內把：
- `response = client.models.generate_content(...)`
改成：
- `response = await gemini_generate_with_retry(client, [...], model="gemini-2.5-flash")`

如仍常見 503，可把 `retries` 提高到 5，或把 `model` 改為 `"gemini-1.5-flash"`。