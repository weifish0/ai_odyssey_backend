# AI Odyssey Backend API

這是一個使用 FastAPI 開發的 AI 學習遊戲平台後端 API，整合了 NCHC (國家高速網路與計算中心) Chat Completions API。

## 功能特色

- 🚀 基於 FastAPI 的高效能 API 服務
- 🔐 JWT 認證系統
- 🎮 AI 學習遊戲模組
- 🐟 模組二：魚池的財富密碼 (Image Recognition)
- 👑 模組三：國王的厭食症 (Generative AI)
- 🔄 完整的 Chat Completions API 轉發
- 📊 健康檢查端點
- 📚 自動生成的 API 文件

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 設定環境變數

複製 `env.example` 為 `.env` 並填入您的 API Key：

```bash
cp env.example .env
```

編輯 `.env` 檔案：
```
NCHC_API_KEY=your_actual_api_key_here
SECRET_KEY=your_jwt_secret_key_here
```

### 3. 啟動服務

```bash
python main.py
```

或使用 uvicorn：

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. 存取 API 文件

開啟瀏覽器前往：http://localhost:8000/docs

## API 端點

### 1. 根路徑
- **GET** `/` - 取得 API 基本資訊

### 2. 健康檢查
- **GET** `/health` - 檢查服務狀態和 API Key 設定

### 3. 帳號與認證 (Authentication)

#### 3.1 使用者註冊
- **POST** `/auth/register` - 建立新使用者帳號

請求範例：
```json
{
    "username": "player001",
    "password": "a_strong_password"
}
```

#### 3.2 使用者登入
- **POST** `/auth/login` - 使用者登入獲取 JWT Token

請求範例：
```json
{
    "username": "player001",
    "password": "a_strong_password"
}
```

#### 3.3 獲取當前使用者資訊
- **GET** `/users/me` - 獲取當前登入使用者資訊 (需要 JWT Token)

### 4. 模組二：魚池的財富密碼 (Image Recognition)

#### 4.1 詢問 AI 寵物
- **POST** `/module2/ask-pet` - 向 AI 寵物提問關於魚類特徵

請求範例：
```json
{
    "question": "銀龍魚和吳郭魚有什麼外觀特徵？"
}
```

### 5. 模組三：國王的厭食症 (Generative AI)

#### 5.1 生成食譜描述
- **POST** `/module3/generate-recipe-text` - 使用 AI 生成創意食譜描述

請求範例：
```json
{
    "prompt": "一道融合東西方風味的創意料理"
}
```

#### 5.2 生成食譜圖片
- **POST** `/module3/generate-recipe-image` - 使用 DALL-E 生成食譜圖片

請求範例：
```json
{
    "prompt": "香煎鮭魚配時蔬"
}
```

**⚠️ 重要提醒：**
- 請使用正面、具體的食物描述
- 避免使用可能觸發安全系統的詞彙
- 建議描述：食材、烹飪方式、菜色外觀、味道特色等

**安全提示詞範例：**
✅ 好的提示詞：
- "香煎鮭魚配時蔬"
- "義大利麵配番茄醬"
- "清蒸鱸魚配薑絲"
- "烤雞胸肉配馬鈴薯泥"

❌ 避免的提示詞：
- 包含暴力、不當內容的描述
- 過於抽象或模糊的描述
- 可能涉及爭議的內容

#### 4.2 獲取待分類的魚類圖片
- **GET** `/module2/training-images` - 獲取待分類的魚類圖片

#### 4.3 提交標註結果
- **POST** `/module2/submit-labels` - 提交魚類分類標註結果

請求範例：
```json
{
    "labels": [
        { "image_id": "img_fish_001", "classification": "arowana" },
        { "image_id": "img_fish_002", "classification": "tilapia" }
    ]
}
```

#### 4.4 進行 AI 辨識
- **POST** `/module2/identify-fish` - 進行 AI 魚類辨識

### 5. 模組三：國王的厭食症 (Generative AI)

#### 5.1 創造食譜描述
- **POST** `/module3/generate-recipe-text` - 根據提示詞生成食譜描述

請求範例：
```json
{
    "prompt": "為一位熱愛星空的國王，設計一道看起來像迷你銀河的甜點"
}
```

#### 5.2 視覺化菜色
- **POST** `/module3/generate-recipe-image` - 生成菜色圖片

請求範例：
```json
{
    "prompt": "一道名為「星夜琉璃脆」的甜點，深黑色的圓潤糕點"
}
```

### 6. 向後相容端點

#### 6.1 完整 Chat Completions API
- **POST** `/chat/completions` - 完整的聊天完成 API，直接轉發到 NCHC

#### 6.2 簡化聊天端點
- **POST** `/chat/simple` - 簡化的聊天端點

#### 6.3 模型列表
- **GET** `/models` - 取得可用的模型列表

## 認證機制

除註冊與登入外，所有需要使用者身份的 API 請求，都必須在 HTTP Header 中帶上 Authorization 欄位：

```
Authorization: Bearer <JWT_TOKEN>
```

## 通用回應格式

### 成功回應
```json
{
    "status": "success",
    "data": {
        // 回應的具體資料
    }
}
```

### 失敗回應
```json
{
    "status": "error",
    "message": "具體的錯誤訊息"
}
```

## 部署到 Render

此專案已配置好部署到 Render 平台：

1. 將程式碼推送到 GitHub
2. 在 Render 中連接您的 GitHub 儲存庫
3. 設定環境變數 `NCHC_API_KEY` 和 `SECRET_KEY`
4. 部署完成後即可使用

## 環境變數

| 變數名稱 | 說明 | 必填 |
|---------|------|------|
| `NCHC_API_KEY` | NCHC API 金鑰 | ✅ |
| `SECRET_KEY` | JWT 簽名金鑰 | ✅ |
| `PORT` | 服務埠號 | ❌ (預設: 8000) |
| `HOST` | 服務主機 | ❌ (預設: 0.0.0.0) |

## 錯誤處理

API 包含完整的錯誤處理機制：

- **400** - 請求格式錯誤
- **401** - 認證失敗
- **408** - 請求超時
- **500** - 伺服器錯誤

## 開發

### 本地開發

```bash
# 安裝開發依賴
pip install -r requirements.txt

# 啟動開發伺服器
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 測試 API

使用 curl 測試：

```bash
# 註冊新使用者
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'

# 登入
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'

# 使用 JWT Token 測試 API
curl -X GET "http://localhost:8000/users/me" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## 授權

此專案僅供學習和研究使用。請確保您有合法的 NCHC API 存取權限。

## 支援

如有問題，請檢查：

1. API Key 是否正確設定
2. JWT Secret Key 是否設定
3. 網路連線是否正常
4. NCHC API 服務是否可用 