import json
import numpy as np
import os
import pickle
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- 1. 設定 API 資訊與檔案路徑 ---
GEMINI_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")
MODEL_NAME = "gemini-embedding-001"
DB_FILE_PATH = "vector_database_1.pkl" # 指定要載入的資料庫檔案

# --- 2. 定義查詢相關函式 (與上個檔案相同) ---

def get_embedding(text_input):
    # 這個函式現在只用於處理使用者的查詢
    try:
        # 創建 Gemini 客戶端
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 使用 Gemini 生成 embedding
        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=text_input
        )
        
        # 返回 embedding 向量
        if result.embeddings and len(result.embeddings) > 0:
            # 將 ContentEmbedding 轉換為 numpy 數組
            embedding = result.embeddings[0]
            if hasattr(embedding, 'values'):
                return np.array(embedding.values)
            else:
                # 如果沒有 values 屬性，嘗試直接轉換
                return np.array(embedding)
        else:
            print(f"無法生成文本的 embedding")
            return None
            
    except Exception as e:
        print(f"Gemini API 請求失敗: {e}")
        return None

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def search(query, database, top_k=5):
    print(f"\n正在搜尋與「{query}」相關的資料...")
    query_embedding = get_embedding(query)
    
    # 調試信息
    print(f"Debug: query_embedding 類型: {type(query_embedding)}")
    if query_embedding is not None:
        print(f"Debug: query_embedding 形狀: {query_embedding.shape if hasattr(query_embedding, 'shape') else 'N/A'}")
    
    # 檢查 query_embedding 是否為 None 或空數組
    if query_embedding is None or (isinstance(query_embedding, np.ndarray) and query_embedding.size == 0):
        print("無法為您的查詢生成 embedding，搜尋中止。")
        return []
    
    # 確保 query_embedding 是 numpy 數組
    if not isinstance(query_embedding, np.ndarray):
        query_vec = np.array(query_embedding)
    else:
        query_vec = query_embedding
    
    print(f"Debug: query_vec 類型: {type(query_vec)}, 形狀: {query_vec.shape}")
    
    scores = []
    for item in database:
        similarity = cosine_similarity(query_vec, item['embedding'])
        scores.append({
            "score": similarity,
            "text": item['text']
        })
    
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    return sorted_scores[:top_k]

# --- 3. 載入向量資料庫 ---

def load_vector_database(path):
    try:
        with open(path, "rb") as f:
            database = pickle.load(f)
        print(f"成功從 {path} 載入向量資料庫。")
        return database
    except FileNotFoundError:
        print(f"錯誤：找不到資料庫檔案 {path}。")
        print("請先執行 `build_database.py` 來建立資料庫。")
        return None

# --- 4. 主程式執行流程 ---
if __name__ == "__main__":
    # 檢查 API 金鑰
    if not GEMINI_API_KEY:
        print("錯誤：請設定 EMBEDDING_MODEL_API_KEY 環境變數")
        print("例如：export EMBEDDING_MODEL_API_KEY='your-gemini-api-key-here'")
        exit(1)
    
    # 步驟 1: 載入預先建立的資料庫
    vector_db = load_vector_database(DB_FILE_PATH)

    if vector_db:
        print(f"✅ 成功載入向量資料庫，共 {len(vector_db)} 個文本塊")
        print(f"🔍 使用模型: {MODEL_NAME}")
        print("=" * 50)
        
        # 步驟 2: 進入一個無限迴圈來接收使用者查詢
        # 在真實的後端應用中，這裡會是一個 API endpoint
        while True:
            user_query = input("\n請輸入你想查詢的關鍵字或問題 (輸入 'q' 退出)：")
            if user_query.lower() == 'q':
                break
            
            # 步驟 3: 執行搜尋
            results = search(user_query, vector_db, top_k=5)

            # 步驟 4: 顯示結果
            print("\n--- 搜尋結果 ---")
            if results:
                for i, result in enumerate(results):
                    print(f"相關度排名 {i+1} (分數: {result['score']:.4f}):")
                    print(result['text'])
                    print("-" * 20)
            else:
                print("找不到相關的資料。")
    else:
        print("❌ 無法載入向量資料庫，請先執行 `build_rag_db.py` 來建立資料庫。")

