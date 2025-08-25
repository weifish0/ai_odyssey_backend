import json
import numpy as np
import os
import pickle  # 我們將使用 pickle 來儲存 Python 物件
import re
from dotenv import load_dotenv
from google import genai

load_dotenv()

# --- 1. 設定 API 資訊與檔案路徑 ---
GEMINI_API_KEY = os.getenv("EMBEDDING_MODEL_API_KEY")
MODEL_NAME = "gemini-embedding-001"
DB_FILE_PATH = "vector_database" # 儲存資料庫的檔案名稱

# --- 2. 準備你的長篇文章 ---
def load_article_text(file_path="./rag_text1.txt"):
    """從文件讀取長篇文章"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"成功從 {file_path} 讀取文章，共 {len(text)} 個字符")
        return text
    except FileNotFoundError:
        print(f"錯誤：找不到文件 {file_path}")
        return ""
    except Exception as e:
        print(f"讀取文件時發生錯誤: {e}")
        return ""

# --- 3. 沿用之前的輔助函式 ---

def chunk_text(text, chunk_size=1):
    # 將字面上的 \n 轉為真正的換行，並標準化各系統換行符號
    if "\\n" in text:
        text = text.replace("\\n", "\n")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 以 2 個以上的空行作為段落分隔，並去除空白段落
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n+', text.strip()) if p.strip()]

    # 以 chunk_size 合併段落
    chunks = ['\n\n'.join(paragraphs[i:i + chunk_size]) for i in range(0, len(paragraphs), chunk_size)]
    print(f"成功將文章切分為 {len(chunks)} 個區塊。")
    return chunks

def get_embedding(text_input):
    """使用 Gemini API 生成文本的向量表示"""
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

def create_vector_database(chunks):
    database = []
    print("\n正在建立向量資料庫...")
    
    # 使用 Gemini 的批量處理功能
    try:
        # 創建 Gemini 客戶端
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # 批量生成 embeddings
        print("正在批量生成 embeddings...")
        result = client.models.embed_content(
            model=MODEL_NAME,
            contents=chunks
        )
        
        # 處理結果
        if result.embeddings and len(result.embeddings) == len(chunks):
            for i, (chunk, embedding) in enumerate(zip(chunks, result.embeddings)):
                # 將 ContentEmbedding 轉換為 numpy 數組
                if hasattr(embedding, 'values'):
                    embedding_array = np.array(embedding.values)
                else:
                    embedding_array = np.array(embedding)
                
                database.append({
                    "text": chunk,
                    "embedding": embedding_array
                })
                print(f"✅ 第 {i+1}/{len(chunks)} 個區塊處理完成")
        else:
            print("批量處理失敗，改用單個處理...")
            # 如果批量處理失敗，回退到單個處理
            for i, chunk in enumerate(chunks):
                print(f"正在處理第 {i+1}/{len(chunks)} 個區塊...")
                embedding = get_embedding(chunk)
                if embedding:
                    database.append({
                        "text": chunk,
                        "embedding": embedding
                    })
    except Exception as e:
        print(f"批量處理失敗: {e}，改用單個處理...")
        # 回退到單個處理
        for i, chunk in enumerate(chunks):
            print(f"正在處理第 {i+1}/{len(chunks)} 個區塊...")
            embedding = get_embedding(chunk)
            if embedding:
                database.append({
                    "text": chunk,
                    "embedding": embedding
                })
    
    return database

# --- 4. 主程式執行流程 ---
if __name__ == "__main__":
    if not GEMINI_API_KEY:
        print("警告：請設定 GEMINI_API_KEY 環境變數。")
        print("例如：export GEMINI_API_KEY='your-api-key-here'")
    else:
        for article_num in range(2):
            # 步驟 1: 讀取文章並分塊
            article_text = load_article_text(file_path=f"./rag_text{article_num+1}.txt")
            if article_text:
                article_chunks = chunk_text(article_text)
                
                # 步驟 2: 建立向量資料庫
                vector_db = create_vector_database(article_chunks)

                # 步驟 3: 將資料庫儲存到檔案
                if vector_db:
                    print(f"\n準備將資料庫儲存至 {DB_FILE_PATH}_{article_num+1}...")
                    with open(f"{DB_FILE_PATH}_{article_num+1}.pkl", "wb") as f:
                        pickle.dump(vector_db, f)
                    print("向量資料庫已成功建立並儲存！")
                else:
                    print("建立向量資料庫失敗，未儲存任何檔案。")
            else:
                print(f"無法讀取文章 {article_num+1}，跳過處理。")