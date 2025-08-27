from typing import List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from core.deps import verify_token
from query_engine import load_vector_database, search


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="查詢問題")
    top_k: int = Field(default=5, ge=1, le=20)


class RAGQueryResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_results: int
    query: str


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query-text1", response_model=RAGQueryResponse)
async def query_rag_text1(request: RAGQueryRequest, username: str = Depends(verify_token)):
    try:
        vector_db = load_vector_database("vector_database_1.pkl")
        if not vector_db:
            raise HTTPException(status_code=500, detail="無法載入 rag_text1.txt 的向量資料庫，請先執行 build_rag_db.py 建立資料庫")

        search_results = search(request.query, vector_db, top_k=request.top_k)
        formatted_results = [{"rank": i + 1, "score": float(r["score"]), "text": r["text"]} for i, r in enumerate(search_results)]
        return RAGQueryResponse(success=True, results=formatted_results, total_results=len(formatted_results), query=request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 查詢失敗: {str(e)}")


@router.post("/query-text2", response_model=RAGQueryResponse)
async def query_rag_text2(request: RAGQueryRequest, username: str = Depends(verify_token)):
    try:
        vector_db = load_vector_database("vector_database_2.pkl")
        if not vector_db:
            raise HTTPException(status_code=500, detail="無法載入 rag_text2.txt 的向量資料庫，請先執行 build_rag_db.py 建立資料庫")

        search_results = search(request.query, vector_db, top_k=request.top_k)
        formatted_results = [{"rank": i + 1, "score": float(r["score"]), "text": r["text"]} for i, r in enumerate(search_results)]
        return RAGQueryResponse(success=True, results=formatted_results, total_results=len(formatted_results), query=request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 查詢失敗: {str(e)}")


