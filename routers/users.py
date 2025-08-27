from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

from core.deps import verify_token
from database import db_manager


class UpdateScoreRequest(BaseModel):
    score: int = Field(..., ge=0, description="新的分數")
    level: Optional[int] = Field(None, ge=1, le=3, description="要更新的關卡（1~3），不提供則更新總分")


class UpdateScoreResponse(BaseModel):
    status: str = "success"
    message: str
    data: dict


router = APIRouter(prefix="/users", tags=["users"])


@router.get("")
async def list_users_scores(current_user: str = Depends(verify_token)):
    """列出所有使用者的分數（需登入）"""
    users = db_manager.get_all_users_with_scores()
    return {
        "status": "success",
        "data": users,
    }

@router.get("/{username}/statistics")
async def get_user_statistics(username: str, current_user: str = Depends(verify_token)):
    if username != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="只能查看自己的統計資訊")
    user = db_manager.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="使用者不存在")
    statistics = db_manager.get_user_statistics(user["id"])
    return {"status": "success", "data": statistics}


@router.put("/{username}/score")
async def update_user_score(
    username: str, 
    request: UpdateScoreRequest, 
    current_user: str = Depends(verify_token)
):
    """更新玩家分數
    - 若 body.level = 1/2/3：更新對應關卡分數並重算總分
    - 若 body.level 沒給：僅更新總分
    """
    if username != current_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="只能更新自己的分數"
        )
    
    try:
        # 更新使用者分數（指定關卡或總分）
        success = db_manager.update_user_score(username, request.score, request.level)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="使用者不存在或更新失敗"
            )
        
        payload = {
            "username": username,
            "new_score": request.score,
        }
        if request.level is not None:
            payload["level"] = request.level
        
        return UpdateScoreResponse(
            message="分數更新成功",
            data=payload
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新分數失敗: {str(e)}"
        )


