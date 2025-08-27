from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from core.config import ACCESS_TOKEN_EXPIRE_MINUTES
from core.deps import create_access_token, verify_token, security
from database import db_manager


class UserRegister(BaseModel):
    username: str = Field(..., description="使用者名稱")
    password: str = Field(..., description="密碼")


class UserLogin(BaseModel):
    username: str = Field(..., description="使用者名稱")
    password: str = Field(..., description="密碼")


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    try:
        user = db_manager.create_user(
            username=user_data.username,
            password=user_data.password,
            initial_money=0,
        )
        return {
            "message": "註冊成功！",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "money": user["money"],
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="註冊過程中發生錯誤，請稍後再試。",
        )


@router.post("/login")
async def login(user_data: UserLogin):
    try:
        user = db_manager.verify_user(user_data.username, user_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="登入失敗：請檢查您的帳號名稱和密碼是否正確。如果忘記密碼，請聯繫管理員。",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user_data.username}, expires_delta=access_token_expires)

        expires_at = datetime.now(timezone.utc) + access_token_expires
        db_manager.create_session(user["id"], access_token, expires_at)

        return {
            "message": "登入成功！",
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "id": user["id"],
                "username": user["username"],
                "money": user["money"],
            },
        }
    except ValueError as e:
        error_message = str(e)
        if "帳號已被停用" in error_message:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="帳號已被停用，請聯繫管理員。")
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_message)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登入過程中發生系統錯誤，請稍後再試。如果問題持續發生，請聯繫技術支援。",
        )


@router.get("/me")
async def get_current_user(username: str = Depends(verify_token)):
    user = db_manager.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="使用者不存在")
    return {
        "status": "success",
        "data": {
            "user": {
                "id": user["id"],
                "username": user["username"],
                "money": user["money"],
                "created_at": user["created_at"],
                "last_login": user["last_login"],
            }
        },
    }


@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if not db_manager.is_session_valid(token):
        return {"message": "會話已過期或不存在"}
    if db_manager.invalidate_session(token):
        return {"message": "登出成功！"}
    return {"message": "會話已過期或不存在"}


