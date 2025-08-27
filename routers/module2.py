from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List

from core.deps import verify_token
from ml_state import is_mobilenet_ready, get_mobilenet, get_user_models, set_user_model, get_user_model
from image_classification import ImageClassificationModel


class ImageInfo(BaseModel):
    name: str
    images: List[str]


class TrainingRequest(BaseModel):
    train_dataset: List[ImageInfo]


class PredictionRequest(BaseModel):
    image_path: str = Field(..., description="要進行預測的圖片在伺服器上的相對路徑")


router = APIRouter(prefix="/module2", tags=["module2"])


@router.post("/train/{user_name}")
async def train_user_model(user_name: str, request: TrainingRequest, current_user: str = Depends(verify_token)):
    if user_name != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="只能訓練自己的模型")
    
    if not is_mobilenet_ready():
        return {"success": False, "error": "基礎模型尚未準備就緒，請稍後再試。"}
    
    base_model = get_mobilenet()
    user_models = get_user_models()
    
    if user_name not in user_models:
        new_model = ImageClassificationModel(user_name=user_name, base_model=base_model)
        set_user_model(user_name, new_model)
    
    model_manager = get_user_model(user_name)
    data_config = request.model_dump()
    if not model_manager.load_training_data(data_config):
        return {"success": False, "error": "載入訓練數據失敗，請檢查圖片路徑。"}
    
    result = await model_manager.train_model()
    set_user_model(user_name, model_manager)
    return result


@router.post("/predict/{user_name}")
async def predict_with_user_model(user_name: str, request: PredictionRequest, current_user: str = Depends(verify_token)):
    if user_name != current_user:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="只能使用自己的模型")
    
    if not is_mobilenet_ready():
        return {"success": False, "error": "基礎模型尚未準備就緒，請稍後再試。"}
    
    base_model = get_mobilenet()
    user_models = get_user_models()
    
    if user_name not in user_models:
        new_model = ImageClassificationModel(user_name=user_name, base_model=base_model)
        set_user_model(user_name, new_model)

    model_manager = get_user_model(user_name)

    if not model_manager.is_trained:
        return {"success": False, "error": f"使用者 {user_name} 的模型尚未訓練，請先調用訓練 API。"}
    
    prediction = model_manager.predict_image(request.image_path)
    return prediction


