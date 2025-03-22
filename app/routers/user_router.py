from fastapi import APIRouter
from ..schemas import User
from ..controllers.user_controller import fetch_user_by_id

router = APIRouter()

@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    return await fetch_user_by_id(user_id)