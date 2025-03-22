from fastapi import HTTPException
from ..schemas import User
from ..services.user_service import get_user_by_id

async def fetch_user_by_id(user_id: int) -> User:
    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user