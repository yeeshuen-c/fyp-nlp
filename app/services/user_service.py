from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient
from ..schemas import User
from ..database import db

def get_user_id_filter(user_id: int) -> Dict[str, int]:
    if user_id in [1, 2, 3, 4, 5]:
        return {"user_id": {"$in": [user_id, user_id + 5, user_id + 10]}}
    return {"user_id": user_id}

async def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    user_id_filter = get_user_id_filter(user_id)
    users_data = await db.users.find(user_id_filter).to_list(length=3)
    
    if not users_data:
        return None

    combined_user = {
        "user_id": user_id,
        "agencies_name": users_data[0]["agencies_name"],
        "password": users_data[0]["password"],
        "username": users_data[0].get("username", ""),
        "platforms": []
    }

    for user_data in users_data:
        platform = {
            "followers": user_data["platform"].get("followers"),
            "platform_name": user_data["platform"]["platform_name"],
            "url": user_data["platform"]["url"]
        }
        combined_user["platforms"].append(platform)

    return combined_user

async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    user_data = await db.users.find_one({"username": username})
    if user_data:
        return user_data
    return None