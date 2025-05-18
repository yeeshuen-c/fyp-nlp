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
        "agencies_name": users_data[0].get("agencies_name", ""),
        "password": users_data[0]["password"],
        "username": users_data[0].get("username", ""),
        "platforms": []
    }

    for user_data in users_data:
        platform_data = user_data.get("platform", {})
        platform = {
            "followers": platform_data.get("followers", ""),
            "platform_name": platform_data.get("platform_name", ""),
            "url": platform_data.get("url", "")
        }
        combined_user["platforms"].append(platform)

    return combined_user

async def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    user_data = await db.users.find_one({"username": username})
    if user_data:
        return user_data
    return None

from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def create_user(username: str, password: str):
    existing_user = await db.users.find_one({"username": username})
    if existing_user:
        return None  # Username already exists
    
    largest_user = await db.users.find_one(sort=[("user_id", -1)])
    new_user_id = largest_user["user_id"] + 1 if largest_user else 1

    hashed_password = pwd_context.hash(password)
    user_data = {
        "user_id": new_user_id,
        "username": username,
        "password": hashed_password
    }
    result = await db.users.insert_one(user_data)
    user_data["user_id"] = str(result.inserted_id)
    print("User created:", user_data)
    return user_data