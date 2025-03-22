from bson import ObjectId
from typing import Dict, Any

def user_serializer(user: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(user["_id"]),
        "user_id": user["user_id"],
        "agencies_name": user["agencies_name"],
        "password": user["password"],
        "platform": {
            "followers": user["platform"].get("followers"),
            "platform_name": user["platform"]["platform_name"]
        }
    }

def users_serializer(users: list) -> list:
    return [user_serializer(user) for user in users]