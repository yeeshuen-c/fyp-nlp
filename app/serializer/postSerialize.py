from bson import ObjectId
from typing import Dict, Any

def post_serializer(post: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(post["_id"]),
        "post_id": post["post_id"],
        "post_url": post["post_url"],
        "post_title": post["post_title"],
        "date": post["date"],
        "content": post["content"],
        "media_url": post["media_url"],
        "image": post["image"],
        "engagement": {
            "likes": post["engagement"]["likes"],
            "shares": post["engagement"]["shares"],
            "comment_count": post["engagement"]["comment_count"]
        },
        "analysis": {
            "scam_framing": post["analysis"]["scam_framing2"],
            "scam_or_not": post["analysis"].get("scam_or_not"),
            "scam_type": post["analysis"]["scam_type"]
        },
        "user_id": post["user_id"],
        "batch": post["batch"],
        "media": post["media"]
    }

def posts_serializer(posts: list) -> list:
    return [post_serializer(post) for post in posts]