from bson import ObjectId
from typing import Dict, Any

def comment_serializer(comment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "_id": str(comment["_id"]),
        "comment_id": comment["comment_id"],
        "platform": comment["platform"],
        "comments": [
            {"comment_content": c["comment_content"]}
            for c in comment["comments"]
        ],
        "post_id": str(comment["post_id"]),
        "analysis": {
            "sentiment_analysis": comment["analysis"].get("sentiment_analysis", "")
        }
    }

def comments_serializer(comments: list) -> list:
    return [comment_serializer(comment) for comment in comments]