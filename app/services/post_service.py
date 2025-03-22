from typing import List, Dict
from motor.motor_asyncio import AsyncIOMotorClient
from ..schemas import Post, Comment
from ..database import db
from datetime import datetime
from bson import ObjectId

def get_user_id_filter(user_id: int) -> Dict[str, int]:
    if user_id in [1, 2, 3, 4, 5]:
        return {"user_id": {"$in": [user_id, user_id + 5, user_id + 10]}}
    return {"user_id": user_id}

async def get_posts_by_user_id(user_id: int) -> List[Post]:
    user_id_filter = get_user_id_filter(user_id)
    posts_data = await db.posts.find(user_id_filter).to_list(length=100)
    posts = []
    for post in posts_data:
        if 'date' not in post or post['date'] == "No Date":
            post['date'] = "No Date"
        else:
            post['date'] = post['date'].isoformat() if isinstance(post['date'], datetime) else post['date']
        post['date'] = post.get('date', "No Date") if post.get('date') is not None else "No Date"
        post['post_title'] = post.get('post_title', "")
        posts.append(Post(**post))
    return posts

async def count_posts_by_user_id(user_id: int) -> int:
    user_id_filter = get_user_id_filter(user_id)
    count = await db.posts.count_documents(user_id_filter)
    return count

async def count_posts_by_user_id_group_by_scam_type(user_id: int) -> Dict[str, int]:
    user_id_filter = get_user_id_filter(user_id)
    pipeline = [
        {"$match": user_id_filter},
        {"$group": {"_id": "$analysis.scam_type", "count": {"$sum": 1}}}
    ]
    result = await db.posts.aggregate(pipeline).to_list(length=None)
    return {item["_id"]: item["count"] for item in result}

async def count_posts_by_user_id_group_by_platform(user_id: int) -> Dict[str, int]:
    user_id_filter = get_user_id_filter(user_id)
    pipeline = [
        {"$match": user_id_filter},
        {"$group": {
            "_id": {
                "$cond": [
                    {"$lte": ["$user_id", 5]}, "Official Website",
                    {"$cond": [
                        {"$lte": ["$user_id", 10]}, "Facebook (meta)",
                        "Twitter (X)"
                    ]}
                ]
            },
            "count": {"$sum": 1}
        }}
    ]
    result = await db.posts.aggregate(pipeline).to_list(length=None)
    return {item["_id"]: item["count"] for item in result}

async def count_posts_by_user_id_group_by_scam_type_and_framing(user_id: int) -> Dict[str, any]:
    user_id_filter = get_user_id_filter(user_id)
    pipeline = [
        {"$match": user_id_filter},
        {"$group": {
            "_id": {
                "scam_type": "$analysis.scam_type",
                "scam_framing": "$analysis.scam_framing"
            },
            "count": {"$sum": 1}
        }}
    ]
    result = await db.posts.aggregate(pipeline).to_list(length=None)
    scam_type_counts = {}
    for item in result:
        scam_type = item["_id"]["scam_type"]
        scam_framing = item["_id"]["scam_framing"]
        if scam_type not in scam_type_counts:
            scam_type_counts[scam_type] = {"total_count": 0, "framings": {}}
        scam_type_counts[scam_type]["total_count"] += item["count"]
        if scam_framing not in scam_type_counts[scam_type]["framings"]:
            scam_type_counts[scam_type]["framings"][scam_framing] = 0
        scam_type_counts[scam_type]["framings"][scam_framing] += item["count"]
    return scam_type_counts

async def get_post_by_id(post_id: int) -> Post:
    post_data = await db.posts.find_one({"post_id": post_id})
    if post_data:
        post_data['date'] = post_data.get('date', "No Date")
        if post_data['date'] is None:
            post_data['date'] = "No Date"
        elif isinstance(post_data['date'], datetime):
            post_data['date'] = post_data['date'].isoformat()
        post_data['post_title'] = post_data.get('post_title', "")
        post_data['image'] = post_data.get('image', {})
        post_data['media'] = post_data.get('media', [])
        post_data['media_url'] = post_data.get('media_url', [])
        return Post(**post_data)
    return None

async def get_comments_by_post_id(post_id: int) -> List[Comment]:
    # First, get the post's ObjectId using the integer post_id
    post_data = await db.posts.find_one({"post_id": post_id})
    if not post_data:
        return []

    post_object_id = post_data["_id"]

    # Use the ObjectId to find comments related to the post
    comments_data = await db.comments.find({"post_id": post_object_id}).to_list(length=100)
    comments = []
    for comment in comments_data:
        comment["post_id"] = post_id
        comments.append(Comment(**comment))
    return comments