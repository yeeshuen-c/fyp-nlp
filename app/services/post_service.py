from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
from ..schemas import Analysis, Engagement, Post, Comment
from ..database import db
from datetime import datetime, timezone
from bson import ObjectId

def get_user_id_filter(user_id: int, platform: Optional[str] = None) -> Dict[str, int]:
    if platform == "Facebook":
        return {"user_id": {"$in": list(range(11, 16))}, "deleted": {"$ne": 1}}
    elif platform == "Twitter":
        return {"user_id": {"$in": list(range(6, 11))}, "deleted": {"$ne": 1}}
    elif platform == "Official Website":
        return {"user_id": user_id, "deleted": {"$ne": 1}}
    else:
         if user_id in [1, 2, 3, 4, 5]:
            return {"user_id": {"$in": [user_id, user_id + 5, user_id + 10]}, "deleted": {"$ne": 1}}

async def get_posts_by_user_id(user_id: int, platform: Optional[str] = None, scam_framing: Optional[str] = None, scam_type: Optional[str] = None) -> List[Post]:
    """
    Retrieve posts by user_id with optional filters for platform, scam_framing, and scam_type.
    Ensure engagement.comment_count2 is used if it exists.
    """
    user_id_filter = get_user_id_filter(user_id, platform)

    if scam_framing:
        user_id_filter["analysis.scam_framing"] = scam_framing
    if scam_type:
        user_id_filter["analysis.scam_type"] = scam_type

    print(user_id_filter, platform)

    # Fetch posts from the database
    posts_data = await db.posts.find(user_id_filter).sort("post_id", -1).to_list(length=None)  # Sort by post_id in descending order
    posts = []
    for post in posts_data:
        # Ensure date is properly formatted
        if 'date' not in post or post['date'] == "No Date":
            post['date'] = "No Date"
        else:
            post['date'] = post['date'].isoformat() if isinstance(post['date'], datetime) else post['date']
        post['date'] = post.get('date', "No Date") if post.get('date') is not None else "No Date"

        # Ensure other fields are set
        post['post_title'] = post.get('post_title', "")
        post['batch'] = int(post['batch']) if isinstance(post['batch'], int) else int(post['batch']) if post['batch'].isdigit() else 0  # Ensure batch is an integer
        post['content'] = post.get('content', "")  # Ensure content is set

        # Use comment_count2 if it exists, otherwise fall back to comment_count
        engagement = post.get('engagement', {})
        if 'comment_count2' in engagement:
            engagement['comment_count'] = engagement['comment_count2']
        post['engagement'] = engagement

        # Append the processed post to the list
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
    
    # Ensure keys and values are valid strings and integers
    scam_type_counts = {}
    for item in result:
        scam_type = item["_id"] if item["_id"] is not None else "Unknown"
        scam_type_counts[scam_type] = item["count"]
    
    return scam_type_counts

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
    """
    Retrieve a post by its post_id and ensure engagement.comment_count2 is used if it exists.
    """
    post_data = await db.posts.find_one({"post_id": post_id, "deleted": {"$ne": 1}}, hint="_id_", max_time_ms=5000)
    print(f"Fetched data from DB: {post_data}")  # Debugging
    if post_data:
        # Ensure date is properly formatted
        post_data['date'] = post_data.get('date', "No Date")
        if post_data['date'] is None:
            post_data['date'] = "No Date"
        elif isinstance(post_data['date'], datetime):
            post_data['date'] = post_data['date'].isoformat()

        # Ensure other fields are set
        post_data['post_title'] = post_data.get('post_title', "")
        post_data['image'] = post_data.get('image', {})
        post_data['media'] = post_data.get('media', [])
        post_data['media_url'] = post_data.get('media_url', [])

        # Use comment_count2 if it exists, otherwise fall back to comment_count
        engagement = post_data.get('engagement', {})
        if 'comment_count2' in engagement:
            engagement['comment_count'] = engagement['comment_count2']
        post_data['engagement'] = engagement

        return Post(**post_data)
    return None 
    post_data = await db.posts.find_one({"post_id": post_id, "deleted": {"$ne": 1}}, hint="_id_", max_time_ms=5000)
    print(f"Fetched data from DB: {post_data}")  # Debugging
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
    post_data = await db.posts.find_one({"post_id": post_id, "deleted": {"$ne": 1}})
    if not post_data:
        return []

    post_object_id = post_data["_id"]

    # Use the ObjectId to find comments related to the post
    comments_data = await db.comments.find({"post_id": post_object_id}).to_list(length=100)
    if not comments_data:
        return []

    combined_comments = []
    platform = None
    analysis = None

    for comment_doc in comments_data:
        # Take the first platform and analysis found
        if platform is None:
            platform = comment_doc.get("platform")
        if analysis is None:
            analysis = comment_doc.get("analysis")

        # Merge all comment arrays and include sentiment_analysis2
        for comment in comment_doc.get("comments", []):
            combined_comments.append({
                "comment_content": comment.get("comment_content"),
                "sentiment_analysis": comment.get("sentiment_analysis2")  # Include sentiment_analysis2
            })

    # Now build a single combined Comment object
    combined_comment_data = {
        "comment_id": comments_data[0]["comment_id"],  # You can decide which comment_id to pick
        "platform": platform,
        "comments": combined_comments,
        "post_id": post_id,
        "analysis": analysis
    }

    return [Comment(**combined_comment_data)]

async def get_combined_comments_by_post_id(post_id: int) -> Dict:
    pipeline = [
        {"$match": {"post_id": post_id}},  # Match documents with the given post_id
        {
            "$group": {
                "_id": "$post_id",  # Group by post_id
                "post_id": {"$first": "$post_id"},  # Retain the post_id
                "platform": {"$first": "$platform"},  # Retain the platform
                "comments": {
                    "$push": {
                        "comment_content": "$comments.comment_content",
                        "sentiment_analysis2": "$comments.sentiment_analysis2"  # Include sentiment_analysis2
                    }
                },
                "analysis": {"$first": "$analysis"}  # Retain the analysis
            }
        },
        {"$project": {"_id": 0}}  # Exclude the _id field from the result
    ]

    result = await db.comments.aggregate(pipeline).to_list(length=1)
    return result[0] if result else {}

# async def get_sentiment_analysis_by_user_id(user_id: int) -> float:
#     user_id_filter = get_user_id_filter(user_id)
    
#     # Fetch all posts by user_id to get post_ids
#     posts = await db.posts.find(user_id_filter).to_list(length=None)
#     post_ids = [post['post_id'] for post in posts]
    
#     sentiment_mapping = {"Positive": 1, "Neutral": 0.5, "Negative": 0}
#     sentiment_analysis = {}
    
#     for post_id in post_ids:
#         comments = await get_comments_by_post_id(post_id)
        
#         for comment in comments:
#             total_sentiment_score = 0
#             total_comments = 0
            
#             for comment_content in comment.comments:
#                 sentiment = getattr(comment_content, "sentiment_analysis", None)
#                 if sentiment in sentiment_mapping:
#                     total_sentiment_score += sentiment_mapping[sentiment]
#                     total_comments += 1
            
#             if total_comments > 0:
#                 average_sentiment = (total_sentiment_score / total_comments) * 100
#                 sentiment_analysis[comment.comment_id] = average_sentiment
    
#     total_sentiment_score = 0
#     total_comments = 0
    
#     for sentiment in sentiment_analysis.values():
#         total_sentiment_score += sentiment
#         total_comments += 1
    
#     if total_comments > 0:
#         overall_average_sentiment = total_sentiment_score / total_comments
#     else:
#         overall_average_sentiment = 0
    
#     return overall_average_sentiment

async def get_sentiment_analysis_by_user_id(user_id: int) -> Dict[str, float]:
    """
    Get sentiment analysis for a user_id and return the percentage of positive, negative, and neutral sentiments.
    """
    user_id_filter = get_user_id_filter(user_id)
    
    # Fetch all posts by user_id to get post_ids
    posts = await db.posts.find(user_id_filter).to_list(length=None)
    post_ids = [post['post_id'] for post in posts]
    
    sentiment_mapping = {"positive": "positive", "neutral": "neutral", "negative": "negative"}
    sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    total_comments = 0
    
    for post_id in post_ids:
        comments = await get_comments_by_post_id(post_id)
        
        for comment in comments:
            for comment_content in comment.comments:
                sentiment = getattr(comment_content, "sentiment_analysis", None)
                if sentiment in sentiment_mapping:
                    sentiment_counts[sentiment_mapping[sentiment]] += 1
                    total_comments += 1
    
    # Calculate percentages
    if total_comments > 0:
        sentiment_percentages = {
            "positive": (sentiment_counts["positive"] / total_comments) * 100,
            "neutral": (sentiment_counts["neutral"] / total_comments) * 100,
            "negative": (sentiment_counts["negative"] / total_comments) * 100,
        }
    else:
        sentiment_percentages = {"positive": 0, "neutral": 0, "negative": 0}
    
    return sentiment_percentages

async def add_new_post(post_title: Optional[str], post_content: str, user_id: int, url: Optional[str]) -> Post:
    # Get the largest post_id from the database
    largest_post = await db.posts.find_one(sort=[("post_id", -1)])
    new_post_id = largest_post["post_id"] + 1 if largest_post else 1

    largest_batch = await db.posts.find_one(sort=[("batch", -1)])
    new_batch_id = largest_batch["batch"] + 1 if largest_batch else 1

    # Create the new post document
    new_post = {
        "post_id": new_post_id,
        "post_title": post_title,
        "content": post_content,
        "date": datetime.now(timezone.utc).isoformat(),  # Convert datetime to string
        "post_url": url,
        "user_id": user_id,
        "engagement": Engagement().dict(),  # Ensure correct format
        "analysis": Analysis().dict(),  # Ensure correct format
        "batch": new_batch_id
    }

    # Insert the new post into the database
    await db.posts.insert_one(new_post)

    return Post(**new_post)

async def update_post(post_id: int, user_id: int, post_title: Optional[str] = None, post_content: Optional[str] = None, url: Optional[str] = None) -> Optional[Post]:
    update_data = {}
    if post_title is not None:
        update_data["post_title"] = post_title
    if post_content is not None:
        update_data["content"] = post_content
    if url is not None:
        update_data["post_url"] = url
    print(update_data)

    if not update_data:
        raise ValueError("No data provided to update")

    result = await db.posts.find_one_and_update(
        {"post_id": post_id, "user_id": user_id},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER
    )

    print(f"Updated document: {result}")  # Debugging

    if result is None:
        return None

    return Post(**result)

async def mark_post_as_deleted(post_id: int, user_id: int) -> Optional[Post]:
    result = await db.posts.find_one_and_update(
        {"post_id": post_id, "user_id": user_id},
        {"$set": {"deleted": 1}},
        return_document=ReturnDocument.AFTER
    )

    if result is None:
        return None

    return Post(**result)

async def count_posts_by_user_id_group_by_scam_framing(user_id: int) -> Dict[str, int]:
    """
    Count posts grouped by scam_framing for a given user_id.
    """
    user_id_filter = get_user_id_filter(user_id)
    pipeline = [
        {"$match": user_id_filter},
        {"$group": {"_id": "$analysis.scam_framing", "count": {"$sum": 1}}}
    ]
    result = await db.posts.aggregate(pipeline).to_list(length=None)
    
    # Ensure keys and values are valid strings and integers
    scam_framing_counts = {}
    for item in result:
        scam_framing = item["_id"] if item["_id"] is not None else "Unknown"
        scam_framing_counts[scam_framing] = item["count"]
    
    return scam_framing_counts