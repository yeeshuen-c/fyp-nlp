from collections import Counter
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReturnDocument
import requests
from ..schemas import Analysis, Engagement, Post, Comment
from ..database import db
from datetime import datetime, timezone
from bson import ObjectId
import joblib
from ml.preprocessing import clean_text, transform_text
from ml.cvtrain import detect_scam_type_by_keywords  # If using rule-based matcher
from app.services.comment_service import classify_comment_sentiment
import os
from dotenv import load_dotenv

load_dotenv()

def get_user_id_filter(user_id: int, platform: Optional[str] = None) -> Dict[str, int]:
    if platform == "Facebook":
        if user_id > 15:
            return {"user_id": {"$in": list(range(11, 16))}, "deleted": {"$ne": 1}}
        return {"user_id": user_id + 10, "deleted": {"$ne": 1}}
    elif platform == "Twitter":
        if user_id > 15:
            # Twitter users are 6-10
            return {"user_id": {"$in": list(range(6, 11))}, "deleted": {"$ne": 1}}
        return {"user_id": user_id + 5, "deleted": {"$ne": 1}}
    elif platform == "Official Website":
        if user_id > 15:
            # Official Website users are 1-5
            return {"user_id": {"$in": list(range(1, 6))}, "deleted": {"$ne": 1}}
        return {"user_id": user_id, "deleted": {"$ne": 1}}
    else:
        if user_id in [1, 2, 3, 4, 5]:
            return {"user_id": {"$in": [user_id, user_id + 5, user_id + 10]}, "deleted": {"$ne": 1}}
        # return {"user_id": user_id, "deleted": {"$ne": 1}}
    if user_id > 15:
        # Remove user_id restriction, only filter by deleted
        return {"deleted": {"$ne": 1}}
    
async def get_posts_by_user_id(
    user_id: int,
    platform: Optional[str] = None,
    scam_framing: Optional[str] = None,
    scam_type: Optional[str] = None,
    offset: int = 0,
    limit: int = 10,
    agency: Optional[str] = None  # Add agency filter
) -> List[Post]:
    # Map agency names to user_id
    agency_map = {
        "BNM": 1,
        "MCMC": 2,
        "PDRM": 3,
        "MOF": 4,
        "SECCOM": 5
    }
    # If agency is provided, override user_id
    if agency:
        user_id = agency_map.get(agency.upper(), user_id)
    print(f"Fetching posts for user_id: {user_id}, platform: {platform}")
    user_id_filter = get_user_id_filter(user_id, platform)

    if scam_framing:
        user_id_filter["analysis.scam_framing"] = scam_framing
    if scam_type:
        user_id_filter["analysis.scam_type"] = scam_type

    posts_data = (
        await db.posts.find(user_id_filter)
        .sort("post_id", -1)
        .skip(offset)
        .limit(limit)
        .to_list(length=limit)
    )

    posts = []
    for post in posts_data:
        # Ensure date is properly formatted
        if 'date' not in post or post['date'] == "No Date":
            post['date'] = "No Date"
        else:
            post['date'] = post['date'].isoformat() if isinstance(post['date'], datetime) else post['date']
        post['date'] = post.get('date', "No Date") if post.get('date') is not None else "No Date"

        post['post_title'] = post.get('post_title', "")
        post['batch'] = int(post['batch']) if isinstance(post['batch'], int) else int(post['batch']) if str(post['batch']).isdigit() else 0
        post['content'] = post.get('content', "")

        engagement = post.get('engagement', {})
        if 'comment_count2' in engagement:
            engagement['comment_count'] = engagement['comment_count2']
        post['engagement'] = engagement

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
    cursor = db.comments.find({"post_id": post_object_id})
    comment_docs = await cursor.to_list(length=None)

    # Flatten subcomments and attach the parent comment_id
    flattened_comments = []
    for doc in comment_docs:
        for subcomment in doc.get("comments", []):
            flattened_comments.append({
                "comment_id": doc.get("comment_id"),
                "comment_content": subcomment.get("comment_content"),
                "sentiment_analysis": subcomment.get("sentiment_analysis") or subcomment.get("sentiment_analysis2", "")
            })

    return {"comments": flattened_comments}

async def delete_comment_by_id(comment_id: int) -> bool:
    """
    Delete a comment by its comment_id.
    Before deleting, decrement the engagement.comment_count of the related post by 1.
    Returns True if the comment was deleted, False otherwise.
    """
    # Find the comment document to get the post_id
    comment_doc = await db.comments.find_one({"comment_id": comment_id})
    if not comment_doc:
        return False

    # Get the post_id (ObjectId) from the comment document
    post_object_id = comment_doc.get("post_id")
    if not post_object_id:
        return False

    # Find the post document using the ObjectId
    post_doc = await db.posts.find_one({"_id": post_object_id})
    if post_doc:
        # Get current comment_count, convert to int if needed
        engagement = post_doc.get("engagement", {})
        current_count = engagement.get("comment_count", 0)
        try:
            current_count = int(current_count)
        except Exception:
            current_count = 0
        # Decrement, but not below zero
        new_count = max(0, current_count - 1)
        await db.posts.update_one(
            {"_id": post_object_id},
            {"$set": {"engagement.comment_count": str(new_count)}}
        )

    # Now delete the comment
    result = await db.comments.delete_one({"comment_id": comment_id})
    return result.deleted_count > 0
    """
    Delete a comment by its comment_id.
    Returns True if the comment was deleted, False otherwise.
    """
    result = await db.comments.delete_one({"comment_id": comment_id})
    return result.deleted_count > 0

async def update_post_likes(post_id: int, increment: bool = True) -> bool:
    """
    Update the engagement.likes count for a specific post.
    Increment or decrement the likes count based on the `increment` parameter.
    Handles conversion between string and numeric types.
    Returns True if the update was successful, False otherwise.
    """
    # Retrieve the current likes value
    post = await db.posts.find_one({"post_id": post_id})
    if not post:
        raise ValueError(f"Post with post_id {post_id} not found.")

    # Get the current likes value and convert it to an integer
    current_likes = post.get("engagement", {}).get("likes", "0")
    if not isinstance(current_likes, (int, float)):
        try:
            current_likes = int(current_likes)
        except ValueError:
            raise ValueError(f"Invalid 'likes' value for post_id {post_id}: {current_likes}")

    # Increment or decrement the likes count
    updated_likes = current_likes + 1 if increment else max(0, current_likes - 1)

    # Convert the updated likes back to a string and update the database
    result = await db.posts.update_one(
        {"post_id": post_id},
        {"$set": {"engagement.likes": str(updated_likes)}}
    )
    return result.modified_count > 0

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

vectorizer = joblib.load("ml/scamtype/vectorizer.pkl")
type_lr_model = joblib.load("ml/scamtype/logistic_regression_model.pkl")

def classify_scamtype(text: str) -> str:
    cleaned_text = clean_text(text)

    # # Rule-based keyword match
    # rule_label = detect_scam_type_by_keywords(cleaned_text)
    # if rule_label:
    #     return rule_label

    # ML-based prediction
    transformed = vectorizer.transform([cleaned_text])
    prediction = type_lr_model.predict(transformed)[0]
    return prediction

framing_lr_model = joblib.load("ml/metrics/scamframing3/logistic_regression_model.pkl")

def classify_scamframing(text: str) -> str:
    cleaned_text = clean_text(text)

    # ML-based prediction
    transformed = vectorizer.transform([cleaned_text])
    prediction = framing_lr_model.predict(transformed)[0]
    return prediction

async def add_new_post(post_title: Optional[str], post_content: str, user_id: int, url: Optional[str]) -> Post:
    # Get the largest post_id from the database
    largest_post = await db.posts.find_one(sort=[("post_id", -1)])
    new_post_id = largest_post["post_id"] + 1 if largest_post else 1

    largest_batch = await db.posts.find_one(sort=[("batch", -1)])
    new_batch_id = largest_batch["batch"] + 1 if largest_batch else 1

    scam_type = classify_scamtype(post_content)
    scam_framing = classify_scamframing(post_content)

    # Create the new post document
    new_post = {
        "post_id": new_post_id,
        "post_title": post_title,
        "content": post_content,
        "date": datetime.now(timezone.utc).isoformat(),  # Convert datetime to string
        "post_url": url,
        "user_id": user_id,
        "engagement": Engagement().dict(),  # Ensure correct format
        "analysis": {
            **Analysis().dict(),
            "scam_type": scam_type,
            "scam_framing2": scam_framing
        },
        "batch": new_batch_id
    }

    # Insert the new post into the database
    await db.posts.insert_one(new_post)
    print(scam_type, scam_framing)

    return Post(**new_post)

async def add_new_comment(post_id: int, comment_content: str) -> Dict:
    """
    Add a new comment to the comments collection.
    Classify the sentiment of the comment using Hugging Face and store it.
    """
    # Get the original post
    post_data = await db.posts.find_one({"post_id": post_id, "deleted": {"$ne": 1}})
    if not post_data:
        raise ValueError(f"Post with post_id {post_id} does not exist.")
    largest_comment = await db.comments.find_one(sort=[("comment_id", -1)])
    new_comment_id = largest_comment["comment_id"] + 1 if largest_comment else 1
    post_object_id = post_data["_id"]

    # Use Hugging Face pipeline
    sentiment_result = classify_comment_sentiment(comment_content)
    sentiment = sentiment_result["sentiment"]

    # Create the new comment document
    new_comment = {
        "post_id": post_object_id,
        "comment_id": new_comment_id,
        "comments": [
            {
                "comment_content": comment_content,
                "sentiment_analysis2": sentiment,
            }
        ]
    }
    # Save to MongoDB
    result = await db.comments.insert_one(new_comment)
    print(new_comment)
    new_comment["post_id"] = post_id 

    # Get the current likes value and convert it to an integer
    current_comment_count = post_data.get("engagement", {}).get("comment_count", "0")
    if not isinstance(current_comment_count, (int, float)):
        try:
            current_comment_count = int(current_comment_count)
        except ValueError:
            raise ValueError(f"Invalid 'comment_count' value for post_id {post_id}: {current_comment_count}")

    # Increment or decrement the likes count
    updated_current_comment_count = current_comment_count + 1 

    # Convert the updated likes back to a string and update the database
    result = await db.posts.update_one(
        {"post_id": post_id},
        {"$set": {"engagement.comment_count": str(updated_current_comment_count)}}
    )

    return new_comment

async def update_post(post_id: int, user_id: int, post_title: Optional[str] = None, post_content: Optional[str] = None, url: Optional[str] = None) -> Optional[Post]:
    update_data = {}
    if post_title is not None:
        update_data["post_title"] = post_title
    if post_content is not None:
        update_data["content"] = post_content
    if url is not None:
        update_data["post_url"] = url
    # print(update_data)
    print(f"Querying with post_id={post_id} ({type(post_id)})")

    if not update_data:
        raise ValueError("No data provided to update")

    result = await db.posts.find_one_and_update(
        {"post_id": post_id},
        {"$set": update_data},
        return_document=ReturnDocument.AFTER
    )

    print(f"Updated document: {result}")  # Debugging

    if result is None:
        return None
    # Normalize datetime fields to string
    if isinstance(result.get("date"), datetime):
        result["date"] = result["date"].isoformat()
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

async def get_scam_type_engagement_counts(user_id: int) -> Dict[str, Dict[str, int]]:
    user_id_filter = get_user_id_filter(user_id)
    pipeline = [
        {"$match": user_id_filter},
        {
            "$group": {
                "_id": "$analysis.scam_type",
                "likes": {
                    "$sum": {
                        "$convert": {
                            "input": "$engagement.likes",
                            "to": "int",
                            "onError": 0,
                            "onNull": 0
                        }
                    }
                },
                "shares": {
                    "$sum": {
                        "$convert": {
                            "input": "$engagement.shares",
                            "to": "int",
                            "onError": 0,
                            "onNull": 0
                        }
                    }
                },
                "comments": {
                    "$sum": {
                        "$convert": {
                            "input": "$engagement.comment_count",
                            "to": "int",
                            "onError": 0,
                            "onNull": 0
                        }
                    }
                }
            }
        }
    ]
    
    results = await db.posts.aggregate(pipeline).to_list(length=None)
    scam_type_engagement = {}
    for item in results:
        scam_type = item["_id"] if item["_id"] else "Unknown"
        scam_type_engagement[scam_type] = {
            "likes": item.get("likes", 0),
            "shares": item.get("shares", 0),
            "comments": item.get("comments", 0)
        }
    return scam_type_engagement

async def get_scam_framing_engagement_counts(user_id: int) -> Dict[str, Dict[str, int]]:
    user_id_filter = get_user_id_filter(user_id)
    pipeline = [
        {"$match": user_id_filter},
        {
            "$group": {
                "_id": "$analysis.scam_framing2",
                "likes": {
                    "$sum": {
                        "$convert": {
                            "input": "$engagement.likes",
                            "to": "int",
                            "onError": 0,
                            "onNull": 0
                        }
                    }
                },
                "shares": {
                    "$sum": {
                        "$convert": {
                            "input": "$engagement.shares",
                            "to": "int",
                            "onError": 0,
                            "onNull": 0
                        }
                    }
                },
                "comments": {
                    "$sum": {
                        "$convert": {
                            "input": "$engagement.comment_count",
                            "to": "int",
                            "onError": 0,
                            "onNull": 0
                        }
                    }
                }
            }
        }
    ]
    
    results = await db.posts.aggregate(pipeline).to_list(length=None)
    scam_framing_engagement = {}
    for item in results:
        scam_type = item["_id"] if item["_id"] else "Unknown"
        scam_framing_engagement[scam_type] = {
            "likes": item.get("likes", 0),
            "shares": item.get("shares", 0),
            "comments": item.get("comments", 0)
        }
    return scam_framing_engagement

async def count_posts_by_scam_type_and_sentiment(user_id: int) -> Dict[str, Dict[str, int]]:
    user_id_filter = get_user_id_filter(user_id)
    adjusted_filter = {f"post.{key}": value for key, value in user_id_filter.items()}

    pipeline = [
        {
            "$lookup": {
                "from": "posts",
                "localField": "post_id",
                "foreignField": "_id",
                "as": "post"
            }
        },
        {
            "$unwind": "$post"
        },
        {
            "$match": adjusted_filter
        },
        {
            "$unwind": "$comments"
        },
        {
            "$addFields": {
                "scam_type": "$post.analysis.scam_type",
                "sentiment": {
                    "$ifNull": ["$comments.sentiment_analysis2", "neutral"]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "scam_type": "$scam_type",
                    "sentiment": "$sentiment"
                },
                "count": {"$sum": 1}
            }
        },
        {
            "$group": {
                "_id": "$_id.scam_type",
                "sentiments": {
                    "$push": {
                        "sentiment": "$_id.sentiment",
                        "count": "$count"
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "scam_type": "$_id",  # Rename _id to scam_type
                "sentiments": {
                    "$arrayToObject": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$sentiments",
                                    "as": "item",
                                    "cond": {"$and": [
                                        {"$ne": ["$$item.sentiment", None]},
                                        {"$ne": ["$$item.count", None]}
                                    ]}
                                }
                            },
                            "as": "item",
                            "in": {"k": "$$item.sentiment", "v": "$$item.count"}
                        }
                    }
                }
            }
        }
    ]

    results = await db.comments.aggregate(pipeline).to_list(None)

    # Check for empty results
    if not results:
        print("Pipeline returned no results")
        return {}

    # Format into nested dict structure
    scam_type_counts = {}
    for doc in results:
        scam_type = doc["scam_type"] or "Unknown"
        scam_type_counts[scam_type] = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }

        # Validate that "sentiments" is a list
        if not isinstance(doc["sentiments"], dict):
            print(f"Unexpected 'sentiments' format: {doc['sentiments']}")  # Debugging
            continue

        for sentiment, count in doc["sentiments"].items():
            label = sentiment.lower()
            if label in scam_type_counts[scam_type]:
                scam_type_counts[scam_type][label] = count

    return scam_type_counts 

async def count_posts_by_scam_framing_and_sentiment(user_id: int) -> Dict[str, Dict[str, int]]:
    user_id_filter = get_user_id_filter(user_id)
    adjusted_filter = {f"post.{key}": value for key, value in user_id_filter.items()}

    pipeline = [
        {
            "$lookup": {
                "from": "posts",
                "localField": "post_id",
                "foreignField": "_id",
                "as": "post"
            }
        },
        {
            "$unwind": "$post"
        },
        {
            "$match": adjusted_filter
        },
        {
            "$unwind": "$comments"
        },
        {
            "$addFields": {
                "scam_framing": "$post.analysis.scam_framing2",
                "sentiment": {
                    "$ifNull": ["$comments.sentiment_analysis2", "neutral"]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "scam_framing": "$scam_framing",
                    "sentiment": "$sentiment"
                },
                "count": {"$sum": 1}
            }
        },
        {
            "$group": {
                "_id": "$_id.scam_framing",
                "sentiments": {
                    "$push": {
                        "sentiment": "$_id.sentiment",
                        "count": "$count"
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "scam_framing": "$_id",  # Rename _id to scam_type
                "sentiments": {
                    "$arrayToObject": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$sentiments",
                                    "as": "item",
                                    "cond": {"$and": [
                                        {"$ne": ["$$item.sentiment", None]},
                                        {"$ne": ["$$item.count", None]}
                                    ]}
                                }
                            },
                            "as": "item",
                            "in": {"k": "$$item.sentiment", "v": "$$item.count"}
                        }
                    }
                }
            }
        }
    ]

    results = await db.comments.aggregate(pipeline).to_list(None)

    # Check for empty results
    if not results:
        print("Pipeline returned no results")
        return {}

    # Format into nested dict structure
    scam_framing_counts = {}
    for doc in results:
        scam_framing = doc["scam_framing"] or "None"
        scam_framing_counts[scam_framing] = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }

        # Validate that "sentiments" is a list
        if not isinstance(doc["sentiments"], dict):
            print(f"Unexpected 'sentiments' format: {doc['sentiments']}")  # Debugging
            continue

        for sentiment, count in doc["sentiments"].items():
            label = sentiment.lower()
            if label in scam_framing_counts[scam_framing]:
                scam_framing_counts[scam_framing][label] = count

    return scam_framing_counts
    user_id_filter = get_user_id_filter(user_id)
    adjusted_filter = {f"post.{key}": value for key, value in user_id_filter.items()}

    pipeline = [
        {
            "$lookup": {
                "from": "posts",
                "localField": "post_id",
                "foreignField": "_id",
                "as": "post"
            }
        },
        {
            "$unwind": "$post"
        },
        {
            "$match": adjusted_filter
        },
        {
            "$unwind": "$comments"
        },
        {
            "$addFields": {
                "scam_type": "$post.analysis.scam_type",
                "sentiment": {
                    "$ifNull": ["$comments.sentiment_analysis2", "neutral"]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "scam_type": "$scam_type",
                    "sentiment": "$sentiment"
                },
                "count": {"$sum": 1}
            }
        },
        {
            "$group": {
                "_id": "$_id.scam_type",
                "sentiments": {
                    "$push": {
                        "sentiment": "$_id.sentiment",
                        "count": "$count"
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "scam_type": "$_id",
                "sentiments": {
                    "$arrayToObject": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$sentiments",
                                    "as": "item",
                                    "cond": {"$and": [
                                        {"$ne": ["$$item.sentiment", None]},
                                        {"$ne": ["$$item.count", None]}
                                    ]}
                                }
                            },
                            "as": "item",
                            "in": {"k": "$$item.sentiment", "v": "$$item.count"}
                        }
                    }
                }
            }
        }
    ]

    results = await db.comments.aggregate(pipeline).to_list(None)

    # Format into nested dict structure
    scam_type_counts = {}
    for doc in results:
        scam_type = doc["scam_type"] or "Unknown"
        scam_type_counts[scam_type] = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
        for s in doc["sentiments"]:
            label = s["sentiment"].lower()
            if label in scam_type_counts[scam_type]:
                scam_type_counts[scam_type][label] = s["count"]

    return scam_type_counts 

    user_id_filter = get_user_id_filter(user_id)
    # Adjust the filter to reference post.user_id
    adjusted_filter = {
        f"post.{key}": value for key, value in user_id_filter.items()
    }
    pipeline = [
        {
            "$lookup": {
                "from": "posts",
                "localField": "post_id",
                "foreignField": "_id",
                "as": "post"
            }
        },
        {
            "$unwind": "$post"  # Unwind the joined post array
        },
        {
            "$match": adjusted_filter
        },
        {
            "$unwind": "$comments"  # Unwind the nested comments array
        },
        {
            "$addFields": {
                "scam_type": "$post.analysis.scam_type",  # Extract scam_type from the post
                "sentiment": {
                    "$ifNull": ["$comments.sentiment_analysis2", "neutral"]  # Extract sentiment_analysis2 from comments
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "scam_type": "$scam_type",
                    "sentiment": "$sentiment"
                },
                "count": {"$sum": 1}  # Count the number of comments for each scam_type and sentiment
            }
        },
        {
            "$group": {
                "_id": "$_id.scam_type",  # Group by scam_type
                "sentiments": {
                    "$push": {
                        "sentiment": "$_id.sentiment",
                        "count": "$count"
                    }
                }
            }
        },
        {
            "$project": {
                "_id": 0,
                "scam_type": "$_id",  # Rename _id to scam_type
                "sentiments": {
                    "$arrayToObject": {
                        "$map": {
                            "input": {
                                "$filter": {
                                    "input": "$sentiments",
                                    "as": "item",
                                    "cond": {"$and": [
                                        {"$ne": ["$$item.sentiment", None]},  # Ensure sentiment is not null
                                        {"$ne": ["$$item.count", None]}  # Ensure count is not null
                                    ]}
                                }
                            },
                            "as": "item",
                            "in": {"k": "$$item.sentiment", "v": "$$item.count"}
                        }
                    }
                }
            }
        }
    ]
    results = await db.comments.aggregate(pipeline).to_list(None)
    # print("Pipeline Stage Results:", results)
    
    #     # Step 1: Perform the $lookup
    # lookup_stage = [
    #     {
    #         "$lookup": {
    #             "from": "posts",
    #             "localField": "post_id",
    #             "foreignField": "_id",
    #             "as": "post"
    #         }
    #     }
    # ]
    # lookup_results = await db.comments.aggregate(lookup_stage).to_list(None)
    # print("After $lookup:", lookup_results)
    
    # # Step 2: Unwind the "post" array
    # unwind_post_stage = [
    #     {
    #         "$unwind": "$post"
    #     }
    # ]
    # unwind_post_results = await db.comments.aggregate(lookup_stage + unwind_post_stage).to_list(None)
    # # Step 3: Apply the $match stage
    # match_stage = [
    #     {
    #         "$match": adjusted_filter
    #     }
    # ]
    # match_results = await db.comments.aggregate(lookup_stage + unwind_post_stage + match_stage).to_list(None)
    
    # print("After $match:", match_results)

    # Format into nested dict structure
    scam_type_counts = {}
    for doc in results:
        scam_type = doc["_id"] or "Unknown"
        scam_type_counts[scam_type] = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
        for s in doc["sentiments"]:
            label = s["sentiment"].lower()
            if label in scam_type_counts[scam_type]:
                scam_type_counts[scam_type][label] = s["count"]

    return scam_type_counts

    user_id_filter = get_user_id_filter(user_id)
    pipeline = [
        {
            "$match": {
                "user_id": user_id_filter
            }
        },
        {
            "$lookup": {
                "from": "comments",
                "localField": "_id",
                "foreignField": "post_id",
                "as": "comments"
            }
        },
        {
            "$unwind": "$comments"
        },
        {
            "$addFields": {
                "sentiment": {
                    "$ifNull": ["$comments.analysis.sentiment_analysis", "neutral"]
                }
            }
        },
        {
            "$group": {
                "_id": {
                    "scam_type": "$analysis.scam_type",
                    "sentiment": "$sentiment"
                },
                "count": {"$sum": 1}
            }
        },
        {
            "$group": {
                "_id": "$_id.scam_type",
                "sentiments": {
                    "$push": {
                        "sentiment": "$_id.sentiment",
                        "count": "$count"
                    }
                }
            }
        }
    ]

    results = await db.posts.aggregate(pipeline).to_list(None)

    # Format into nested dict structure
    scam_type_counts = {}
    for doc in results:
        scam_type = doc["_id"] or "Unknown"
        scam_type_counts[scam_type] = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }
        for s in doc["sentiments"]:
            label = s["sentiment"].lower()
            if label in scam_type_counts[scam_type]:
                scam_type_counts[scam_type][label] = s["count"]

    return scam_type_counts

    """
    Count posts grouped by scam_type and sentiment_analysis2 for a given user_id.
    Returns a nested dictionary where the outer keys are scam types and the inner keys are sentiment labels.
    """
    user_id_filter = get_user_id_filter(user_id)
    
    # Fetch all posts by user_id to get post_ids
    posts = await db.posts.find(user_id_filter).to_list(length=None)
    post_ids = [post['_id'] for post in posts]  # Use ObjectId for joining with comments

    pipeline = [
        {"$match": {"post_id": {"$in": post_ids}}},  # Match comments for the user's posts
        {"$unwind": "$comments"},  # Unwind the comments array
        {"$group": {
            "_id": {
                "scam_type": "$analysis.scam_type",  # Group by scam_type
                "sentiment": "$comments.sentiment_analysis2"  # Group by sentiment_analysis2
            },
            "count": {"$sum": 1}  # Count the number of comments
        }},
        {"$group": {
            "_id": "$_id.scam_type",  # Group by scam_type
            "sentiments": {
                "$push": {
                    "sentiment": "$_id.sentiment",
                    "count": "$count"
                }
            }
        }},
        {"$project": {
            "_id": 0,
            "scam_type": "$_id",
            "sentiments": {
                "$arrayToObject": {
                    "$map": {
                        "input": {
                            "$filter": {
                                "input": "$sentiments",
                                "as": "item",
                                "cond": {"$and": [
                                    {"$ne": ["$$item.sentiment", None]},  # Ensure sentiment is not null
                                    {"$ne": ["$$item.count", None]}  # Ensure count is not null
                                ]}
                            }
                        },
                        "as": "item",
                        "in": {"k": "$$item.sentiment", "v": "$$item.count"}
                    }
                }
            }
        }}
    ]
    result = await db.comments.aggregate(pipeline).to_list(length=None)

    # Convert the result into a more usable dictionary format
    scam_type_sentiment_counts = {}
    for item in result:
        scam_type = item["scam_type"] if item["scam_type"] else "Unknown"
        scam_type_sentiment_counts[scam_type] = item["sentiments"]

    return scam_type_sentiment_counts

async def post_to_facebook_by_post_id(post_id: int) -> dict:
    """
    Retrieve a post by post_id and share its title and content to a Facebook Page.
    """
    # Fetch the post from the database
    post = await db.posts.find_one({"post_id": post_id, "deleted": {"$ne": 1}})
    if not post:
        raise ValueError(f"Post with post_id {post_id} not found.")

    # Compose the message: include title and content (and optionally URL)
    post_title = post.get("post_title", "")
    content = post.get("content", "")
    url = post.get("post_url", "")
    message = f"{post_title}\n\n{content}"
    if url:
        message += f"\n\n{url}"

    PAGE_ID = os.getenv("FB_PAGE_ID")
    PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_TOKEN")    

    # Post to Facebook Page
    fb_url = f"https://graph.facebook.com/{PAGE_ID}/feed"
    payload = {
        "message": message,
        "access_token": PAGE_ACCESS_TOKEN
    }
    response = requests.post(fb_url, data=payload)
    result = response.json()

    shared_post_url = None
    if response.status_code == 200 and "id" in result:
        # Get current shares value and increment
        current_shares = post.get("engagement", {}).get("shares", 0)
        try:
            current_shares = int(current_shares)
        except Exception:
            current_shares = 0
        new_shares = current_shares + 1

        await db.posts.update_one(
            {"post_id": post_id},
            {"$set": {"engagement.shares": new_shares}}
        )

        # Construct the shared post URL
        fb_id = result["id"]
        page_id, post_id_fb = fb_id.split("_")
        shared_post_url = f"https://www.facebook.com/{page_id}/posts/{post_id_fb}"

    return {
        "success": response.status_code == 200 and "id" in result,
        "facebook_response": result,
        "shared_post_url": shared_post_url
    }

def parse_count(value):
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        value = value.strip().upper()
        if value.endswith("K"):
            try:
                return int(float(value[:-1]) * 1000)
            except ValueError:
                return 0
        elif value.endswith("M"):
            try:
                return int(float(value[:-1]) * 1000000)
            except ValueError:
                return 0
        try:
            return int(value)
        except ValueError:
            return 0
    return 0

async def get_post_metrics(user_id: int) -> dict:
    """
    Return platform counts, engagement metrics, and sentiment analysis percentages
    for posts filtered by user_id (using get_user_id_filter).
    """
    user_id_filter = get_user_id_filter(user_id)
    posts = await db.posts.find(user_id_filter).to_list(length=None)
    post_object_ids = [post["_id"] for post in posts]

    # Only fetch comments for these posts
    comments = await db.comments.find({"post_id": {"$in": post_object_ids}}).to_list(length=None)

    # --- PLATFORM COUNT LOGIC ---
    platform_counts = {
        "official_website": len([p for p in posts if 1 <= p.get("user_id", 0) <= 5]),
        "twitter": len([p for p in posts if 6 <= p.get("user_id", 0) <= 10]),
        "facebook": len([p for p in posts if 11 <= p.get("user_id", 0) <= 15]),
    }

    # --- ENGAGEMENT METRICS ---
    total_likes = sum(parse_count(p.get("engagement", {}).get("likes", 0)) for p in posts)
    total_shares = sum(parse_count(p.get("engagement", {}).get("shares", 0)) for p in posts)
    total_comments = sum(parse_count(p.get("engagement", {}).get("comment_count", 0)) for p in posts)
    total_engagement = total_likes + total_shares + total_comments

    # --- SENTIMENT METRICS ---
    sentiment_counter = Counter()
    total_sentiments = 0

    # From posts
    for post in posts:
        sentiment = post.get("sentiment_analysis")
        if sentiment:
            sentiment_counter[sentiment.lower()] += 1
            total_sentiments += 1

    # From subcomments in comment documents
    for doc in comments:
        for c in doc.get("comments", []):
            sentiment = c.get("sentiment_analysis") or c.get("sentiment_analysis2")
            if sentiment:
                sentiment_counter[sentiment.lower()] += 1
                total_sentiments += 1

    # Calculate percentage
    sentiment_percentages = {
        "positive": round(100 * sentiment_counter.get("positive", 0) / total_sentiments, 2) if total_sentiments else 0.0,
        "neutral": round(100 * sentiment_counter.get("neutral", 0) / total_sentiments, 2) if total_sentiments else 0.0,
        "negative": round(100 * sentiment_counter.get("negative", 0) / total_sentiments, 2) if total_sentiments else 0.0,
    }

    return {
        "platform_counts": platform_counts,
        "engagement": {
            "likes": total_likes,
            "shares": total_shares,
            "comments": total_comments,
            "total": total_engagement,
        },
        "sentiment_analysis": sentiment_percentages
    }
    """
    Return platform counts, engagement metrics, and sentiment analysis percentages
    for posts filtered by user_id (using get_user_id_filter).
    """
    user_id_filter = get_user_id_filter(user_id)
    posts = await db.posts.find(user_id_filter).to_list(length=None)
    post_object_ids = [post["_id"] for post in posts]

    # Only fetch comments for these posts
    comments = await db.comments.find({"post_id": {"$in": post_object_ids}}).to_list(length=None)

    # --- PLATFORM COUNT LOGIC ---
    platform_counts = {
        "official_website": len([p for p in posts if 1 <= p.get("user_id", 0) <= 5]),
        "twitter": len([p for p in posts if 6 <= p.get("user_id", 0) <= 10]),
        "facebook": len([p for p in posts if 11 <= p.get("user_id", 0) <= 15]),
    }

    # --- ENGAGEMENT METRICS ---
    total_likes = sum(int(p.get("engagement", {}).get("likes", 0)) for p in posts)
    total_shares = sum(int(p.get("engagement", {}).get("shares", 0)) for p in posts)
    total_comments = sum(int(p.get("engagement", {}).get("comment_count", 0)) for p in posts)
    total_engagement = total_likes + total_shares + total_comments

    # --- SENTIMENT METRICS ---
    sentiment_counter = Counter()
    total_sentiments = 0

    # From posts
    for post in posts:
        sentiment = post.get("sentiment_analysis")
        if sentiment:
            sentiment_counter[sentiment.lower()] += 1
            total_sentiments += 1

    # From subcomments in comment documents
    for doc in comments:
        for c in doc.get("comments", []):
            sentiment = c.get("sentiment_analysis") or c.get("sentiment_analysis2")
            if sentiment:
                sentiment_counter[sentiment.lower()] += 1
                total_sentiments += 1

    # Calculate percentage
    sentiment_percentages = {
        "positive": round(100 * sentiment_counter.get("positive", 0) / total_sentiments, 2) if total_sentiments else 0.0,
        "neutral": round(100 * sentiment_counter.get("neutral", 0) / total_sentiments, 2) if total_sentiments else 0.0,
        "negative": round(100 * sentiment_counter.get("negative", 0) / total_sentiments, 2) if total_sentiments else 0.0,
    }

    return {
        "platform_counts": platform_counts,
        "engagement": {
            "likes": total_likes,
            "shares": total_shares,
            "comments": total_comments,
            "total": total_engagement,
        },
        "sentiment_analysis": sentiment_percentages
    }