from typing import List, Dict, Optional
from fastapi import HTTPException
from ..schemas import Post, Comment
from ..services.post_service import add_new_post, count_posts_by_user_id_group_by_scam_type_and_framing, get_comments_by_post_id, get_posts_by_user_id, count_posts_by_user_id, count_posts_by_user_id_group_by_scam_type, count_posts_by_user_id_group_by_platform,get_post_by_id, get_sentiment_analysis_by_user_id, mark_post_as_deleted, update_post

async def fetch_posts_by_user_id(user_id: int, platform: Optional[str] = None, scam_framing: Optional[str] = None, scam_type: Optional[str] = None) -> List[Post]:
    return await get_posts_by_user_id(user_id, platform, scam_framing, scam_type)

async def fetch_post_count_by_user_id(user_id: int) -> int:
    count = await count_posts_by_user_id(user_id)
    return count

async def fetch_post_count_by_user_id_group_by_scam_type(user_id: int) -> Dict[str, int]:
    count_by_scam_type = await count_posts_by_user_id_group_by_scam_type(user_id)
    return count_by_scam_type

async def fetch_post_count_by_user_id_group_by_platform(user_id: int) -> Dict[str, int]:
    count_by_platform = await count_posts_by_user_id_group_by_platform(user_id)
    return count_by_platform

async def fetch_post_count_by_user_id_group_by_scam_type_and_framing(user_id: int) -> Dict[str, any]:
    return await count_posts_by_user_id_group_by_scam_type_and_framing(user_id)

async def fetch_post_by_id(post_id: int) -> Post:
    return await get_post_by_id(post_id)

async def fetch_comments_by_post_id(post_id: int) -> List[Comment]:
    return await get_comments_by_post_id(post_id)

async def fetch_sentiment_analysis_by_user_id(user_id: int) -> float:
    return await get_sentiment_analysis_by_user_id(user_id)

async def create_new_post(post_title: Optional[str], post_content: str, user_id: int, url: Optional[str]) -> Post:
    new_post = await add_new_post(post_title, post_content, user_id, url)
    return new_post

async def update_post_controller(post_id: int, user_id: int, post_title: Optional[str] = None, post_content: Optional[str] = None, url: Optional[str] = None) -> Post:
    updated_post = await update_post(post_id, user_id, post_title, post_content, url)
    if updated_post is None:
        raise ValueError("Post not found or user not authorized")
    return updated_post

async def mark_post_as_deleted_controller(post_id: int, user_id: int) -> Post:
    deleted_post = await mark_post_as_deleted(post_id, user_id)
    if deleted_post is None:
        raise ValueError("Post not found or user not authorized")
    return deleted_post