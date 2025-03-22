from typing import List, Dict
from fastapi import HTTPException
from ..schemas import Post, Comment
from ..services.post_service import count_posts_by_user_id_group_by_scam_type_and_framing, get_comments_by_post_id, get_posts_by_user_id, count_posts_by_user_id, count_posts_by_user_id_group_by_scam_type, count_posts_by_user_id_group_by_platform,get_post_by_id

async def fetch_posts_by_user_id(user_id: int) -> List[Post]:
    posts = await get_posts_by_user_id(user_id)
    if not posts:
        raise HTTPException(status_code=404, detail="Posts not found")
    return posts

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