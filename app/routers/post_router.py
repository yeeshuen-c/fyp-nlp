from fastapi import APIRouter
from typing import List, Dict
from ..schemas import Post, Comment
from ..controllers.post_controller import fetch_comments_by_post_id, fetch_post_count_by_user_id_group_by_scam_type_and_framing, fetch_posts_by_user_id, fetch_post_count_by_user_id, fetch_post_count_by_user_id_group_by_scam_type, fetch_post_count_by_user_id_group_by_platform,fetch_post_by_id

router = APIRouter()

@router.get("/posts/user/{user_id}", response_model=List[Post])
async def get_posts(user_id: int):
    return await fetch_posts_by_user_id(user_id)

@router.get("/posts/user/{user_id}/count", response_model=int)
async def get_post_count(user_id: int):
    return await fetch_post_count_by_user_id(user_id)

@router.get("/posts/user/{user_id}/count_by_scam_type", response_model=Dict[str, int])
async def get_post_count_by_scam_type(user_id: int):
    return await fetch_post_count_by_user_id_group_by_scam_type(user_id)

@router.get("/posts/user/{user_id}/count_by_platform", response_model=Dict[str, int])
async def get_post_count_by_platform(user_id: int):
    return await fetch_post_count_by_user_id_group_by_platform(user_id)

# @router.get("/posts/user/{user_id}/count_by_scam_type", response_model=Dict[str, any])
# async def get_post_count_by_scam_type(user_id: int):
#     return await fetch_post_count_by_user_id_group_by_scam_type_and_framing(user_id)

@router.get("/posts/{post_id}", response_model=Post)
async def get_post(post_id: int):
    return await fetch_post_by_id(post_id)

@router.get("/posts/{post_id}/comments", response_model=List[Comment])
async def get_comments(post_id: int):
    return await fetch_comments_by_post_id(post_id)