from fastapi import APIRouter, Body, HTTPException, Query
from typing import List, Dict, Optional
from ..schemas import Post, Comment
from ..controllers.post_controller import create_new_post, fetch_comments_by_post_id, fetch_post_count_by_user_id_group_by_scam_type_and_framing, fetch_posts_by_user_id, fetch_post_count_by_user_id, fetch_post_count_by_user_id_group_by_scam_type, fetch_post_count_by_user_id_group_by_platform, fetch_post_by_id, fetch_sentiment_analysis_by_user_id, get_scam_framing_counts, mark_post_as_deleted_controller, update_post_controller

router = APIRouter()

@router.get("/posts/user/{user_id}", response_model=List[Post])
async def get_posts(user_id: int, platform: Optional[str] = Query(None), scam_framing: Optional[str] = Query(None), scam_type: Optional[str] = Query(None)):
    return await fetch_posts_by_user_id(user_id, platform, scam_framing, scam_type)

@router.get("/posts/user/{user_id}/count", response_model=int)
async def get_post_count(user_id: int):
    return await fetch_post_count_by_user_id(user_id)

@router.get("/posts/user/{user_id}/count_by_scam_type", response_model=Dict[str, int])
async def get_post_count_by_scam_type(user_id: int):
    return await fetch_post_count_by_user_id_group_by_scam_type(user_id)

@router.get("/posts/user/{user_id}/count_by_platform", response_model=Dict[str, int])
async def get_post_count_by_platform(user_id: int):
    return await fetch_post_count_by_user_id_group_by_platform(user_id)

@router.get("/posts/{post_id}", response_model=Post)
async def get_post(post_id: int):
    return await fetch_post_by_id(post_id)

@router.get("/posts/{post_id}/comments", response_model=List[Comment])
async def get_comments(post_id: int):
    return await fetch_comments_by_post_id(post_id)

@router.get("/posts/user/{user_id}/sentiment_analysis", response_model=Dict[str, float])
async def get_sentiment_analysis(user_id: int):
    return await fetch_sentiment_analysis_by_user_id(user_id)

@router.post("/posts", response_model=Post)
async def create_post(
    user_id: int = Body(...),
    post_content: str = Body(...),
    post_title: Optional[str] = Body(None),
    url: Optional[str] = Body(None)
):
    try:
        new_post = await create_new_post(post_title, post_content, user_id, url)
        return new_post
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/posts/{post_id}", response_model=Post)
async def update_post_endpoint(
    post_id: int,
    user_id: int = Body(...),
    post_title: Optional[str] = Body(None),
    post_content: Optional[str] = Body(None),
    url: Optional[str] = Body(None)
):
    try:
        updated_post = await update_post_controller(post_id, user_id, post_title, post_content, url)
        return updated_post
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/posts/{post_id}", response_model=Post)
async def delete_post_endpoint(post_id: int, user_id: int = Query(...)):
    try:
        deleted_post = await mark_post_as_deleted_controller(post_id, user_id)
        if deleted_post is None:
            raise HTTPException(status_code=404, detail="Post not found or user not authorized")
        return deleted_post
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/posts/{user_id}/scam_framing_counts", response_model=Dict[str, int])
async def get_scam_framing_counts_route(user_id: int):
    """
    API endpoint to get scam_framing counts for a given user_id.
    """
    return await get_scam_framing_counts(user_id)
