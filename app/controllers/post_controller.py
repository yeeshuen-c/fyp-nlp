from typing import List, Dict, Optional
from fastapi import HTTPException
from ..schemas import CommentResponse, Post, Comment
from ..services.post_service import add_new_comment, add_new_post, count_posts_by_scam_framing_and_sentiment, count_posts_by_scam_type_and_sentiment, count_posts_by_user_id_group_by_scam_framing, count_posts_by_user_id_group_by_scam_type_and_framing, delete_comment_by_id, get_combined_comments_by_post_id, get_comments_by_post_id, get_posts_by_user_id, count_posts_by_user_id, count_posts_by_user_id_group_by_scam_type, count_posts_by_user_id_group_by_platform,get_post_by_id, get_scam_framing_engagement_counts, get_scam_type_engagement_counts, get_sentiment_analysis_by_user_id, mark_post_as_deleted, post_to_facebook_by_post_id, update_post, update_post_likes

async def fetch_posts_by_user_id(
    user_id: int,
    platform: Optional[str] = None,
    scam_framing: Optional[str] = None,
    scam_type: Optional[str] = None,
    offset: int = 0,
    limit: int = 10,
    agency: Optional[str] = None
) -> List[Post]:
    return await get_posts_by_user_id(user_id, platform, scam_framing, scam_type, offset, limit,agency)

async def fetch_post_count_by_user_id(user_id: int) -> int:
    # if user_id > 15:
    #     return {"user_id": user_id, "post_count": 0}
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

async def fetch_comments_by_post_id(post_id: int) -> Dict:
    return await get_comments_by_post_id(post_id)

async def fetch_combined_comments_by_post_id(post_id: int) -> Dict:
    return await get_combined_comments_by_post_id(post_id)

async def fetch_sentiment_analysis_by_user_id(user_id: int) -> Dict[str, float]:
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

async def get_scam_framing_counts(user_id: int) -> Dict[str, int]:
    """
    Controller function to get scam_framing counts for a given user_id.
    """
    try:
        scam_framing_counts = await count_posts_by_user_id_group_by_scam_framing(user_id)
        return scam_framing_counts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching scam_framing counts: {str(e)}")
    

async def get_scam_type_and_sentiment_counts(user_id: int) -> Dict[str, Dict[str, int]]:
    """
    Controller to handle the request for scam type and sentiment counts.
    """
    try:
        # Call the service function
        result = await count_posts_by_scam_type_and_sentiment(user_id)
        return result
    except Exception as e:
        # Handle errors and return a 500 response
        raise HTTPException(status_code=500, detail=str(e))
    
async def get_scam_framing_and_sentiment_counts(user_id: int) -> Dict[str, Dict[str, int]]:
    """
    Controller to handle the request for scam type and sentiment counts.
    """
    try:
        # Call the service function
        result = await count_posts_by_scam_framing_and_sentiment(user_id)
        return result
    except Exception as e:
        # Handle errors and return a 500 response
        raise HTTPException(status_code=500, detail=str(e))
    
async def create_comment(post_id: int, comment_content: str) -> CommentResponse:
    """
    Controller function to handle adding a new comment.
    """
    try:
        # Call the service function to add the comment
        new_comment = await add_new_comment(post_id, comment_content)
        return CommentResponse(**new_comment)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred while adding the comment.")

async def delete_comment(comment_id: int) -> dict:
    """
    Controller function to delete a comment by its comment_id.
    """
    success = await delete_comment_by_id(comment_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Comment with comment_id {comment_id} not found.")
    return {"message": f"Comment with comment_id {comment_id} has been deleted successfully."}

# async def like_post(post_id: int) -> dict:
#     """
#     Controller function to increment the likes count for a post.
#     """
#     success = await increment_post_likes(post_id)
#     if not success:
#         raise HTTPException(status_code=404, detail=f"Post with post_id {post_id} not found.")
#     return {"message": f"Likes count for post_id {post_id} has been incremented."}

async def update_post_likes_controller(post_id: int, increment: bool) -> dict:
    """
    Controller function to update the likes count for a post.
    """
    try:
        success = await update_post_likes(post_id, increment)
        action = "incremented" if increment else "decremented"
        if success:
            return {"message": f"Likes count for post_id {post_id} has been {action}."}
        else:
            raise HTTPException(status_code=404, detail=f"Post with post_id {post_id} not found.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

async def fetch_scam_type_engagement_counts(user_id: int) -> Dict[str, Dict[str, int]]:
    """
    Controller to get scam type engagement counts.
    """
    result = await get_scam_type_engagement_counts(user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No engagement data found.")
    return result

async def fetch_scam_framing_engagement_counts(user_id: int) -> Dict[str, Dict[str, int]]:
    """
    Controller to get scam framing engagement counts.
    """
    result = await get_scam_framing_engagement_counts(user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No engagement data found.")
    return result

async def share_post_to_facebook_controller(post_id: int):
    try:
        result = await post_to_facebook_by_post_id(post_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"]["message"])
        return {"success": True, "facebook_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))