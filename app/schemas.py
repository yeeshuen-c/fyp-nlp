from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union

class Engagement(BaseModel):
    likes: Optional[Union[str, int]] = "0"
    shares: Optional[Union[str, int]] = "0"
    comment_count: Optional[Union[str, int]] = "0"

    @field_validator('likes', 'shares', 'comment_count', mode='before')
    def parse_engagement(cls, value):
        if isinstance(value, str):
            if 'K' in value:
                return int(float(value.replace('K', '')) * 1000)
            return int(value)
        return value

class Analysis(BaseModel):
    scam_framing: str
    scam_or_not: Optional[str]
    scam_type: str

# class MediaUrl(BaseModel):
#     url: str

# class Media(BaseModel):
#     media_url: str
#     media_file: str

class Post(BaseModel):
    post_id: int
    post_url: Optional[str] = ""
    post_title: str = ""
    date: str
    content: str
    engagement: Engagement
    analysis: Analysis
    user_id: int
    batch: int

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class Platform(BaseModel):
    followers: Optional[str]
    platform_name: str
    url: str

class User(BaseModel):
    user_id: int
    agencies_name: str
    password: str
    platforms: List[Platform]
    username:Optional[str] = ""

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

class CommentContent(BaseModel):
    comment_content: str

class CommentAnalysis(BaseModel):
    sentiment_analysis: Optional[str]

class Comment(BaseModel):
    comment_id: int
    platform: str
    comments: List[CommentContent]
    post_id: int
    analysis: CommentAnalysis

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True