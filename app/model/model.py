from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid objectid')
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='string')

class Engagement(BaseModel):
    likes: int
    shares: int
    comment_count: int

class Analysis(BaseModel):
    scam_framing: str
    scam_or_not: Optional[str]
    scam_type: str

class Post(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    post_id: int
    post_url: str
    post_title: str
    date: str
    content: str
    media_url: List[str]
    image: Dict[str, Any]
    engagement: Engagement
    analysis: Analysis
    user_id: int
    batch: int
    media: List[Any]

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Platform(BaseModel):
    followers: Optional[int]
    platform_name: str

class User(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: int
    agencies_name: str
    password: str
    platform: Platform

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class CommentContent(BaseModel):
    comment_content: str

class CommentAnalysis(BaseModel):
    sentiment_analysis: Optional[str]

class Comment(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    comment_id: int
    platform: str
    comments: List[CommentContent]
    post_id: PyObjectId = Field(default_factory=PyObjectId, alias="post_id")
    analysis: CommentAnalysis

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}