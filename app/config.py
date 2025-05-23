from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    mongodb_url: str
    fb_page_id: str = ""
    fb_page_token: str = ""

    class Config:
        env_file = ".env"