from motor.motor_asyncio import AsyncIOMotorClient
from .config import Settings

settings = Settings()
client = AsyncIOMotorClient(settings.mongodb_url)

# Connect to the database named 'scam_db'
db = client.scam_db2