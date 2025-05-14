from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import post_router, user_router, auth_router,classifier_router

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Add your frontend URL here
    # Add other origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(post_router.router, prefix="/api")
app.include_router(user_router.router, prefix="/api")
app.include_router(auth_router.router, prefix="/api")
app.include_router(classifier_router.router, prefix="/api")

@app.get("/api/healthchecker")
async def root():
    return {"message": "Welcome to FastAPI with MongoDB"}