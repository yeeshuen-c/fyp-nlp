import asyncio
import os
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.context import CryptContext
import pandas as pd

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

async def update_user_passwords():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["scam_db2"]
    users = await db.users.find().to_list(length=None)
    for user in users:
        if not user["password"].startswith("$2b$"):  # Check if the password is not hashed
            hashed_password = pwd_context.hash(user["password"])
            await db.users.update_one({"_id": user["_id"]}, {"$set": {"password": hashed_password}})
    print("Passwords updated successfully.")

async def export_post_labelling():
    # Connect to MongoDB
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["scam_db2"]

    # Fetch posts and users data
    posts = await db.posts.find().to_list(length=None)
    users = await db.users.find().to_list(length=None)

    # Create a dictionary for quick lookup of user details by user_id
    user_dict = {user["user_id"]: user for user in users}

    # Prepare data for Excel
    data = []
    for post in posts:
        user_id = post.get("user_id")
        user = user_dict.get(user_id, {})
        agency_name = user.get("agencies_name", "N/A")
        platform_name = user.get("platform", {}).get("platform_name", "N/A")
        scam_type = post.get("analysis", {}).get("scam_type", "N/A")
        scam_framing = post.get("analysis", {}).get("scam_framing", "N/A")

        data.append({
            "post_id": post.get("post_id", "N/A"),
            "content": post.get("content", "N/A"),
            "user_id": user_id,
            "agency name": agency_name,
            "platform name": platform_name,
            "scam type": scam_type,
            "scam prevention framing": scam_framing
        })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Ensure the 'excel' directory exists
    output_dir = "excel"
    os.makedirs(output_dir, exist_ok=True)

    # Export to Excel
    output_file = os.path.join(output_dir, "post_labelling.xlsx")

    df.to_excel(output_file, index=False)
    print(f"Data exported successfully to {output_file}")


def compare_excel_files():
    # File paths
    excel_dir = "excel"
    file1 = os.path.join(excel_dir, "post_labelling.xlsx")
    file2 = os.path.join(excel_dir, "scam_prevention_posts (osb).xlsx")
    output_file = os.path.join(excel_dir, "difference.xlsx")

    # Load the Excel files
    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    # Normalize the columns for comparison (ignore case and strip whitespace)
    df1["scam type"] = df1["scam type"].str.lower().str.strip()
    df1["scam prevention framing"] = df1["scam prevention framing"].str.lower().str.strip()
    df1["content"] = df1["content"].str.lower().str.strip()
    df2["scam type"] = df2["scam type"].str.lower().str.strip()
    df2["scam prevention framing"] = df2["scam prevention framing"].str.lower().str.strip()
    df2["content"] = df2["content"].str.lower().str.strip()

    # Merge the two DataFrames on `content` to align rows
    merged_df = pd.merge(df1, df2, on="content", suffixes=("_file1", "_file2"))

    # Find rows with differences in `scam type` or `scam prevention framing`
    differences = merged_df[
        (merged_df["scam type_file1"] != merged_df["scam type_file2"]) |
        (merged_df["scam prevention framing_file1"] != merged_df["scam prevention framing_file2"])
    ]

    # Select relevant columns for the output
    output_df = differences[[
        "content",
        "scam type_file1", "scam prevention framing_file1",
        "scam type_file2", "scam prevention framing_file2"
    ]]

    # Save the differences to a new Excel file
    output_df.to_excel(output_file, index=False)
    print(f"Differences exported successfully to {output_file}")

if __name__ == "__main__":
    # asyncio.run(update_user_passwords())
    # asyncio.run(export_post_labelling())
    compare_excel_files() 