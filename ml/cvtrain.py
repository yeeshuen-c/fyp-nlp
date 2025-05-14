import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.database import db
from ml.preprocessing import clean_text, fit_vectorizer, transform_text
import asyncio
import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier

# Define scam type keywords
keyword_to_scam_type = {
    "love scam": "love scam",
    "macau scam": "phone scam",
    "job scam": "job scam",
    # Add more mappings as needed
}

# Keyword matching function
def detect_scam_type_by_keywords(text):
    text_lower = text.lower()
    for keyword, scam_type in keyword_to_scam_type.items():
        if keyword in text_lower:
            return scam_type
    return None

# Fetch data from MongoDB
async def get_all_posts():
    """Fetch all posts from the MongoDB collection and return their content and labels."""
    posts_cursor = db.posts.find(
        {"deleted": {"$ne": 1}},  # Exclude deleted posts
        {"content": 1, "analysis.scam_type": 1, "_id": 0}  # Fetch content and scam_type fields
    )
    posts = []
    labels = []
    async for post in posts_cursor:  # Use async for to iterate over the cursor
        if "content" in post and "analysis" in post and "scam_type" in post["analysis"]:
            posts.append(post["content"])
            labels.append(post["analysis"]["scam_type"])
    return posts, labels

# Fetch and preprocess data
async def fetch_and_preprocess_data():
    posts, labels = await get_all_posts()  # Await the async function
    texts = []
    final_labels = []

    for post_text, label in zip(posts, labels):
        cleaned_text = clean_text(post_text)

        # # Apply keyword matching
        # keyword_detected_label = detect_scam_type_by_keywords(cleaned_text)
        # if keyword_detected_label:
        #     final_labels.append(keyword_detected_label)  # Override label with detected one
        # else:
        #     final_labels.append(label)  # Use original label

        final_labels.append(label)  # Use the original label

        texts.append(cleaned_text)

    return texts, final_labels

# Main function
def main():
    # Run the async function using an event loop
    texts, labels = asyncio.run(fetch_and_preprocess_data())

    # Vectorization
    vectorizer = fit_vectorizer(texts)
    X_vec = transform_text(texts)

    # Define models and their hyperparameter grids
    model_params = {
        "SVM": {
            "model": SVC(probability=True),
            "params": {
                "kernel": ["linear", "rbf"],
                "C": [0.1, 1, 10]
            }
        },
        # "KNN": {
        #     "model": KNeighborsClassifier(),
        #     "params": {
        #         "n_neighbors": [3, 5, 7],
        #         "weights": ["uniform", "distance"]
        #     }
        # },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"]
            }
        },
        # "Naive Bayes": {
        #     "model": MultinomialNB(),
        #     "params": {
        #         "alpha": [0.1, 0.5, 1.0]
        #     }
        # },
            "Decision Tree": {
            "model": DecisionTreeClassifier(),
            "params": {
                "criterion": ["gini", "entropy"],  # Splitting criteria
                "max_depth": [None, 10, 20, 30],  # Maximum depth of the tree
                "min_samples_split": [2, 5, 10],  # Minimum samples required to split
                "min_samples_leaf": [1, 2, 4]  # Minimum samples required in a leaf node
            }
        }
    }

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Perform cross-validation with GridSearchCV
    for model_name, config in model_params.items():
        try:
            # Check if a run is already active and end it
            if mlflow.active_run():
                mlflow.end_run()

            with mlflow.start_run(run_name=model_name):  # Start an MLflow run
                mlflow.log_param("TF-IDF ngram_range", "(1, 1)")
                mlflow.log_param("TF-IDF max_features", 2000)
                print(f"\n{model_name} Cross-Validation Results:")
                grid_search = GridSearchCV(
                    estimator=config["model"],
                    param_grid=config["params"],
                    scoring="accuracy",
                    cv=skf,
                    n_jobs=-1
                )
                grid_search.fit(X_vec, labels)

                # Log the best parameters and best score
                best_params = grid_search.best_params_
                best_score = grid_search.best_score_
                mlflow.log_params(best_params)
                mlflow.log_metric("Best Accuracy", best_score)

                print(f"Best Parameters for {model_name}: {best_params}")
                print(f"Best Accuracy for {model_name}: {best_score}")

                # Log the best model to MLflow
                best_model = grid_search.best_estimator_
                mlflow.sklearn.log_model(best_model, model_name)

                model_filename = f"ml/{model_name.lower().replace(' ', '_')}_model.pkl"
                joblib.dump(best_model, model_filename)

        finally:
            # Ensure the current MLflow run is ended
            if mlflow.active_run():
                mlflow.end_run()

    print("Cross-validation with hyperparameter tuning logged to MLflow")

# Run the main function
if __name__ == "__main__":
    main()