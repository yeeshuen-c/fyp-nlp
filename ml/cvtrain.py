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
    texts = [clean_text(post) for post in posts]  # Preprocess text
    return texts, labels

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
        # }
    }

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Perform cross-validation with GridSearchCV
    for model_name, config in model_params.items():
        with mlflow.start_run(run_name=model_name):  # Start an MLflow run
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

            # Collect misclassified data
            best_model = grid_search.best_estimator_
            all_misclassified = []
            fold = 1
            for train_index, test_index in skf.split(X_vec, labels):
                X_train, X_test = X_vec[train_index], X_vec[test_index]
                y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
                texts_test = np.array(texts)[test_index]  # Get the corresponding test texts

                # Train the best model
                best_model.fit(X_train, y_train)

                # Make predictions
                preds = best_model.predict(X_test)

                # Collect misclassified data
                for i in range(len(y_test)):
                    if preds[i] != y_test[i]:
                        all_misclassified.append({
                            "Text": texts_test[i],
                            "True Label": y_test[i],
                            "Predicted Label": preds[i]
                        })
                fold += 1

            # Save misclassified data to an Excel file
            if all_misclassified:
                df_misclassified = pd.DataFrame(all_misclassified)
                excel_file = f"excel/{model_name}_scamtype_misclassified.xlsx"
                df_misclassified.to_excel(excel_file, index=False)
                print(f"Misclassified data for {model_name} saved to {excel_file}")

                # Log the misclassified Excel file to MLflow
                mlflow.log_artifact(excel_file)

            # Log the best model to MLflow
            mlflow.sklearn.log_model(best_model, model_name)

    print("Cross-validation with hyperparameter tuning logged to MLflow")

# Run the main function
if __name__ == "__main__":
    main()