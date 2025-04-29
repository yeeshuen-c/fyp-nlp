import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.database import db
from ml.preprocessing import clean_text, fit_vectorizer, transform_text
import asyncio
import mlflow
import mlflow.sklearn

# Fetch data from MongoDB
async def get_all_posts():
    """Fetch all posts from the MongoDB collection and return their content and labels."""
    posts_cursor = db.posts.find(
        {"deleted": {"$ne": 1}},  # Exclude deleted posts
        {"content": 1, "analysis.scam_framing": 1, "_id": 0}  # Fetch content and scam_framing fields
    )
    posts = []
    labels = []
    async for post in posts_cursor:  # Use async for to iterate over the cursor
        if "content" in post and "analysis" in post and "scam_framing" in post["analysis"]:
            posts.append(post["content"])
            labels.append(post["analysis"]["scam_framing"])
    return posts, labels

# Fetch and preprocess data
async def fetch_and_preprocess_data():
    posts, labels = await get_all_posts()  # Await the async function
    texts = [clean_text(post) for post in posts]  # Preprocess text
    return texts, labels

# Main function
def main():
    # Start the timer
    start_time = time.time()

    # Run the async function using an event loop
    texts, labels = asyncio.run(fetch_and_preprocess_data())

    # Vectorization
    vectorizer = fit_vectorizer(texts)
    X_vec = transform_text(texts)

    # Initialize models
    models = {
        "SVM": SVC(kernel="linear", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    
    # Perform cross-validation
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):  # Start an MLflow run
            print(f"\n{model_name} Cross-Validation Results:")
            fold = 1
            accuracies = []
            all_misclassified = []  # To store misclassified data for all folds
            for train_index, test_index in skf.split(X_vec, labels):
                X_train, X_test = X_vec[train_index], X_vec[test_index]
                y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
                texts_test = np.array(texts)[test_index]  # Get the corresponding test texts

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                preds = model.predict(X_test)

                # Calculate metrics
                acc = accuracy_score(y_test, preds)
                accuracies.append(acc)
                report = classification_report(y_test, preds, output_dict=True, zero_division=0)
                conf_matrix = confusion_matrix(y_test, preds)

                print(f"\nFold {fold}:")
                print(f"Accuracy: {acc}")
                print("Classification Report:")
                print(report)
                print("Confusion Matrix:")
                print(conf_matrix)

                # Collect misclassified data
                misclassified = []
                for i in range(len(y_test)):
                    if preds[i] != y_test[i]:
                        misclassified.append({
                            "Text": texts_test[i],
                            "True Label": y_test[i],
                            "Predicted Label": preds[i]
                        })
                all_misclassified.extend(misclassified)
                fold += 1

            # Log metrics and parameters to MLflow
            avg_accuracy = np.mean(accuracies)
            mlflow.log_param("Model", model_name)
            mlflow.log_metric("Average Accuracy", avg_accuracy)

            # # Save misclassified data to an Excel file
            # if all_misclassified:
            #     df_misclassified = pd.DataFrame(all_misclassified)
            #     excel_file = f"excel/{model_name}_framing_misclassified.xlsx"
            #     df_misclassified.to_excel(excel_file, index=False)
            #     print(f"Misclassified data for {model_name} saved to {excel_file}")

            #     # Log the misclassified Excel file to MLflow
            #     mlflow.log_artifact(excel_file)

            # Log the trained model to MLflow
            mlflow.sklearn.log_model(model, model_name)
            
        # End the timer
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total time used: {total_time:.2f} seconds")
        mlflow.log_metric("Total_Time_Used_seconds", total_time)

    print(f"Cross-validation results saved to mlflow")

# Run the main function
if __name__ == "__main__":
    main()