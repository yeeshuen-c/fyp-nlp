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
        {"content": 1, "analysis.scam_framing2": 1, "_id": 0}  # Fetch content and scam_framing fields
    )
    posts = []
    labels = []
    async for post in posts_cursor:  # Use async for to iterate over the cursor
        if "content" in post and "analysis" in post and "scam_framing2" in post["analysis"]:
            posts.append(post["content"])
            labels.append(post["analysis"]["scam_framing2"])
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
        try:
            # Check if a run is already active and end it
            if mlflow.active_run():
                mlflow.end_run()
            with mlflow.start_run(run_name=model_name):  # Start an MLflow run
                mlflow.log_param("scam_framing_label", "scam_framing2")  # Log the parameter
                print(f"\n{model_name} Cross-Validation Results:")
                fold = 1
                accuracies = []
                all_misclassified = []  # To store misclassified data for all folds

                # Open a text file to save the cross-validation results
                result_file_path = f"metrics_result/{model_name}_cross_validation_results.txt"
                with open(result_file_path, "w") as result_file:
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
                        report = classification_report(y_test, preds, output_dict=False, zero_division=0)
                        conf_matrix = confusion_matrix(y_test, preds)

                        # Print and save results for the current fold
                        result_file.write(f"\nFold {fold}:\n")
                        result_file.write(f"Accuracy: {acc}\n")
                        result_file.write("Classification Report:\n")
                        result_file.write(f"{report}\n")
                        result_file.write("Confusion Matrix:\n")
                        result_file.write(f"{conf_matrix}\n")

                        # print(f"\nFold {fold}:")
                        # print(f"Accuracy: {acc}")
                        # print("Classification Report:")
                        # print(report)
                        # print("Confusion Matrix:")
                        # print(conf_matrix)

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

                    # Save average accuracy to the result file
                    result_file.write(f"\nAverage Accuracy: {avg_accuracy}\n")

                print(f"Cross-validation results saved to {result_file_path}")

                # Log the trained model to MLflow
                mlflow.sklearn.log_model(model, model_name)

            # End the timer
            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time used: {total_time:.2f} seconds")
            mlflow.log_metric("Total_Time_Used_seconds", total_time)
        finally:
            # Ensure the current MLflow run is ended
            if mlflow.active_run():
                mlflow.end_run()

    print(f"Cross-validation results saved to mlflow")

# Run the main function
if __name__ == "__main__":
    main()