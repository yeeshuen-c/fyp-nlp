import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from app.database import db
from ml.preprocessing import clean_text, fit_vectorizer, transform_text
import asyncio
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Track the best model
    best_overall_score = -1
    best_overall_model = None
    best_overall_model_name = ""
    best_overall_y_pred = None

    # Perform cross-validation
    for model_name, model in models.items():
        try:
            if mlflow.active_run():
                mlflow.end_run()
            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("scam_framing_label", "scam_framing2")
                print(f"\n{model_name} Cross-Validation Results:")
                fold = 1
                accuracies = []
                all_misclassified = []

                result_file_path = f"metrics_result/{model_name}_cross_validation_results.txt"
                with open(result_file_path, "w") as result_file:
                    for train_index, test_index in skf.split(X_vec, labels):
                        X_train, X_test = X_vec[train_index], X_vec[test_index]
                        y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]
                        texts_test = np.array(texts)[test_index]

                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)

                        acc = accuracy_score(y_test, preds)
                        accuracies.append(acc)
                        report = classification_report(y_test, preds, output_dict=False, zero_division=0)
                        conf_matrix = confusion_matrix(y_test, preds)

                        result_file.write(f"\nFold {fold}:\n")
                        result_file.write(f"Accuracy: {acc}\n")
                        result_file.write("Classification Report:\n")
                        result_file.write(f"{report}\n")
                        result_file.write("Confusion Matrix:\n")
                        result_file.write(f"{conf_matrix}\n")

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

                    avg_accuracy = np.mean(accuracies)
                    # mlflow.log_param("Model", model_name)
                    # mlflow.log_metric("Average Accuracy", avg_accuracy)
                    # result_file.write(f"\nAverage Accuracy: {avg_accuracy}\n")

                # Use mean accuracy as score for model selection
                best_score = avg_accuracy
                if best_score > best_overall_score:
                    best_overall_score = best_score
                    best_overall_model = model
                    best_overall_model_name = model_name
                    best_overall_y_pred = best_overall_model.predict(X_vec)

                print(f"Cross-validation results saved to {result_file_path}")

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Total time used: {total_time:.2f} seconds")
            mlflow.log_metric("Total_Time_Used_seconds", total_time)
        finally:
            if mlflow.active_run():
                mlflow.end_run()

    if best_overall_model is not None:
        model_name_safe = best_overall_model_name.lower().replace(" ", "_")
        model_filename = f"ml/metrics/scamframing2/{model_name_safe}_model.pkl"
        joblib.dump(best_overall_model, model_filename)

        # --- Classification Report ---
        report = classification_report(labels, best_overall_y_pred, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :].T

        report_fig, report_ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(report_df, annot=True, fmt=".2f", cmap="RdYlGn", ax=report_ax)
        report_ax.set_title(f"{best_overall_model_name} Classification Report")
        plt.tight_layout()
        report_img_path = f"ml/metrics/scamframing2/{model_name_safe}_classification_report.png"
        plt.savefig(report_img_path)
        plt.close(report_fig)

        # Save classification report as text
        report_txt_path = f"ml/metrics/scamframing2/{model_name_safe}_classification_report.txt"
        with open(report_txt_path, "w") as f:
            f.write(classification_report(labels, best_overall_y_pred))

        # --- Confusion Matrix ---
        cm = confusion_matrix(labels, best_overall_y_pred)
        cm_labels = np.unique(labels)
        cm_fig, cm_ax = plt.subplots(figsize=(6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
        # Use reversed colormap so red is high, green is low
        disp.plot(ax=cm_ax, cmap="RdYlGn_r", colorbar=False, values_format='d')
        cm_ax.set_title(f"{best_overall_model_name} Confusion Matrix", fontsize=18)
        plt.setp(cm_ax.get_xticklabels(), rotation=90, fontsize=14)  # Rotate x-axis labels 90 degrees, larger font
        plt.setp(cm_ax.get_yticklabels(), fontsize=14)  # Larger font for y-axis
        cm_ax.xaxis.label.set_size(16)
        cm_ax.yaxis.label.set_size(16)
        plt.tight_layout()
        cm_img_path = f"ml/metrics/scamframing2/{model_name_safe}_confusion_matrix.png"
        plt.savefig(cm_img_path)
        plt.close(cm_fig)

        # Save confusion matrix as text
        cm_txt_path = f"ml/metrics/scamframing2/{model_name_safe}_confusion_matrix.txt"
        with open(cm_txt_path, "w") as f:
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(cm))
            f.write("\nLabels:\n")
            f.write(", ".join(map(str, cm_labels)))

    print(f"Cross-validation results saved to mlflow")

if __name__ == "__main__":
    main()