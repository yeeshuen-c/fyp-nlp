import joblib
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
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
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split

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
        {
        "deleted": {"$ne": 1},
        "content": {"$nin": [None, ""]}
        },
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
    texts, labels = asyncio.run(fetch_and_preprocess_data())

    # Split data into train/test sets (70/30 stratified)
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, stratify=labels, random_state=42
    )

    # Fit vectorizer only on training data
    vectorizer = fit_vectorizer(X_train_texts)
    X_train_vec = transform_text(X_train_texts)
    X_test_vec = transform_text(X_test_texts)

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
        "Naive Bayes": {
            "model": MultinomialNB(),
            "params": {
                "alpha": [0.1, 0.5, 1.0]
            }
        },
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

    best_overall_score = -1
    best_overall_model = None
    best_overall_model_name = ""
    best_overall_y_pred = None

    # Prepare a single file to store all classification reports
    all_reports_path = "ml/metrics/scamtype/all_models_classification_report.txt"
    with open(all_reports_path, "w") as all_reports_file:

        # Perform cross-validation with GridSearchCV
        for model_name, config in model_params.items():
            try:
                if mlflow.active_run():
                    mlflow.end_run()

                with mlflow.start_run(run_name=model_name):
                    print(f"\n{model_name} Cross-Validation Results:")
                    grid_search = GridSearchCV(
                        estimator=config["model"],
                        param_grid=config["params"],
                        scoring="accuracy",
                        cv=skf,
                        n_jobs=-1
                    )
                    grid_search.fit(X_train_vec, y_train)

                    best_score = grid_search.best_score_
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test_vec)

                    # Write classification report for this model to the single file
                    all_reports_file.write(f"=== {model_name} ===\n")
                    all_reports_file.write(f"Best Parameters: {grid_search.best_params_}\n")
                    all_reports_file.write(f"Best CV Accuracy: {best_score:.4f}\n")
                    all_reports_file.write(classification_report(y_test, y_pred))
                    all_reports_file.write("\n\n")

                    if best_score > best_overall_score:
                        best_overall_score = best_score
                        best_overall_model = best_model
                        best_overall_model_name = model_name
                        best_overall_y_pred = y_pred

            finally:
                if mlflow.active_run():
                    mlflow.end_run()

    # --- Save and log only the best model and its evaluation images ---
    # if best_overall_model is not None:
    #     model_name_safe = best_overall_model_name.lower().replace(" ", "_")
    #     model_filename = f"ml/metrics/scamtype/{model_name_safe}_model.pkl"
    #     joblib.dump(best_overall_model, model_filename)

    #     # --- Classification Report ---
    #     report = classification_report(y_test, best_overall_y_pred, output_dict=True)
    #     report_df = pd.DataFrame(report).iloc[:-1, :].T

    #     report_fig, report_ax = plt.subplots(figsize=(8, 4))
    #     sns.heatmap(report_df, annot=True, fmt=".2f", cmap="RdYlGn", ax=report_ax)
    #     report_ax.set_title(f"{best_overall_model_name} Classification Report")
    #     plt.tight_layout()
    #     report_img_path = f"ml/metrics/scamtype/{model_name_safe}_classification_report.png"
    #     plt.savefig(report_img_path)
    #     plt.close(report_fig)

    #     # Save classification report as text
    #     report_txt_path = f"ml/metrics/scamtype/{model_name_safe}_classification_report.txt"
    #     with open(report_txt_path, "w") as f:
    #         f.write(classification_report(y_test, best_overall_y_pred))

    #     # --- Confusion Matrix ---
    #     cm = confusion_matrix(y_test, best_overall_y_pred)
    #     cm_labels = np.unique(y_test)
    #     cm_fig, cm_ax = plt.subplots(figsize=(6, 6))
    #     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    #     disp.plot(ax=cm_ax, cmap="RdYlGn_r", colorbar=False, values_format='d')
    #     cm_ax.set_title(f"{best_overall_model_name} Confusion Matrix")
    #     plt.setp(cm_ax.get_xticklabels(), rotation=90)
    #     plt.tight_layout()
    #     cm_img_path = f"ml/metrics/scamtype/{model_name_safe}_confusion_matrix.png"
    #     plt.savefig(cm_img_path)
    #     plt.close(cm_fig)

    #     # Save confusion matrix as text
    #     cm_txt_path = f"ml/metrics/scamtype/{model_name_safe}_confusion_matrix.txt"
    #     with open(cm_txt_path, "w") as f:
    #         f.write("Confusion Matrix:\n")
    #         f.write(np.array2string(cm))
    #         f.write("\nLabels:\n")
    #         f.write(", ".join(map(str, cm_labels)))

    print("Best model, classification report, and confusion matrix saved.")
    print("Cross-validation with hyperparameter tuning logged to MLflow")

# def main():
#     model = joblib.load("ml/metrics/scamtype/logistic_regression_model.pkl")
#     print(model.get_params())

# Run the main function
if __name__ == "__main__":
    main()