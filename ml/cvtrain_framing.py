import time 
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    posts_cursor = db.posts.find(
        {
        "deleted": {"$ne": 1},
        "content": {"$nin": [None, ""]}
        },
        {"content": 1, "analysis.scam_framing2": 1, "_id": 0}
    )
    posts = []
    labels = []
    async for post in posts_cursor:
        if "content" in post and "analysis" in post and "scam_framing2" in post["analysis"]:
            posts.append(post["content"])
            labels.append(post["analysis"]["scam_framing2"])
    return posts, labels

async def fetch_and_preprocess_data():
    posts, labels = await get_all_posts()
    texts = [clean_text(post) for post in posts]
    return texts, labels

def main():
    start_time = time.time()
    texts, labels = asyncio.run(fetch_and_preprocess_data())

    # Split into train and test sets
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Vectorization
    vectorizer = fit_vectorizer(X_train_texts)
    X_train_vec = transform_text(X_train_texts)
    X_test_vec = transform_text(X_test_texts)

    models = {
        # "SVM": SVC(kernel="linear", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }

    best_score = -1
    best_model = None
    best_model_name = ""
    best_y_pred = None

    for model_name, model in models.items():
        try:
            if mlflow.active_run():
                mlflow.end_run()
            with mlflow.start_run(run_name=model_name):
                mlflow.log_param("scam_framing_label", "scam_framing2")
                print(f"\nTraining {model_name}...")

                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_test_vec)

                acc = accuracy_score(y_test, y_pred)
                print(f"Accuracy: {acc:.4f}")
                print(classification_report(y_test, y_pred))

                mlflow.log_metric("Accuracy", acc)

                if acc > best_score:
                    best_score = acc
                    best_model = model
                    best_model_name = model_name
                    best_y_pred = y_pred

        finally:
            if mlflow.active_run():
                mlflow.end_run()

    if best_model is not None:
        model_name_safe = best_model_name.lower().replace(" ", "_")
        joblib.dump(best_model, f"ml/metrics/scamframing2/{model_name_safe}_model.pkl")
    
        # Save classification report as text
        report_txt_path = f"ml/metrics/scamframing2/{model_name_safe}_classification_report.txt"
        with open(report_txt_path, "w") as f:
            f.write(classification_report(y_test, best_y_pred))
    
        # Save classification report as heatmap image
        report = classification_report(y_test, best_y_pred, output_dict=True)
        report_df = pd.DataFrame(report).iloc[:-1, :].T
        report_fig, report_ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(report_df, annot=True, fmt=".2f", cmap="RdYlGn", ax=report_ax)
        report_ax.set_title(f"{best_model_name} Classification Report")
        plt.tight_layout()
        plt.savefig(f"ml/metrics/scamframing2/{model_name_safe}_classification_report.png")
        plt.close(report_fig)
    
        # Confusion matrix with larger font size for cell values and white text color
        cm = confusion_matrix(y_test, best_y_pred)
        cm_labels = np.unique(y_test)
        cm_fig, cm_ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
        disp.plot(ax=cm_ax, cmap="RdYlGn_r", colorbar=False, values_format='d')
        
        # Make cell value labels larger and white
        for text in disp.text_.ravel():
            text.set_fontsize(24)
            text.set_color("white")
        
        cm_ax.set_title(f"{best_model_name} Confusion Matrix", fontsize=22)
        plt.setp(cm_ax.get_xticklabels(), rotation=90, fontsize=18)
        plt.setp(cm_ax.get_yticklabels(), fontsize=18)
        cm_ax.xaxis.label.set_size(18)
        cm_ax.yaxis.label.set_size(18)
        plt.tight_layout()
        plt.savefig(f"ml/metrics/scamframing2/{model_name_safe}_confusion_matrix.png")
        plt.close(cm_fig)
    end_time = time.time()
    print(f"Total time used: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
