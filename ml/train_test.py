import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_curve, auc
)
from app.database import db
from ml.preprocessing import clean_text, fit_vectorizer, transform_text
import asyncio

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = fit_vectorizer(X_train)
    X_train_vec = transform_text(X_train)
    X_test_vec = transform_text(X_test)

    # Train SVM
    svm_model = SVC(kernel="linear", probability=True)  # Enable probability estimates
    svm_model.fit(X_train_vec, y_train)
    joblib.dump(svm_model, "ml/svm_model.pkl")

    # Train KNN
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_vec, y_train)
    joblib.dump(knn_model, "ml/knn_model.pkl")

    # Predictions
    svm_preds = svm_model.predict(X_test_vec)
    knn_preds = knn_model.predict(X_test_vec)

    # Probabilities for ROC curve
    svm_probs = svm_model.predict_proba(X_test_vec)[:, 1]
    knn_probs = knn_model.predict_proba(X_test_vec)[:, 1]

    # Metrics
    svm_acc = accuracy_score(y_test, svm_preds)
    knn_acc = accuracy_score(y_test, knn_preds)
    svm_report = classification_report(y_test, svm_preds, output_dict=True)
    knn_report = classification_report(y_test, knn_preds, output_dict=True)
    svm_conf_matrix = confusion_matrix(y_test, svm_preds).tolist()
    knn_conf_matrix = confusion_matrix(y_test, knn_preds).tolist()

    # # Save metrics
    # metrics = {
    #     "svm": {"accuracy": svm_acc, "report": svm_report, "confusion_matrix": svm_conf_matrix},
    #     "knn": {"accuracy": knn_acc, "report": knn_report, "confusion_matrix": knn_conf_matrix}
    # }
    # joblib.dump(metrics, "ml/metrics.pkl")

    # Print metrics to the terminal
    print("SVM Metrics:")
    print(f"Accuracy: {svm_acc}")
    print("Classification Report:")
    for label, report in svm_report.items():
        print(f"{label}: {report}")
    print("Confusion Matrix:")
    print(svm_conf_matrix)

    print("\nKNN Metrics:")
    print(f"Accuracy: {knn_acc}")
    print("Classification Report:")
    for label, report in knn_report.items():
        print(f"{label}: {report}")
    print("Confusion Matrix:")
    print(knn_conf_matrix)

    # ROC Curve
    # fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
    # fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)
    # roc_auc_svm = auc(fpr_svm, tpr_svm)
    # roc_auc_knn = auc(fpr_knn, tpr_knn)

    # plt.figure(figsize=(6, 6))
    # plt.plot(fpr_svm, tpr_svm, label=f"SVM (AUC = {roc_auc_svm:.2f})")
    # plt.plot(fpr_knn, tpr_knn, label=f"KNN (AUC = {roc_auc_knn:.2f})")
    # plt.plot([0, 1], [0, 1], "k--")  # Random line
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve")
    # plt.legend()
    # plt.savefig("ml/roc_curve.png")  # Save the graph

# Run the main function
if __name__ == "__main__":
    main()