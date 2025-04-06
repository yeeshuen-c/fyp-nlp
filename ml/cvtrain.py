import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from app.database import db
from ml.preprocessing import clean_text, fit_vectorizer, transform_text
import asyncio
from sklearn.preprocessing import MinMaxScaler

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

    # scaler for bert vector 
    scaler = MinMaxScaler()
    X_vec = scaler.fit_transform(X_vec)

    # Initialize models
    models = {
        "SVM": SVC(kernel="linear", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
    }

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Open a file to save the output
    output_file = "cv_bert_pca.txt"
    with open(output_file, "w") as f:
        # Perform cross-validation
        for model_name, model in models.items():
            f.write(f"\n{model_name} Cross-Validation Results:\n")
            fold = 1
            accuracies = []
            for train_index, test_index in skf.split(X_vec, labels):
                X_train, X_test = X_vec[train_index], X_vec[test_index]
                y_train, y_test = np.array(labels)[train_index], np.array(labels)[test_index]

                # Train the model
                model.fit(X_train, y_train)

                # Make predictions
                preds = model.predict(X_test)

                # Calculate metrics
                acc = accuracy_score(y_test, preds)
                accuracies.append(acc)
                report = classification_report(y_test, preds, output_dict=True, zero_division=0)
                conf_matrix = confusion_matrix(y_test, preds)

                f.write(f"\nFold {fold}:\n")
                f.write(f"Accuracy: {acc}\n")
                f.write("Classification Report:\n")
                for label, metrics in report.items():
                    f.write(f"{label}: {metrics}\n")
                f.write("Confusion Matrix:\n")
                f.write(f"{conf_matrix}\n")
                fold += 1

            # Write average accuracy across all folds
            f.write(f"\nAverage Accuracy for {model_name}: {np.mean(accuracies):.4f}\n")

    print(f"Cross-validation results saved to {output_file}")

# Run the main function
if __name__ == "__main__":
    main()