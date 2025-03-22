import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Example dataset (replace this with your social media data)
data = pd.DataFrame({
    'text': [
        "Win a free iPhone now! Click the link below!",
        "Your bank account has been compromised. Call us immediately.",
        "Congratulations! You've won $10,000 in cash prizes!",
        "Hello, just checking in to see how you're doing.",
        "Important notice: Your account will be deactivated soon."
    ],
    'label': [1, 1, 1, 0, 1]  # 1 = Scam, 0 = Not Scam
})

# --- Step 1: Text Preprocessing ---
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[\d' + string.punctuation + ']', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stop words
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply text cleaning and preprocessing
data['cleaned_text'] = data['text'].apply(clean_text)
data['processed_text'] = data['cleaned_text'].apply(preprocess_text)

# --- Step 2: Feature Engineering ---
# Convert text into numerical representations using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Use n-grams to capture patterns
X = vectorizer.fit_transform(data['processed_text'])
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Step 3: Model Training ---
# Train an SVM model
svm = SVC()
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10]
}
# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Evaluate the best model on the test set
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# --- Step 4: Performance Tuning ---
# Output performance metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
