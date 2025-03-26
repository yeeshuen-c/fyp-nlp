import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from deep_translator import GoogleTranslator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load stopwords for English and Malay
stop_words_en = set(stopwords.words("english"))
stop_words_my = set([
    "dan", "yang", "untuk", "dengan", "tidak", "dalam", "ini", "itu", 
    "oleh", "pada", "jika", "kerana", "adalah", "ke", "di"
])

def preprocess_text(text):
    """
    Preprocesses Malay and English text:
    1. Translates Malay text to English
    2. Converts to lowercase
    3. Removes punctuation & special characters
    4. Tokenizes text
    5. Removes stopwords
    6. Applies stemming
    """
    # Detect and translate Malay to English
    translated_text = GoogleTranslator(source="auto", target="en").translate(text)

    # Lowercasing
    text = translated_text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal (Malay & English)
    tokens = [word for word in tokens if word not in stop_words_en and word not in stop_words_my]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return " ".join(tokens)  # Convert list back to string

def get_preprocessed():
    max_f = 200  

    """Step 1: Load Data"""
    from final_process_2 import noStopWord_comments, noStopWord_class  # Import pre-processed text and labels

    """Step 2: Preprocessing"""
    # Apply text preprocessing to all comments
    corpus = [preprocess_text(comment) for comment in noStopWord_comments]

    # Convert text into numerical representation
    tfidf = TfidfVectorizer()
    retfidf = tfidf.fit_transform(corpus)
    input_data_matrix = retfidf.toarray()

    """Step 3: Train-Test Split"""
    x_train, x_test, y_train, y_test = train_test_split(
        input_data_matrix, noStopWord_class, test_size=0.2, random_state=400
    )

    """Step 4: Standardization & Normalization"""
    scaler = preprocessing.MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    """Step 5: Dimensionality Reduction (PCA)"""
    pca = PCA(n_components=0.9)
    x_train = pca.fit_transform(x_train)
    x_test = pca.transform(x_test)

    return x_train, x_test, y_train, y_test
