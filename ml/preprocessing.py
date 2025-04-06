import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from deep_translator import GoogleTranslator
from transformers import BertTokenizer, BertModel
import torch
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def translate_to_english(text):
    """Translate text to English using Deep Translator, handling long texts."""
    try:
        # Split text into chunks of 4000 characters
        max_length = 4000
        chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
        
        # Translate each chunk and combine the results
        translated_chunks = [GoogleTranslator(source='auto', target='en').translate(chunk) for chunk in chunks]
        return " ".join(translated_chunks)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return the original text if translation fails

def clean_text(text):
    # Translate to English
    text = translate_to_english(text)
    # Lowercase
    text = text.lower()
    # Remove URLs and mentions
    text = re.sub(r"http\S+|www\S+|@\S+", "", text)
    # Remove special characters
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove stopwords & lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

# def fit_vectorizer(texts):
#     """Fit and save a TF-IDF vectorizer"""
#     vectorizer = TfidfVectorizer(max_features=2000, norm='l2')
#     vectorizer.fit(texts)
#     joblib.dump(vectorizer, "ml/vectorizer.pkl")
#     return vectorizer

# def transform_text(texts):
#     """Load vectorizer and transform text"""
#     vectorizer = joblib.load("ml/vectorizer.pkl")
#     return vectorizer.transform(texts)

def mean_pooling(model_output, attention_mask):
    """Apply mean pooling to BERT output"""
    token_embeddings = model_output[0]  # First element of output contains embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
    return sum_embeddings / sum_mask

def fit_vectorizer(texts):
    """Initialize and save BERT tokenizer and model (no fitting like TF-IDF)"""
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    # Save tokenizer and model
    tokenizer.save_pretrained("ml/bert_vectorizer")
    model.save_pretrained("ml/bert_vectorizer")

    return tokenizer, model


def transform_text(texts):
    """Load BERT tokenizer/model and convert texts into embeddings"""
    tokenizer = BertTokenizer.from_pretrained("ml/bert_vectorizer")
    model = BertModel.from_pretrained("ml/bert_vectorizer")
    model.eval()

    with torch.no_grad():
        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        output = model(**encoded_input)
        embeddings = mean_pooling(output, encoded_input['attention_mask'])
    return embeddings
