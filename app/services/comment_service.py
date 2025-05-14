import re
import emoji
from transformers import pipeline
from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect, DetectorFactory

# Set deterministic language detection
DetectorFactory.seed = 0

# Initialize Hugging Face sentiment pipeline once
huggingface_pipe = pipeline("text-classification", model="j-hartmann/sentiment-roberta-large-english-3-classes", truncation=True)

# Text preprocessing
def preprocess_text(text):
    text = emoji.demojize(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s:]', '', text)
    text = text.lower()
    return text

# Text chunking
def split_text_into_chunks(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Translation logic
def translate_text(text):
    try:
        lang = detect(text)
    except:
        lang = "auto"

    if lang != 'en':
        if lang == 'zh-cn': lang = 'zh-CN'
        elif lang == 'zh-tw': lang = 'zh-TW'
        try:
            return GoogleTranslator(source=lang, target='en').translate(text)
        except:
            return MyMemoryTranslator(source=lang, target='en').translate(text)
    return text

# Prediction logic
def classify_comment_sentiment(comment_text: str):
    translated = translate_text(comment_text)
    cleaned = preprocess_text(translated)
    chunks = split_text_into_chunks(cleaned)

    sentiments = []
    for chunk in chunks:
        if chunk.strip():
            result = huggingface_pipe(chunk)[0]
            sentiments.append(result['label'])

    overall_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "neutral"
    
    return {
        "original_comment": comment_text,
        "translated_comment": translated,
        "cleaned_comment": cleaned,
        "sentiment": overall_sentiment,
    }
