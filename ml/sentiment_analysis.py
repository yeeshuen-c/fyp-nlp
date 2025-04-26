import re
import emoji
from transformers import pipeline
from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect, DetectorFactory
from app.database import db
import pandas as pd

# Initialize Hugging Face pipeline
huggingface_pipe = pipeline("text-classification", model="j-hartmann/sentiment-roberta-large-english-3-classes", truncation=True)

# Preprocessing function
def preprocess_text(text):
    # Convert emojis to text
    text = emoji.demojize(text)
    # Remove URLs, numbers, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s:]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Function to split long text into chunks
def split_text_into_chunks(text, max_length=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # Include space
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Step 4: Read comments from MongoDB and translate from Malay and Chinese to English
DetectorFactory.seed = 0
async def read_and_translate_comments():
    comments_data = await db.comments.find({}).to_list()
    translated_comments = []
    for comment_data in comments_data:
        for comment in comment_data['comments']:
            comment_content = comment['comment_content']

            try:
                # Detect language
                detected_lang = detect(comment_content)
            except:
                detected_lang = 'auto'  # Default if detection fails

            # Translate only if detected language is not English
            if detected_lang != 'en':
                if detected_lang == 'zh-cn':
                    detected_lang = 'zh-CN'
                elif detected_lang == 'zh-tw':
                    detected_lang = 'zh-TW'
                elif detected_lang == 'ms':
                    detected_lang = 'ms'
                try:
                    translated_text = GoogleTranslator(source=detected_lang, target='en').translate(comment_content)
                except Exception as e:
                    print(f"GoogleTranslator failed: {e}. Trying MyMemoryTranslator...")
                    translated_text = MyMemoryTranslator(source=detected_lang, target='en').translate(comment_content)
            else:
                translated_text = comment_content  # No translation needed

            translated_comments.append({
                'comment_id': comment_data['comment_id'],
                'post_id': comment_data['post_id'],
                'original_comment': comment_content,
                'translated_comment': translated_text
            })

    return translated_comments

# Main function
if __name__ == "__main__":
    import asyncio

    async def main():
        # Read and translate comments
        comments = await read_and_translate_comments()

        # Perform sentiment analysis and collect results
        results = []
        for comment in comments:
            # Preprocess text
            cleaned_comment = preprocess_text(comment['translated_comment'])

            # Split long text into chunks
            chunks = split_text_into_chunks(cleaned_comment, max_length=512)

            # Perform sentiment analysis on each chunk and aggregate results
            sentiments = []
            for chunk in chunks:
                if chunk.strip():  # Ensure the chunk is not empty
                    huggingface_result = huggingface_pipe(chunk)[0]
                    sentiments.append(huggingface_result['label'])

            # Determine the overall sentiment based on the majority vote
            if sentiments:
                overall_sentiment = max(set(sentiments), key=sentiments.count)
            else:
                overall_sentiment = "Neutral"  # Default sentiment if no chunks are processed

            # Append results to the list
            results.append({
                'Comment ID': comment['comment_id'],
                'Post ID': comment['post_id'],
                'Original Comment': comment['original_comment'],
                'Translated Comment': comment['translated_comment'],
                'Cleaned Comment': cleaned_comment,
                'Sentiment': overall_sentiment
            })

        # Save the results to an Excel file
        df = pd.DataFrame(results)
        output_file = "sentiment_analysis_hf.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Sentiment analysis results saved to {output_file}")

    asyncio.run(main()) 

   