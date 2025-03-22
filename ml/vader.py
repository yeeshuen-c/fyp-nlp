# Step 1: Install necessary libraries
# pip install nltk emoji vaderSentiment googletrans==4.0.0-rc1 pandas openpyxl deep-translator langdetect

import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from deep_translator import GoogleTranslator, MyMemoryTranslator
from langdetect import detect, DetectorFactory
from app.database import db

# Step 2: Function for preprocessing text
def preprocess_text(text):
    # Convert emojis to text (using emoji library)
    text = emoji.demojize(text)
    
    # Remove special characters, URLs, and numbers
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Remove URLs
    text = re.sub(r'\d+', '', text)                     # Remove numbers
    text = re.sub(r'[^\w\s:]', '', text)                # Remove special characters except ':' for emojis
    
    # Convert to lowercase
    text = text.lower()
    return text

# Step 3: Extend the VADER lexicon for Malay words
def extend_vader_for_malay(analyzer):
    # Malay words and their sentiment scores (custom lexicon)
    malay_lexicon = {
        "baik": 2.0,         # good
        "teruk": -2.5,       # bad
        "gembira": 2.5,      # happy
        "sedih": -2.5,       # sad
        "bagus": 2.3,        # excellent
        "menyakitkan": -2.8, # painful
        "marah": -2.0,       # angry
        "teruja": 2.0,       # excited
        "menarik": 2.2,      # interesting
        "bosan": -2.0,       # boring
        "malas": -2.0,       # lazy
        "penipu": -3.0,      # scammer
        "memenangi": 2.3,    # winning
        "rugi": -2.7,        # loss
        "percuma": 1.8       # free
    }
    analyzer.lexicon.update(malay_lexicon)

# Step 4: Read comments from MongoDB and translate from Malay and Chinese to English
DetectorFactory.seed = 0

async def read_and_translate_comments():
    comments_data = await db.comments.find({"comment_id": {"$lt": 101}}).to_list()
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

# Step 5: Main program
if __name__ == "__main__":
    import asyncio

    async def main():
        # Initialize SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        # analyzer.lexicon['scam'] = 0.0 # Neutral sentiment for 'scam'
        
        # Extend VADER lexicon for Malay words
        extend_vader_for_malay(analyzer)
        
        # Read and translate comments from MongoDB
        comments = await read_and_translate_comments()
        
        # Group comments by comment_id
        grouped_comments = {}
        for comment in comments:
            comment_id = comment['comment_id']
            if comment_id not in grouped_comments:
                grouped_comments[comment_id] = []
            grouped_comments[comment_id].append(comment)
        
        # Preprocess comments and perform sentiment analysis
        results = []
        for comment_id, comment_group in grouped_comments.items():
            overall_sentiment_scores = []
            for comment in comment_group:
                cleaned_comment = preprocess_text(comment['translated_comment'])
                sentiment_scores = analyzer.polarity_scores(cleaned_comment)
                
                # Classify sentiment as Positive, Negative, or Neutral
                if sentiment_scores['pos'] == 0 and sentiment_scores['neg'] == 0:
                    sentiment = 'Neutral'
                elif sentiment_scores['pos'] > sentiment_scores['neg']:
                    sentiment = 'Positive'
                elif sentiment_scores['neg'] > sentiment_scores['pos']:
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                
                results.append({
                    'Original Comment': comment['original_comment'],
                    'Cleaned Comment': cleaned_comment,
                    'Compound Score': sentiment_scores['compound'],
                    'Positive Scores': sentiment_scores['pos'],
                    'Negative Scores': sentiment_scores['neg'],
                    'Sentiment': sentiment
                })

                # Update the sentiment analysis result for each comment in the database
                await db.comments.update_one(
                    {'comment_id': comment['comment_id'], 'comments.comment_content': comment['original_comment']},
                    {'$set': {'comments.$.sentiment_analysis': sentiment}}
                )

                # Collect sentiment scores for overall analysis
                overall_sentiment_scores.append(sentiment_scores['compound'])

            # Calculate overall sentiment based on average compound score
            if overall_sentiment_scores:
                average_sentiment_score = sum(overall_sentiment_scores) / len(overall_sentiment_scores)
                if average_sentiment_score >= 0.05:
                    overall_sentiment = 'Positive'
                elif average_sentiment_score <= -0.05:
                    overall_sentiment = 'Negative'
                else:
                    overall_sentiment = 'Neutral'
            else:
                overall_sentiment = 'Neutral'

            # print(f"Comment ID: {comment_id}, Overall Sentiment: {overall_sentiment}")
            # Update the overall sentiment analysis result in the database
            # await db.comments.update_one(
            #     {'comment_id': comment_id},
            #     {'$set': {'analysis.sentiment_analysis': overall_sentiment}}
            # )

        print("Sentiment analysis completed.")    

        # Display the results in a DataFrame
        # df = pd.DataFrame(results)
        # df.to_excel("sentiment_analysis_results.xlsx", index=False)
        # print("Sentiment analysis results saved to sentiment_analysis_results.xlsx")

    asyncio.run(main())