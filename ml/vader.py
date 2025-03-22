# Step 1: Install necessary libraries
# pip install nltk emoji vaderSentiment googletrans==4.0.0-rc1 pandas openpyxl deep-translator

import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from deep_translator import GoogleTranslator
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
async def read_and_translate_comments():
    comments_data = await db.comments.find({"comment_id": {"$gt": 100}}).to_list()
    translator = GoogleTranslator(target='en')
    translated_comments = []
    for comment_data in comments_data:
        for comment in comment_data['comments']:
            comment_content = comment['comment_content']
            # Detect language and translate to English
            translated_text = translator.translate(comment_content)
            translated_comments.append({
                'comment_id': comment_data['comment_id'],
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
        analyzer.lexicon['scam'] = 0.0 # Neutral sentiment for 'scam'
        
        # Extend VADER lexicon for Malay words
        extend_vader_for_malay(analyzer)
        
        # Read and translate comments from MongoDB
        comments = await read_and_translate_comments()
        
        # Preprocess comments and perform sentiment analysis
        results = []
        for comment in comments:
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

            # Update the sentiment analysis result in the database
            # await db.comments.update_one(
            #     {'comment_id': comment['comment_id']},
            #     {'$set': {'analysis.sentiment_analysis': sentiment}}
            # )

        # Display the results in a DataFrame
        df = pd.DataFrame(results)
        df.to_excel("sentiment_analysis_results.xlsx", index=False)
        print("Sentiment analysis results saved to sentiment_analysis_results.xlsx")

    asyncio.run(main())
