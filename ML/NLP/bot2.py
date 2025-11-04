# This is an improved marvin bot using the TextBlob library for sentiment analysis
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor

# Import and create a conll extractor to use later
extractor = ConllExtractor()

# Get user input
user_input = input("> ")

# Create TextBlob with noun phrase extractor
user_input_blob = TextBlob(user_input, np_extractor=extractor) 
np = user_input_blob.noun_phrases

# Analyze sentiment and respond
if user_input_blob.polarity <= -0.5:
    response = "Oh dear, that sounds bad."
elif user_input_blob.polarity <= 0:
    response = "Hmm, that's not great"
elif user_input_blob.polarity <= 0.5:
    response = "Well, that sounds positive."
elif user_input_blob.polarity <= 1:
    response = "Wow, that sounds great."

print(response)