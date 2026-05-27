# Hotel Reviews Sentiment Analysis & NLP Pipeline

## Overview

This project demonstrates a comprehensive Natural Language Processing (NLP) pipeline for analyzing hotel reviews and extracting sentiment insights. Using a dataset of 515,000+ reviews across 1,493 hotels in 6 European cities, the project showcases the complete workflow: data exploration, text preprocessing, sentiment analysis, and feature engineering. The processed data is designed to power a hotel recommendation system that considers both review sentiment and guest demographics. This is a practical example of how NLP transforms raw text data into structured, actionable insights.

## Dataset

**Source:** Hotel Reviews Dataset  
**Files:**
- `Hotel_Reviews.csv` - Raw dataset (515,000+ reviews, ~50MB)
- `Hotel_Reviews_Filtered.csv` - Cleaned and preprocessed
- `Hotel_Reviews_NLP.csv` - Final processed dataset with sentiment scores

**Size:** 515,000+ reviews of 1,493 hotels  
**Locations:** 6 European cities (Amsterdam, Barcelona, London, Milan, Paris, Vienna)  
**Features:** Hotel names, addresses, review text (positive/negative), reviewer nationality, scores, dates, and guest/trip type tags

The dataset captures reviews across multiple dimensions including reviewer nationality (50+ countries), hotel locations, and guest characteristics (families, couples, business travelers, solo travelers, groups).

## Methods

### Data Exploration & Analysis
- Loaded and profiled 515,000+ reviews from CSV file
- Analyzed reviewer nationality distribution across 50+ countries
- Identified most-reviewed hotels per nationality
- Calculated frequency distributions and aggregate metrics
- Examined review scores and their consistency

### Data Cleaning & Standardization
- **Address Normalization:** Standardized hotel addresses to city/country format
- **Score Recalculation:** Computed average hotel scores from individual reviewer scores rather than using provided values
- **Outlier Detection:** Identified and analyzed discrepancies between provided and calculated scores
- **Row Filtering:** Removed duplicate hotel entries and irrelevant columns
- **Missing Data Handling:** Identified and processed "No Negative" and "No Positive" review placeholders

### Natural Language Processing
- **Stop Word Removal:** Eliminated 179 common English words that don't affect sentiment
- **Text Processing:** Cleaned review text for sentiment analysis
- **Performance Optimization:** Used set-based stop word removal for fast processing on large datasets

### Sentiment Analysis
- **VADER Sentiment Analyzer:** Applied NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner)
  - Compound score range: -1 (most negative) to +1 (most positive)
  - Handles negations, intensifiers, and punctuation
- **Dual Analysis:** Computed separate sentiment scores for positive and negative review columns
- **Null Handling:** Assigned score of 0 to "No Negative" and "No Positive" placeholders
- **Validation:** Compared sentiment scores against actual reviewer scores to detect mismatches

### Feature Engineering
Extracted 8 guest/trip type features from review tags:
- **Trip Type:** Leisure trip, Business trip
- **Group Type:** Couple, Solo traveler, Group, Family with young children, Family with older children
- **Other:** With a pet

These binary features enable segmented hotel recommendations for different traveler types.

## Results

The project successfully processed 515,000+ reviews into a structured, analytics-ready dataset:
- **Data Quality:** Standardized addresses and calculated consistent hotel scores
- **Sentiment Extraction:** Computed sentiment scores for all 515,000+ reviews using VADER analysis
- **Feature Engineering:** Created 8 traveler-type features for segmented analysis
- **Processed Output:** Generated `Hotel_Reviews_NLP.csv` with 18 columns ready for recommendation system
- **Sentiment Insights:** 
  - Positive reviews showed average compound sentiment scores
  - Negative reviews showed distinct sentiment patterns
  - Some reviews exhibited sarcasm/complexity not fully captured by sentiment analyzer
- **Demographic Analysis:** Identified nationality-based review patterns and hotel preferences

The final dataset enables building recommendation systems that consider sentiment, traveler type, and hotel characteristics.

## Tech Stack

- **Python 3**
- **pandas** - Data loading, manipulation, groupby operations, filtering, column reordering
- **NumPy** - Numerical operations
- **NLTK (Natural Language Toolkit)** 
  - Stopwords corpus (179 common English words)
  - VADER sentiment analyzer for social media text
  - Sentiment intensity scoring
- **TextBlob** - Alternative sentiment analysis library (supporting scripts)
- **time** - Performance measurement and monitoring

## How to Run

### Prerequisites
Install required libraries:
```bash
pip install pandas numpy nltk textblob
```

Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

### Step 1: Data Exploration & Cleaning
1. Open `HotelReview.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load Hotel_Reviews.csv (note: this is a large file, ~50MB)
   - Analyze reviewer nationality and hotel frequency distributions
   - Explore review scores and calculate averages
   - Identify data quality issues and missing values
   - Process and standardize hotel addresses
   - Create guest/trip type binary features
   - Generate Hotel_Reviews_Filtered.csv
3. Expected output: Cleaned dataset with ~515,000 rows and standardized features

### Step 2: Sentiment Analysis & NLP Processing
1. Open `sentimentAnalysis.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load Hotel_Reviews_Filtered.csv
   - Remove stop words from review text
   - Apply VADER sentiment analysis to positive and negative reviews
   - Calculate sentiment compound scores (-1 to +1)
   - Validate sentiment against reviewer scores
   - Reorder columns for analysis
   - Save Hotel_Reviews_NLP.csv (final processed dataset)
3. Expected output: Final NLP-processed dataset ready for recommendation system

### Using Processed Data
The `Hotel_Reviews_NLP.csv` output contains:
- Hotel information (name, address, review counts, average scores)
- Reviewer information (nationality, score, sentiment scores)
- Guest characteristics (trip type, group type, pet, family status)
- Processed reviews (negative and positive text without stop words)

This data is ready for:
- Hotel recommendation system development
- Sentiment-based hotel ranking
- Demographic analysis and segmentation
- Chatbot training (see supporting scripts)

## Project Files

- **HotelReview.ipynb** - Data exploration and cleaning notebook: loads raw data, analyzes distributions, standardizes addresses, creates features
- **sentimentAnalysis.ipynb** - NLP and sentiment analysis notebook: stop word removal, VADER sentiment analysis, final processing
- **Hotel_Reviews_Filtered.csv** - Intermediate output from HotelReview.ipynb (cleaned, with features)
- **Hotel_Reviews_NLP.csv** - Final output (processed with sentiment scores, ready for use)
- **sentiment.py** - Example script showing TextBlob sentiment analysis
- **simplebot.py** - Supporting script for chatbot integration
- **bot2.py** - Alternative chatbot implementation
- **translation.py** - Language translation utilities (optional)

## Notes

- The VADER sentiment analyzer is specifically tuned for social media text and informal language, making it ideal for review text
- Some sarcasm and complex language may not be accurately detected by sentiment analyzers; always validate results
- Stop word removal speeds up processing while maintaining sentiment accuracy
- The dataset contains real-world data with varying data quality; some address and score inconsistencies exist
- Processing 515,000 reviews takes several minutes; performance depends on system resources
