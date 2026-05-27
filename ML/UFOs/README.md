# UFO Sightings Geographic Classifier

## Overview

This project builds a machine learning classifier that predicts the country of origin for UFO sightings based on spatial and temporal characteristics. Using 80,000 historical UFO sighting reports from NUFORC (National UFO Reporting Center), the notebook walks through the complete machine learning lifecycle: data loading, exploration, cleaning, feature engineering, model training, evaluation, and model serialization. The trained classifier is deployed in a Flask web application where users can input sighting characteristics (duration, latitude, longitude) and receive a country prediction. The project demonstrates that geographic location is a strong predictor of where UFO sightings are reported—a practical application of geospatial classification.

## Dataset

**Source:** NUFORC (National UFO Reporting Center)  
**File:** ufos.csv  
**Total Records:** 80,000 UFO sightings from multiple countries

**Available Features:**
- **city** - City where sighting occurred
- **state** - State/province information
- **country** - Country of sighting
- **shape** - Reported shape of UFO object
- **duration (seconds)** - Duration of sighting in seconds
- **latitude** - Geographic latitude coordinate
- **longitude** - Geographic longitude coordinate
- **text** - Detailed description of sighting (e.g., "A man emerges from a beam of light..." or "the lights chased us")

**Data Characteristics:**
- Diverse sighting descriptions from brief reports ("the lights chased us") to detailed narratives
- Global coverage with sightings from multiple countries
- Temporal span covering years of reported sightings
- Varying sighting durations from seconds to hours

**Target Countries:** Australia, Canada, Germany, United Kingdom, United States

## Methods

### Data Exploration & Loading (ufos.ipynb)
- Load complete UFO dataset (80,000 sightings)
- Display sample records to understand data structure
- Examine available columns and data types
- Explore example sighting descriptions for context
- Check unique countries in the dataset

### Data Cleaning & Preprocessing

**Feature Selection:**
Selected three features for geographic classification:
1. **Seconds** - Sighting duration (1-60 seconds)
2. **Latitude** - Geographic latitude of sighting
3. **Longitude** - Geographic longitude of sighting

**Filtering:**
- **Duration Range:** Filter to sightings lasting 1-60 seconds (realistic sighting lengths)
  - Removes very short observations (likely false positives)
  - Removes extremely long sightings (likely different phenomena or misreporting)
- **Missing Values:** Drop rows with null/missing values in any column
- **Result:** Cleaned dataset ready for modeling

**Categorical Encoding:**
- Convert country names (text) to numeric labels using **LabelEncoder**
- Alphabetical encoding: Australia→0, Canada→1, Germany→2, UK→3, US→4
- Enables Logistic Regression to work with categorical target variable

### Model Architecture

**Algorithm:** Logistic Regression
- **Type:** Multi-class classifier (handles 5 countries)
- **Hyperparameters:**
  - **max_iter=1000** - Maximum iterations for solver convergence
- **Input Features:** Seconds, Latitude, Longitude (3 features)
- **Output:** Predicted country class (0-4)

**Why Logistic Regression:**
- Interpretable: Provides probability estimates for each class
- Efficient: Fast training on 80,000+ records
- Baseline: Good starting point before exploring complex models
- Geographic Intuition: Latitude/Longitude alone are strong country predictors

### Training & Evaluation

**Data Splitting:**
- **Train Set:** 80% of cleaned data
- **Test Set:** 20% of cleaned data
- **Random State:** 0 (reproducible splits)

**Metrics:**
- **Accuracy:** Overall proportion of correct predictions
- **Classification Report:** Per-country precision, recall, F1-score
  - **Precision:** When model predicts a country, how often is it correct?
  - **Recall:** Of all sightings from a country, how many does the model identify?
  - **F1-Score:** Harmonic mean balancing precision and recall

### Model Deployment

**Serialization:**
- Model pickled and saved as `ufo_model.pkl`
- Enables loading in production without retraining
- Small file size for easy distribution

**Web Application (web-app/):**
- **Backend:** Flask application (app.py)
  - Route `/` serves HTML form for user input
  - Route `/predict` accepts POST requests with features
  - Loads pickled model and generates predictions
  - Maps numeric predictions to readable country names
  
- **Country Mapping:** [Australia, Canada, Germany, UK, US]
  - Returns human-readable country name in web interface
  - User-friendly instead of numeric class labels

- **Frontend:** HTML form (templates/index.html)
  - Input fields: Duration (seconds), Latitude, Longitude
  - Submit button to trigger prediction
  - Displays predicted country
  
- **Styling:** CSS (static/css/styles.css)
  - Professional appearance
  - Responsive layout

## Results

### Model Performance
- **Trained Logistic Regression:** Multi-class classifier for 5 countries
- **Classification Report:** Shows per-country metrics (Precision, Recall, F1)
- **Overall Accuracy:** Percentage of correct country predictions on test set
- **Key Insight:** Geographic location (latitude/longitude) alone is a strong predictor of country—the model leverages this geospatial structure

### Predictions
- **Example Prediction:** [50 seconds, latitude 44, longitude -12] → Predicted country
- Model successfully maps sighting characteristics to countries
- Can be used to infer country from any sighting's duration and coordinates

### Model Serialization
- Successfully pickled model for production deployment
- Model loaded from pickle file and verified working
- Ready for web application integration

## Tech Stack

- **Python 3**
- **pandas** - Load CSV, create DataFrames, data exploration
- **NumPy** - Numerical array operations
- **scikit-learn**
  - LabelEncoder for country categorical encoding
  - train_test_split for 80/20 data partition
  - LogisticRegression for multi-class classification
  - accuracy_score for overall accuracy metric
  - classification_report for per-class metrics (precision, recall, F1)
- **Flask** - Web framework for model deployment
- **pickle** - Model serialization/deserialization
- **HTML/CSS** - Web interface for user interaction

## How to Run

### Prerequisites
Install required libraries:
```bash
pip install pandas numpy scikit-learn flask
```

### Project 1: Model Training & Evaluation (ufos.ipynb)

1. Open `ufos.ipynb` in Jupyter Notebook
2. Ensure `ufos.csv` is in the same directory
3. Run all cells to:
   - Load 80,000 UFO sighting records
   - Filter to 1-60 second sightings
   - Remove null values
   - Encode country names to numeric labels
   - Create Seconds, Latitude, Longitude features
   - Split data into 80% train / 20% test
   - Train Logistic Regression model
   - Generate classification report with per-country metrics
   - Print accuracy and sample predictions
   - Pickle model as `ufo_model.pkl`
   - Test pickled model with example input [50, 44, -12]
4. **Output:**
   - Classification report (precision, recall, F1 per country)
   - Overall accuracy score
   - Pickled model file (`ufo_model.pkl`)
5. **Time Required:** <2 minutes

### Project 2: Web Application Deployment (web-app/)

1. Copy `ufo_model.pkl` from notebook directory to `web-app/` folder
   - Ensure model file is in same directory as `app.py`

2. Navigate to web-app directory:
   ```bash
   cd web-app
   ```

3. Install Flask dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run Flask application:
   ```bash
   python app.py
   ```

5. Access web interface:
   - Open browser to `http://127.0.0.1:5000`
   - Enter sighting characteristics:
     - **Seconds:** Duration of sighting (1-60 seconds)
     - **Latitude:** Geographic latitude
     - **Longitude:** Geographic longitude
   - Click predict button
   - View predicted country

6. **Output:**
   - Web form interface
   - Predictions in format: "Likely country: [Australia/Canada/Germany/UK/US]"

7. **Time Required:** <1 minute to setup and first prediction

### Example Predictions

Try these example inputs in the web app:

| Seconds | Latitude | Longitude | Expected Country |
|---------|----------|-----------|------------------|
| 30      | 40.71    | -74.01    | US               |
| 45      | 51.51    | -0.13     | UK               |
| 50      | -33.87   | 151.21    | Australia        |
| 52      | 48.86    | 2.29      | Germany          |
| 40      | 43.65    | -79.38    | Canada           |

## Key Insights

### Geographic Patterns
- **Strong Geographic Signal:** Latitude and longitude alone are highly predictive of sighting country
- **Clustered Sightings:** UFO sightings concentrate in specific geographic regions per country
- **Country Boundaries:** Model implicitly learns geographic boundaries between countries

### Model Characteristics
- **Efficient:** Logistic Regression trains quickly on large datasets
- **Interpretable:** Can examine feature weights to understand country-location relationships
- **Practical:** Achieves good accuracy with just 3 simple features (duration, lat, lon)

### Data Insights
- **Duration Distribution:** Most sightings last 1-60 seconds (filter removes outliers)
- **Five Countries:** Australia, Canada, Germany, UK, US represent major reporting countries
- **NUFORC Coverage:** Primarily English-speaking countries with active UFO reporting communities

## Project Files

- **ufos.ipynb** - Main notebook with complete ML pipeline: load, explore, clean, train, evaluate, pickle
- **ufo_model.pkl** - Serialized trained Logistic Regression model (generated by notebook)
- **ufo_model1.pkl** - Alternative pickled model version
- **ufos.csv** - NUFORC dataset with 80,000 UFO sighting records
- **web-app/app.py** - Flask application with `/` (home) and `/predict` (inference) routes
- **web-app/requirements.txt** - Python dependencies (scikit-learn, pandas, numpy, flask)
- **web-app/templates/index.html** - HTML form for user input and prediction display
- **web-app/static/css/styles.css** - CSS styling for web interface
- **README.md** - This file

## Extensions & Future Work

- **Multi-Step Prediction:** Predict additional features (shape, state/province) from coordinates
- **Confidence Scores:** Return probability for each country instead of single prediction
- **Temporal Analysis:** Incorporate sighting date/time to detect temporal patterns
- **Shape Classification:** Classify reported UFO shape (circle, triangle, etc.) by location
- **Interactive Map:** Plot sightings on world map with model predictions
- **Ensemble Models:** Compare Logistic Regression with Random Forest, SVM, Neural Networks
- **API Deployment:** Deploy as REST API (AWS Lambda, Google Cloud Functions)
- **Real-Time Updates:** Automatically retrain on new NUFORC sighting data
- **Sighting Clustering:** Use unsupervised learning to identify hotspots and temporal clusters
