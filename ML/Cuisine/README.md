# Cuisine Classification

## Overview

This project builds a multi-class machine learning classifier that predicts cuisine types from recipe ingredients. Using a dataset of 2,000+ recipes with 380+ ingredient features, the project tackles the real-world challenge of imbalanced data using SMOTE, trains multiple classification algorithms to find the best performer, and exports the final model to ONNX format for deployment in a Flask web application. This demonstrates a complete ML pipeline from data cleaning to production-ready deployment.

## Dataset

**Source:** Cuisines dataset  
**Files:** 
- `cuisines.csv` - Raw dataset with ~2,000 recipes
- `cleaned_cuisines.csv` - Balanced dataset after preprocessing

**Size:** 2,000+ recipes (5 cuisines: Thai, Japanese, Chinese, Indian, Korean)  
**Features:** 380+ binary ingredient indicators (0 = not present, 1 = present in recipe)

The dataset exhibits significant class imbalance, with Indian cuisine having more samples than others. Common ingredients like rice, garlic, and ginger were removed during preprocessing as they appear across multiple cuisines and add noise to the classifier.

## Methods

### Data Exploration & Preprocessing
- Loaded and profiled recipes across 5 cuisine categories
- Identified class imbalance using value counts and visualizations
- Analyzed ingredient distributions per cuisine using bar plots
- Extracted and ranked most common ingredients per cuisine type

### Feature Engineering & Balancing
- Removed confusing/common ingredients (rice, garlic, ginger) that appear across cuisines
- Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset
- Generated synthetic samples for underrepresented cuisines through interpolation
- Standardized class distribution for better model training

### Model Selection & Training
Trained five different classifiers and compared performance:
1. **Logistic Regression** (multi_class='ovr', solver='liblinear')
2. **Linear SVC** (Support Vector Classifier with kernel='linear', C=10)
3. **K-Neighbors Classifier**
4. **Random Forest Classifier** (100 estimators)
5. **AdaBoost Classifier** (100 estimators)

Used 70/30 train-test split and evaluated each model with accuracy, precision, and classification reports.

### Model Export for Deployment
- Selected the best-performing Linear SVC model
- Converted scikit-learn model to ONNX format using skl2onnx
- Configured opset version 12 for broad web compatibility
- Validated ONNX model structure and inputs/outputs

## Results

The project successfully built a multi-class cuisine classifier with strong performance:
- **Best Model:** Linear SVC with high accuracy across all five cuisine types
- **Classification Performance:** Detailed precision, recall, and F1-scores per cuisine
- **Model Deployment:** Successfully converted to ONNX format (cuisine_model1.onnx) for web deployment
- **Web Integration:** ONNX model integrated into Flask web application for real-time cuisine predictions

Individual recipe testing demonstrated the model correctly predicts cuisine based on ingredient lists, with probability scores showing confidence levels for each cuisine class.

## Tech Stack

- **Python 3**
- **pandas** - Data loading, manipulation, and analysis
- **NumPy** - Numerical operations and array handling
- **scikit-learn** - Logistic Regression, SVC, KNeighborsClassifier, RandomForestClassifier, AdaBoostClassifier, train_test_split, classification metrics
- **imbalanced-learn** - SMOTE for handling imbalanced data
- **skl2onnx** - Converting scikit-learn models to ONNX format
- **ONNX** - Model validation and cross-platform deployment
- **matplotlib** - Static visualization
- **Flask** - Web framework for the interactive application
- **HTML/CSS** - Frontend for the web app

## How to Run

### Prerequisites
Ensure you have Python and the required libraries installed:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib skl2onnx onnx flask
```

### Steps for Model Training & Analysis
1. Open `cuisines.ipynb` in Jupyter Notebook
2. Run all cells in sequence to:
   - Load and explore the cuisines dataset
   - Analyze ingredient distributions per cuisine
   - Balance the data using SMOTE
   - Train and compare multiple classifiers
   - Generate classification reports and accuracy metrics
3. The notebook will output the cleaned dataset (`cleaned_cuisines.csv`) for use in the web app

### Steps for Model Conversion
1. Open `CuisinesWebApp/Cuicuisine.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load the cleaned and balanced dataset
   - Train the SVC classification model
   - Convert the model to ONNX format
   - Validate the ONNX model structure
3. The notebook will generate `cuisine_model1.onnx` for deployment

### Running the Web Application
1. Navigate to the `CuisinesWebApp/` directory
2. Ensure `cleaned_cuisines.csv` and `cuisine_model1.onnx` are in the same directory as `app.py`
3. Start the Flask app:
   ```bash
   python app.py
   ```
4. Open your browser to `http://localhost:5000` and use the web interface to predict cuisines from ingredient lists

## Project Files

- **cuisines.ipynb** - Main analysis notebook: data exploration, ingredient analysis, model training comparison
- **cleaned_cuisines.csv** - Balanced dataset after SMOTE preprocessing (generated by cuisines.ipynb)
- **CuisinesWebApp/Cuicuisine.ipynb** - Model conversion notebook: trains SVC and exports to ONNX
- **CuisinesWebApp/app.py** - Flask web application for interactive cuisine predictions
- **CuisinesWebApp/cuisine_model1.onnx** - Trained SVC model in ONNX format
- **CuisinesWebApp/index.html** - Web interface HTML
- **CuisinesWebApp/static/css/styles.css** - Web interface styling
