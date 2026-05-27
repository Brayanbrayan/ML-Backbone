# Time Series Forecasting: Electricity Load Prediction

## Overview

This project demonstrates three complementary approaches to time series forecasting for predicting hourly electricity load. Using 3 years of energy consumption data from the GEFCom2014 competition, the project compares statistical (ARIMA) and machine learning (SVR) techniques side-by-side. The notebooks showcase the complete workflow: data exploration and visualization, data preprocessing, model training with different approaches, and performance evaluation. This is a practical demonstration of how traditional statistical methods and modern ML techniques can solve the same real-world forecasting problem with different strengths and weaknesses.

## Dataset

**Source:** GEFCom2014 Energy Forecasting Competition  
**File:** `energy.csv`  
**Time Range:** January 1, 2012 - December 31, 2014 (3 years)  
**Frequency:** Hourly measurements  
**Size:** 26,280 hourly records

**Features:**
- **load** - Hourly electricity consumption (in MW or similar units)
- **temperature** - Hourly temperature (used in some analyses)

**Train/Test Split:**
- **Training Period:** September 1 - October 31, 2014 (2 months)
- **Testing Period:** November 1 - December 31, 2014 (2 months)
- **Rationale:** Short training windows are sufficient for stable forecasting with sufficient data

The dataset exhibits clear hourly, daily, and seasonal patterns typical of electricity demand: lower night consumption, peaks during business hours, and seasonal variation across months.

## Methods

### Data Exploration & Visualization (timeries.ipynb)
- Load 3 years of hourly electricity data
- Plot complete time series to understand patterns
- Examine weekly patterns (e.g., July 1-7 sample week)
- Identify temporal seasonality: hourly, daily, weekly, and yearly cycles
- Baseline understanding for both statistical and ML approaches

### Data Preprocessing
**Scaling:** MinMaxScaler normalization to [0, 1] range
- Fitted on training data only
- Applied to test data using fitted scaler
- Prevents data leakage and ensures models learn on normalized values

**Filtering:** Select specific time periods for train/test sets
- Remove irrelevant features and timestamps
- Focus on load column only

**Time-Step Transformation (SVR only):**
- Convert 1D time series into 2D supervised learning format
- Create sliding windows of 5 consecutive hours
- Map: 4 previous hours → 1 future hour prediction
- Enable SVR to capture temporal dependencies

### Approach 1: ARIMA/SARIMAX (arima.ipynb)

**Model Type:** Seasonal AutoRegressive Integrated Moving Average

**Parameters:**
- **ARIMA Order (p,d,q):** (4, 1, 0)
  - p=4: 4 autoregressive lags (use last 4 hours)
  - d=1: First-order differencing (handle non-stationarity)
  - q=0: No moving average terms
  
- **Seasonal Order (P,D,Q,s):** (1, 1, 0, 24)
  - P=1: 1 seasonal autoregressive lag
  - D=1: Seasonal differencing
  - Q=0: No seasonal moving average
  - s=24: 24-hour seasonality (daily cycle)

**Training Strategy:**
- Fixed 720-hour window (30 days) for each model
- Iterative sliding window: train → predict → slide window
- Refit model at each test timestep
- Multi-step forecasting: 3-hour horizon (t+1, t+2, t+3)

**Strengths:**
- Excellent for linear, stationary patterns
- Interpretable parameters and seasonal handling
- Built on statistical theory with confidence intervals

**Weaknesses:**
- Assumes linearity (electricity load has non-linear patterns)
- Sensitive to parameter selection
- Computationally expensive with frequent refitting

### Approach 2: Support Vector Regression (SVR.ipynb)

**Model Type:** Non-linear Support Vector Machine for Regression

**Architecture:**
- Input: 4 consecutive hours of electricity load
- Output: 1-hour ahead prediction
- Kernel: RBF (Radial Basis Function)

**Hyperparameters:**
- **kernel='rbf'**: Non-linear kernel to capture complex patterns
- **gamma=0.5**: RBF kernel coefficient (controls complexity)
- **C=10**: Regularization parameter (penalty for errors)
- **epsilon=0.05**: Epsilon-tube for error tolerance

**Training Strategy:**
- Train once on 2-month window
- Single model for all predictions
- Single-step forecasting (t+1 only)
- No model refitting

**Strengths:**
- Handles non-linearity in electricity load patterns
- Single model training (efficient)
- Good generalization with proper regularization
- Memory-efficient with support vectors

**Weaknesses:**
- Less interpretable than statistical models
- Requires hyperparameter tuning
- Single-step predictions only (not multi-step)
- Difficult to extrapolate beyond training range

### Evaluation Metrics

**MAPE (Mean Absolute Percentage Error):**
$$MAPE = \frac{1}{n} \sum_{i=1}^{n} \left|\frac{actual_i - predicted_i}{actual_i}\right| \times 100\%$$

**Advantages:**
- Scale-independent (works across different load ranges)
- Interpretable (percentage error)
- Symmetric (treats over/under-predictions equally)

**Per-Horizon Analysis:**
- ARIMA: Separate MAPE for t+1, t+2, t+3
- SVR: Single-step MAPE

## Results

### ARIMA/SARIMAX Forecasting
- **One-Step Forecast MAPE:** Calculated and reported
- **Multi-Step Forecast MAPE:** Accuracy decreases with horizon (t+1 best, t+3 worst)
- **Pattern Capture:** Excellent at capturing seasonal 24-hour cycle
- **Behavior:** Predictions tend toward mean as forecast horizon increases
- **Visual:** Multi-step predictions shown as decreasing confidence bands

### Support Vector Regression
- **Training MAPE:** Measured on full 2-month training set
- **Test MAPE:** Measured on full 2-month test set
- **Pattern Capture:** Captures non-linear variations in load
- **Single-Step:** Focuses on immediate next-hour prediction
- **Consistency:** More stable predictions across time without horizon decay

### Comparative Insights
- **ARIMA:** Better for understanding seasonal patterns and statistical forecasting
- **SVR:** Better for handling non-linear electricity load variations
- **Computational:** SVR more efficient (single training), ARIMA iterative (many retrain cycles)
- **Interpretability:** ARIMA more interpretable, SVR more of a "black box"
- **Multi-Step:** ARIMA naturally handles multi-step, SVR requires separate models per step

## Tech Stack

- **Python 3**
- **pandas** - Time series data loading, resampling to hourly frequency, date indexing, filtering
- **NumPy** - Array operations, time-step reshaping, mathematical operations
- **Matplotlib** - Time series visualization, forecasting plots with multiple lines
- **scikit-learn** 
  - MinMaxScaler for data normalization
  - SVR for Support Vector Regression
- **statsmodels** 
  - SARIMAX for ARIMA and Seasonal ARIMA modeling
  - Visualization for autocorrelation analysis
- **Custom utils module (common/)** - load_data() for GEFCom data, mape() for evaluation

## How to Run

### Prerequisites
Install required libraries:
```bash
pip install pandas numpy matplotlib scikit-learn statsmodels
```

### Project 1: Data Exploration & Time Series Basics (timeries.ipynb)

1. Open `timeries.ipynb` in Jupyter Notebook
2. Ensure `energy.csv` is in the same directory
3. Run all cells to:
   - Load 3 years of electricity load data
   - Plot complete time series (2012-2014)
   - Visualize weekly patterns (sample week: July 1-7, 2014)
   - Understand hourly, daily, and seasonal patterns
4. **Output:** Visual understanding of time series structure and patterns
5. **Time Required:** <1 minute

### Project 2: ARIMA/SARIMAX Forecasting (arima.ipynb)

1. Open `arima.ipynb` in Jupyter Notebook
2. Ensure `energy.csv` is in same directory and `common/utils.py` is accessible
3. Run all cells to:
   - Load and filter data for Sept-Dec 2014
   - Split into training (Sept-Oct) and testing (Nov-Dec)
   - Scale data using MinMaxScaler (fitted on training)
   - Define SARIMAX model with order (4,1,0) and seasonal (1,1,0,24)
   - Train and predict using 720-hour sliding window
   - Evaluate with MAPE for 1-step, 2-step, 3-step horizons
   - Plot predictions vs. actual (multi-step horizon visualization)
4. **Output:** 
   - Forecast MAPE scores per horizon
   - Visual comparison of actual vs. predicted load
   - Statistical summary of model fit
5. **Time Required:** 5-10 minutes (depends on dataset size)

### Project 3: Support Vector Regression Forecasting (SVR.ipynb)

1. Open `SVR.ipynb` in Jupyter Notebook
2. Ensure `energy.csv` is in same directory and `common/utils.py` is accessible
3. Run all cells to:
   - Load and filter data for Sept-Dec 2014
   - Split into training and testing sets
   - Scale data with MinMaxScaler
   - Create 5-hour time-step tensors (4 input → 1 output)
   - Train SVR with RBF kernel (gamma=0.5, C=10, epsilon=0.05)
   - Make predictions on training and test sets
   - Inverse-scale predictions to original units
   - Evaluate with MAPE on both sets
   - Plot training and test predictions
4. **Output:**
   - Training and test MAPE scores
   - Visual comparison of actual vs. predicted for both periods
   - Learned SVR model parameters
5. **Time Required:** <2 minutes

### Comparing Results

After running all three notebooks:
1. Compare MAPE scores between ARIMA and SVR
2. Examine multi-step vs. single-step forecasting trade-offs
3. Analyze computational efficiency (ARIMA refitting vs. SVR single train)
4. Visualize both approaches' predictions side-by-side
5. Consider which approach is better for different forecasting scenarios

## Key Insights

**ARIMA Strengths:**
- Excellent seasonal pattern capture (24-hour daily cycle)
- Principled statistical framework
- Natural multi-step forecasting
- Confidence intervals available

**ARIMA Weaknesses:**
- Assumes linearity (electricity demand has non-linear variation)
- Frequent retraining computationally expensive
- Parameter selection can be subjective

**SVR Strengths:**
- Handles non-linearity in electricity demand
- Single efficient training
- Good generalization
- No manual parameter selection for seasonality

**SVR Weaknesses:**
- Less interpretable
- Requires hyperparameter tuning
- Single-step by default (multi-step requires multiple models)
- May not capture seasonal patterns as explicitly

## Project Files

- **timeries.ipynb** - Data exploration: load 3-year dataset, visualize patterns, understand time series structure
- **arima.ipynb** - ARIMA/SARIMAX approach: seasonal model training, multi-step forecasting, MAPE evaluation
- **SVR.ipynb** - Support Vector Regression: time-step transformation, single-model training, visualization
- **energy.csv** - GEFCom2014 dataset (3 years hourly electricity load)
- **common/utils.py** - Utility functions: load_data(), mape() for evaluation
- **common/__init__.py** - Python package initialization

## Extensions & Future Work

- **Hybrid Models:** Combine ARIMA and SVR for enhanced forecasting
- **Deep Learning:** LSTM/GRU for longer-term multi-step forecasting
- **Exogenous Variables:** Include temperature, day-of-week, holidays
- **Real-Time:** Implement rolling window updates for live forecasting
- **Uncertainty:** Add confidence intervals and probabilistic forecasts
- **Multiple Locations:** Forecast for multiple substations/regions
- **Demand Optimization:** Use forecasts for energy scheduling and grid management
