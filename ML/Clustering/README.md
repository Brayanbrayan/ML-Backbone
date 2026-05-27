# Nigerian Music Clustering Analysis

## Overview

This project applies K-Means clustering to analyze audio characteristics of Nigerian music from Spotify. By examining features like danceability, energy, acousticness, and loudness, the clustering algorithm identifies natural groupings within the dataset and evaluates how well it can automatically discover three distinct music genres: afro dancehall, afropop, and nigerian pop. The project demonstrates both the technical clustering workflow and insights into Nigerian music preferences.

## Dataset

**Source:** Nigerian songs dataset from Spotify  
**File:** `nigerian-songs.csv`  
**Size:** ~500+ tracks after filtering  

The dataset contains 16 audio features per track including:
- **Metadata:** name, album, artist, artist_top_genre, release_date
- **Audio Features:** popularity, danceability, acousticness, energy, instrumentalness, liveness, loudness, speechiness, tempo, time_signature, length

The dataset was filtered to focus on three genres with sufficient data: afro dancehall, afropop, and nigerian pop, and songs with non-zero popularity values.

## Methods

### Data Exploration & Cleaning
- Loaded and profiled the dataset using pandas
- Identified and removed outliers using the Interquartile Range (IQR) method
- Analyzed data distributions with box plots and correlation heatmaps
- Discovered that energy and loudness show the strongest correlation

### Feature Engineering
- Selected relevant audio features: artist_top_genre, popularity, danceability, acousticness, loudness, energy
- Applied Label Encoding to convert categorical genre data to numeric format
- Standardized all features using StandardScaler for equal weighting during clustering

### Clustering Algorithm
- **K-Means Clustering** with k=3 (matching the number of genres in the filtered dataset)
- Tested cluster quality using the **Elbow Method** to visualize within-cluster sum of squares (WCSS)
- Evaluated model performance using **Silhouette Score**

### Visualization
- Correlation heatmaps to identify feature relationships
- Scatter plots of popularity vs. danceability colored by genre
- Elbow curve for determining optimal cluster count
- KDE (kernel density estimation) plots and FacetGrid visualizations showing genre distributions

## Results

The K-Means model successfully clustered the Nigerian music data into three groups and compared them against the true genre labels. Key findings:
- The algorithm achieved an accuracy score by matching predicted clusters with actual genres
- Silhouette score was computed to measure cluster separation quality
- Elbow curve analysis confirmed that 3 clusters is a reasonable choice for this dataset
- Scatter plot visualizations showed distinct separation between genres in popularity-danceability space

The analysis revealed that Nigerian music tastes may converge around specific levels of danceability across genres, as evidenced by concentric clustering patterns.

## Tech Stack

- **Python 3**
- **pandas** - Data loading, manipulation, and analysis
- **NumPy** - Numerical operations and array handling
- **scikit-learn** - K-Means clustering, StandardScaler, LabelEncoder, silhouette_score, metrics
- **Matplotlib** - Static plotting and visualization
- **Seaborn** - Statistical data visualization (box plots, bar plots, heatmaps, KDE plots)

## How to Run

### Prerequisites
Ensure you have Python and the required libraries installed:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Steps
1. Place the `nigerian-songs.csv` file in the same directory as the notebooks
2. Open either `NigerianMusicanalysis.ipynb` or `songs.ipynb` in Jupyter Notebook
3. Run all cells in sequence to:
   - Load and explore the data
   - Clean and preprocess features
   - Train the K-Means clustering model
   - Generate visualizations and evaluation metrics
4. The notebooks will output accuracy scores, silhouette scores, and various plots showing cluster results

### Key Notebooks
- **songs.ipynb** - Initial data exploration, visualization, and correlation analysis
- **NigerianMusicanalysis.ipynb** - Complete clustering pipeline including outlier removal, scaling, model training, and accuracy evaluation
