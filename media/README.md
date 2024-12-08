# Dataset Description

The dataset `media.csv` contains information about various media, specifically movies, with the following features:

- **date**: The release date in a string format.
- **language**: The language of the movie (e.g., Tamil, Telugu).
- **type**: The type of media (all entries appear to be movies).
- **title**: The title of the movie.
- **by**: The names of the actors or creators involved.
- **overall**: Overall rating (on a scale from 1 to 5).
- **quality**: Quality rating (on a scale from 1 to 5).
- **repeatability**: A score indicating how likely the media is to be watched again (1-3).

## Statistical Analysis Summary

### Summary Statistics
- The dataset contains **2652 entries**.
- Ratings (overall, quality, repeatability) show a mean value around **3** with a standard deviation indicating variability.
- Quality ratings have the highest average score (mean = 3.21), while overall ratings are slightly lower (mean = 3.05).

### Missing Values
- The column `date` has **99 missing values** while `by` has **262 missing values**. Other columns do not have missing values.

### Unique Values
- There are **2055 unique dates**, indicating a diverse time span for media releases. 
- The `title` column has **2312 unique entries**, showing a wide variety of movies.

### Duplicates
- The dataset contains **1 duplicated row**, which may require removal for accurate analysis.

### Correlation Analysis
- Strong positive correlation between **overall** and **quality** ratings (0.83), suggesting that higher quality often corresponds with higher overall ratings.
- Some correlation exists between **overall** and **repeatability** (0.51) indicating that more highly rated movies may be more likely to be re-watched.

# Outlier Detection Analysis

## Analysis Overview
The outlier detection analysis was conducted on the dataset `media.csv`, which consists of 2,652 samples. The objective of the analysis was to identify and quantify anomalies present in the dataset.

## Key Findings
- **Number of Anomalies Detected**: 725
- **Total Samples Analyzed**: 2,652

## Insights
The analysis revealed that approximately 27.3% (725 out of 2,652) of the samples are considered outliers. This proportion is significant and suggests that a substantial portion of the data might not conform to the expected patterns or behaviors. 

### Implications
- **Data Integrity**: The high number of anomalies indicates potential issues in data integrity, such as errors in data entry, sensor malfunctions, or unusual behavior that may need further investigation.
- **Segment Analysis**: Further analysis could involve segmenting the dataset based on features to understand the characteristics of both normal and anomalous data points. 

### Conclusion
The outlier detection analysis highlights a need for deeper scrutiny of the identified anomalies. This could involve exploring the underlying causes and determining whether they represent valuable insights or require correction. By addressing these outliers, data quality and subsequent analyses can be improved.

# Correlation Analysis of Media.csv

## Analysis Overview
The analysis conducted involves a correlation examination between three key variables: overall, quality, and repeatability. The resulting correlation matrix indicates the strength and direction of the relationships among these variables.

## Findings from the Correlation Analysis

### Correlation Matrix
|                | overall  | quality  | repeatability |
|----------------|----------|----------|---------------|
| overall        | 1.000000 | 0.825935 | 0.512600      |
| quality        | 0.825935 | 1.000000 | 0.312127      |
| repeatability   | 0.512600 | 0.312127 | 1.000000      |

### Insights:
1. **High Correlation**:
   - The correlation between **overall** and **quality** is strong (0.83). This suggests that higher quality ratings lead to higher overall ratings.
   
2. **Moderate Correlation**:
   - The correlation between **overall** and **repeatability** is moderate (0.51), indicating a noteworthy but less pronounced influence on overall ratings compared to quality.
   
3. **Weak Correlation**:
   - The correlation between **quality** and **repeatability** is weak (0.31), implying that quality and repeatability operate more independently of each other.

## Conclusion
- Prioritizing improvements in quality is likely the most effective strategy to boost overall ratings.
- While repeatability shows some influence over overall satisfaction, its lower correlation suggests that enhancements in quality are more impactful.
- Addressing repeatability should still be considered but may require distinct approaches separate from overall quality improvement efforts.

![Correlation Analysis Chart](chart_1.png)

# Regression Analysis on Media Data

## Analysis Overview
The regression analysis was conducted on a dataset named `media.csv`, focusing on predicting the target variable `repeatability` based on two input features: `overall` and `quality`. The following metrics were evaluated to measure the performance of the regression model:

- **R² Score**: 0.28
- **Mean Absolute Error (MAE)**: 0.42
- **Mean Squared Error (MSE)**: 0.26
- **Root Mean Squared Error (RMSE)**: 0.51
- **Coefficients**: `[0.48, -0.21]`
- **Intercept**: 1.50

## Key Observations

1. **Reference Line**: The line \( y = x \) serves as a benchmark for perfect predictions. Points on this line indicate accurate predictions and help assess the model's performance.

2. **Data Distribution**: The predicted values (blue dots) show a mixed performance with some closely aligned to actual values while others deviate significantly.

3. **Clustering**: Clusters of points indicate both strong predictions and notable discrepancies. The range of values for both actual and predicted outcomes lies primarily between 0.5 and 3.0.

### Insights

- **Strong Predictions**: Several points near the reference line demonstrate instances where the model effectively predicts the repeatability metric.
  
- **Inconsistencies**: Significant deviations from the reference line highlight areas where the model's predictions can be improved.

- **Model Performance**: The overall distribution indicates that while the model is adequate in certain ranges, some values require further investigation for potential enhancements.

## Conclusions

To further boost the model's predictive capabilities, examining outlier predictions will be essential. Identifying underlying factors leading to inaccuracies could provide insights for model refinement or the inclusion of additional features.

![Actual vs Predicted](chart_2.png)

