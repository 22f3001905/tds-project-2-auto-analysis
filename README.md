# Dataset Description

The provided dataset `media.csv` contains information about various media titles, primarily movies, with features including:

- **date**: Release date of the media in object format.
- **language**: Language of the media (e.g., Tamil, Telugu).
- **type**: Type of media (e.g., movie).
- **title**: Title of the media.
- **by**: Names of the main contributors (actors).
- **overall**: Overall rating (scale of 1 to 5).
- **quality**: Quality rating (scale of 1 to 5).
- **repeatability**: Indicator of repeatability (1 to 3).

The dataset contains a total of 2652 entries, with various unique values across features.

## Statistical Analysis Summary

1. **Summary Statistics**:
   - **Count**: All numerical fields are complete except for `by`, which has 262 missing values.
   - **Mean Ratings**:
     - Overall: 3.05
     - Quality: 3.21
     - Repeatability: 1.49
   - **Standard Deviation**: Indicates variability, particularly in overall and quality ratings.
   - **Minimum/Maximum Values**: Ratings range from 1 to 5, and repeatability from 1 to 3.
   - **Quartiles**: Most ratings cluster around 3 with a slight preference for higher quality.

2. **Missing Values**:
   - A total of 99 missing entries in the `date` column, and 262 in the `by` column, highlighting data quality issues.

3. **Unique Values**:
   - The dataset contains 2055 unique release dates, 11 languages, and 2312 titles, indicating diversity in the media represented.

4. **Duplicated Rows**:
   - Only 1 duplicated entry found, contributing negligible redundancy.

5. **Correlation Analysis**:
   - Strong positive correlation between **overall** and **quality** ratings (0.83), indicating that higher quality tends to associate with higher overall ratings.
   - Moderate correlation between **overall** and **repeatability** (0.51), implying that repeatable titles may also receive favorable overall ratings.
   - The correlation between **quality** and **repeatability** is weaker (0.31).

This analysis provides insight into the ratings and characteristics of the media in the dataset, useful for understanding trends and influences on media evaluations.

# Outlier Detection Analysis Insights

## Overview
The outlier detection analysis was conducted on the `media.csv` dataset, yielding the following results:

- **Number of Anomalies Detected:** 725
- **Total Number of Samples Analyzed:** 2652

## Insights

1. **Anomalies Representation:**
   - The analysis identified **725 anomalies** out of **2652 samples**, indicating that approximately **27.3%** of the data points are considered outliers. This is a significant portion, suggesting potential issues within the dataset or special cases worth investigating further.

2. **Potential Data Quality Issues:**
   - A high number of detected anomalies may indicate data quality issues such as measurement errors, data entry mistakes, or extreme variations in user behavior or events captured in the dataset. It prompts further examination to clean or validate these outliers.

3. **Need for Further Investigation:**
   - These outliers could represent unusual or critical events that require deeper analysis. Understanding why these anomalies exist can lead to insights regarding anomalies in media consumption patterns, operational inefficiencies, or incorrect data reporting.

4. **Impact on Decision Making:**
   - If decisions are derived from this dataset, accounting for these anomalies will be crucial. Ignoring them may lead to skewed insights and ineffective strategies, whether in user engagement metrics, ad spending, or content performance analysis.

## Purpose of Analysis
This outlier detection analysis was performed to:

- Identify and assess data irregularities that could impact the validity and reliability of insights drawn from the dataset.
- Enhance data quality by detecting abnormal observations before further analytical processing, ensuring more accurate predictive modeling and reporting.

Understanding the nature of these anomalies can assist stakeholders in making more informed decisions and improving processes related to media metrics and performance evaluations.

## Insights from Correlation Analysis of `media.csv`

### Correlation Matrix
The correlation analysis produced the following correlation matrix:

|                | overall   | quality   | repeatability |
|----------------|-----------|-----------|---------------|
| **overall**     | 1.000000  | 0.825935  | 0.512600      |
| **quality**     | 0.825935  | 1.000000  | 0.312127      |
| **repeatability** | 0.512600 | 0.312127  | 1.000000      |

### Key Findings

1. **Strong Correlation Between Overall and Quality**:
   - The correlation coefficient of **0.825935** indicates a strong positive relationship. This implies that as the quality of the media increases, the overall rating also tends to increase. 

2. **Moderate Correlation Between Overall and Repeatability**:
   - A correlation of **0.512600** suggests a moderate positive link between overall ratings and repeatability. This indicates that better repeatability is associated with higher overall satisfaction, albeit less strongly than quality.

3. **Weak Correlation Between Quality and Repeatability**:
   - A correlation of **0.312127** is relatively low, suggesting that improvements in quality do not significantly impact the repeatability of the media. This may indicate that these two attributes operate independently in this context.

### Reason for Performing Correlation Analysis
The correlation analysis was conducted to understand relationships between different attributes of the media dataset. By identifying these correlations, stakeholders can make informed decisions regarding improvements and investments in media quality and repeatability, potentially leading to better overall media performance and user satisfaction. This analysis serves as a foundation for further statistical exploration or predictive modeling.

## Insights from Regression Analysis on Media Data

### Analysis Overview
The regression analysis was performed on a dataset (`media.csv`) to understand the relationship between the input features `overall` and `quality` and the target variable `repeatability`. 

### Key Findings

1. **Model Performance**:
   - **R² Score**: `0.281`  
     This indicates that approximately 28.1% of the variance in the target variable (`repeatability`) can be explained by the model. This suggests that the model has limited predictive power and may not capture all the underlying relationships.

   - **Mean Absolute Error (MAE)**: `0.424`  
     The average absolute difference between predicted and actual values is about 0.424. This can be considered a measure of prediction accuracy—lower values are better.

   - **Mean Squared Error (MSE)**: `0.255`  
     This reflects the average of the squares of the errors, highlighting variability since larger errors are punished more. A value of 0.255 indicates moderate error.

   - **Root Mean Squared Error (RMSE)**: `0.505`  
     RMSE provides insight into the average magnitude of the error in the same units as the target variable. An RMSE of 0.505 shows that the predictions deviate moderately from the actual values.

2. **Coefficients**:
   - **Overall Coefficient**: `0.484`  
     This suggests that an increase in `overall` by one unit is associated with a 0.484 increase in `repeatability`, holding `quality` constant. This indicates a positive relationship.

   - **Quality Coefficient**: `-0.211`  
     Conversely, an increase in `quality` by one unit is associated with a 0.211 decrease in `repeatability`, holding `overall` constant. This indicates a negative relationship, suggesting that higher quality might correlate with lower repeatability.

3. **Intercept**: `1.498`  
   This value represents the expected mean value of `repeatability` when both `overall` and `quality` are zero.

### Purpose of Analysis
The primary goal of performing this regression analysis was to quantify the influence of `overall` and `quality` on `repeatability`. Understanding these relationships can be critical for stakeholders in making informed decisions about product selections and improvements. 

### Conclusion
The analysis reveals a complex interaction where `overall` positively impacts `repeatability`, while `quality` appears to have a counterintuitive negative effect. These insights could guide further investigations or data collection efforts to refine the model and uncover deeper insights.

