# Analysis of Media Ratings Dataset

## Dataset Overview
The dataset comprises 2,652 entries of various media titles, including movies, each characterized by attributes such as the date of release, language, type, title, contributors, and ratings in three categories: overall, quality, and repeatability. This rich dataset allows for a multifaceted exploration of audience perceptions and trends in media consumption.

## Analysis Conducted
I undertook a series of analyses to derive insights from the data. These analyses included:

1. **Generic Analysis**: I assessed missing values, summary statistics, unique value counts, and performed correlation analysis to understand relationships between numerical features.
   
2. **Time Series Analysis**: I examined the trends over time concerning overall ratings, identifying patterns in audience sentiments from 2006 to 2024.

3. **Classification Analysis**: I predicted repeatability ratings and evaluated the model performance using metrics like accuracy, precision, and recall.

4. **Dynamic Analysis**: I analyzed the dataset for missing values, skewness, and correlations among variables while encoding categorical data.

### Visualizations
#### Correlation Heatmap
![Correlation Heatmap](chart_1.png)

#### Time Series Visualization
![Time Series of Overall Ratings](chart_2.png)

#### Classification Confusion Matrix
![Classification Confusion Matrix](chart_3.png)

## Insights Discovered
The analysis unveiled several key insights:

- **Completeness of Data**: There were no missing values in any columns, indicating a robust dataset ready for analysis.
  
- **Correlation Insights**: The strong correlation between overall and quality ratings (0.83) suggests that improving content quality could elevate audience perceptions, as these ratings are closely intertwined.

- **Trends Over Time**: The time series analysis showed fluctuations in overall ratings, with a subtle increase in recent years, indicating a potential resurgence in audience engagement or higher quality releases.

- **Classification Challenges**: The classification analysis revealed solid predictive ability for repeatability levels but indicated difficulties in accurately predicting mid-range ratings.

## Implications of Findings
The insights derived from this analysis have several implications:

1. **Targeted Quality Improvement**: Media organizations should focus on enhancing quality content, as it is likely to improve overall ratings and audience satisfaction collectively.

2. **Monitoring Trends**: Continuous monitoring of audience ratings over time can provide valuable feedback to creators and marketers, helping them align content offerings with emerging audience preferences.

3. **Refining Targeting Strategies**: Given the classification challenges, further modeling could improve accuracy, enabling more strategic marketing efforts based on predicted repeatability ratings.

4. **Data-Driven Decision Making**: The findings suggest adopting a data-driven approach to media development and marketing strategies, utilizing relationship insights to inform future projects.

In conclusion, the analysis not only highlighted the dataset's integrity and the value of its information but also pointed to actionable strategies for improving content and enhancing audience engagement in the media landscape.

