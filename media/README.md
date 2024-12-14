# Analysis of Media Dataset

## Overview of the Dataset

The dataset comprises 2,652 entries detailing various media content, including attributes such as date, language, type, title, contributors, and ratings (overall, quality, repeatability). The primary focus is on understanding how these features relate to viewer satisfaction.

## Analysis Carried Out

To explore the dataset, a series of analytical steps were undertaken:

1. **Generic Analysis:** This provided a foundational understanding of the dataset, revealing eight columns, including missing values, unique counts, and the types of data within each column. Notably, overall ratings ranged from 1 to 5, with a mean of approximately 3.05.

   ![Covariance Matrix Chart](chart_1.png)

2. **Regression Analysis:** A linear regression was performed with overall ratings as the target variable, using quality and repeatability as predictors. The resulting R² score of 0.745 indicated a strong relationship, suggesting that around 74.5% of the variance in overall ratings could be attributed to these features.

   ![Regression Analysis Chart](chart_2.png)

3. **Time Series Analysis:** This analysis focused on overall ratings over time, determining that the series is stationary, which means its statistical properties do not change over time. This conclusion was based on the ADF statistic and its comparison to critical values, ensuring consistency in the data.

   ![Time Series Analysis Chart](chart_3.png)

4. **Dynamic Analysis:** Lastly, I conducted a dynamic analysis that examined skewness, kurtosis, and correlation strictly among the numerical variables. Notable observations included a positive skewness in repeatability and strong correlations between overall ratings and quality.

## Insights Discovered

The multiple analyses unveiled several critical insights:

- **Quality's Impact:** Quality ratings were strongly correlated with overall ratings, demonstrating that improvements in media quality are likely to enhance viewer satisfaction significantly.
- **Repeatability's Role:** While repeatability showed moderate influence, its positive skewness suggests that most media content tends to have high repeatability, which may be perceived positively by viewers.
- **Stationarity of Ratings:** The stationary nature of overall ratings over time indicates that past trends can reliably forecast future ratings, providing a stable foundation for continuous improvement strategies.

## Implications of Findings

The insights gained from this comprehensive analysis can guide strategic decisions:

- **Focus on Quality:** Media producers should prioritize enhancing the quality of their content, as this is likely to resonate positively with audiences and drive higher overall ratings.
- **Monitor Repeatability:** Understanding the factors behind repeatability can help creators maintain viewer interest in reviewing or rewatching content, potentially leading to greater engagement.
- **Data-Driven Decisions:** The stationary nature of the overall ratings encourages a data-driven approach to forecasting audience reception for future media projects.

By leveraging these insights, media organizations can make informed decisions that align with viewer preferences, ultimately enhancing content relevance and satisfaction.

