## Dataset Description

The dataset `media.csv` contains information about various media titles (primarily movies) characterized by several features:

- **`date`**: The release date of the media, recorded as an object.
- **`language`**: Language of the media (e.g., Tamil, Telugu).
- **`type`**: Type of media, predominantly movies.
- **`title`**: Title of the media.
- **`by`**: Creators or cast of the title.
- **`overall`**: General rating given to the media (integer scale).
- **`quality`**: Quality rating (integer scale).
- **`repeatability`**: Frequency of being watched or the likelihood of rewatching (integer scale).

## Statistical Analysis

### Summary Statistics
- **Count**: 2652 entries in total for ratings.
- **Mean Ratings**: 
  - Overall average rating: 3.05
  - Average quality rating: 3.21
  - Average repeatability: 1.49
- **Standard Deviation**: Indicates variability:
  - Overall: 0.76
  - Quality: 0.80
  - Repeatability: 0.60
- **Rating Ranges**: 
  - Overall and quality ratings range from 1 to 5.
  - Repeatability values range from 1 to 3.
- **Percentiles**:
  - 25% of overall ratings are ≥ 3.
  - 50% (median) ratings are 3 for both overall and quality.
  - 75% quality ratings are 4.

### Missing Values
- **`date`**: 99 missing values.
- **`by`**: 262 missing values.
- Other columns have no missing values.

### Unique Values
- The dataset features a diverse range of unique values:
  - **`date`**: 2055 unique dates.
  - **`language`**: 11 languages.
  - **`type`**: 8 different types.
  - **`title`**: 2312 unique titles.
  - **`by`**: 1528 contributors.
  
### Duplicates
- There is 1 duplicated row in the dataset.

### Correlation Analysis
- **Overall and Quality Ratings**: Strong positive correlation (0.83).
- **Overall and Repeatability**: Moderate positive correlation (0.51).
- **Quality and Repeatability**: Weaker correlation (0.31).

This analysis identifies potential relationships between ratings, indicating that as overall ratings increase, quality ratings also tend to be higher, which may inform further study into media performance based on these features.
