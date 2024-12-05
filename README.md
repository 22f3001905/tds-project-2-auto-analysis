# Dataset Description

The `media.csv` dataset contains information about various media items, specifically films, including their release date, language, type, title, cast, and ratings. It consists of:
- **Features**: 
  - `date`: Release date of the media item (object).
  - `language`: Language of the media (object).
  - `type`: Type of media (e.g., movie) (object).
  - `title`: Title of the media (object).
  - `by`: Cast of the media (object).
  - `overall`: Overall rating (integer).
  - `quality`: Quality rating (integer).
  - `repeatability`: Repeatability rating (integer).

# Statistical Analysis Summary

## Summary Statistics
- **Count**: Total entries are 2652 across ratings.
- **Mean Ratings**:
  - `overall`: 3.05
  - `quality`: 3.21
  - `repeatability`: 1.49
- **Standard Deviation**: 
  - `overall`: 0.76
  - `quality`: 0.80
  - `repeatability`: 0.60
- **Rating Ranges**: 
  - Ratings span from 1 to 5 for `overall` and `quality`. 
  - `repeatability` ranges from 1 to 3.

## Missing Values
- **Missing Entries**:
  - `date`: 99 missing values.
  - `by`: 262 missing values.
- Other features have no missing values.

## Unique Value Counts
- **Date**: 2055 unique release dates.
- **Languages**: 11 unique languages.
- **Types**: 8 unique types of media.
- **Titles**: 2312 unique titles.
- **Contributors**: 1528 unique casts.

## Duplicates
- There is **1 duplicated row** in the dataset.

## Correlation Analysis
- **Correlation Coefficients**:
  - `overall` to `quality`: Strong positive correlation (0.83).
  - `overall` to `repeatability`: Moderate positive correlation (0.51).
  - `quality` to `repeatability`: Weak-to-moderate positive correlation (0.31).
  
Overall, the dataset captures a variety of media ratings with certain strong relationships, particularly between overall ratings and quality. The presence of missing values, particularly in the `date` and `by` fields, may require further attention during analysis.