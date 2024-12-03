## Dataset Analysis

### First 5 Rows

| date      | language   | type   | title       | by                            |   overall |   quality |   repeatability |
|:----------|:-----------|:-------|:------------|:------------------------------|----------:|----------:|----------------:|
| 15-Nov-24 | Tamil      | movie  | Meiyazhagan | Arvind Swamy, Karthi          |         4 |         5 |               1 |
| 10-Nov-24 | Tamil      | movie  | Vettaiyan   | Rajnikanth, Fahad Fazil       |         2 |         2 |               1 |
| 09-Nov-24 | Tamil      | movie  | Amaran      | Siva Karthikeyan, Sai Pallavi |         4 |         4 |               1 |
| 11-Oct-24 | Telugu     | movie  | Kushi       | Vijay Devarakonda, Samantha   |         3 |         3 |               1 |
| 05-Oct-24 | Tamil      | movie  | GOAT        | Vijay                         |         3 |         3 |               1 |

### Dataset Info

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2652 entries, 0 to 2651
Data columns (total 8 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   date           2553 non-null   object
 1   language       2652 non-null   object
 2   type           2652 non-null   object
 3   title          2652 non-null   object
 4   by             2390 non-null   object
 5   overall        2652 non-null   int64 
 6   quality        2652 non-null   int64 
 7   repeatability  2652 non-null   int64 
dtypes: int64(3), object(5)
memory usage: 165.9+ KB

```

### Summary Statistics

|       |    overall |     quality |   repeatability |
|:------|-----------:|------------:|----------------:|
| count | 2652       | 2652        |     2652        |
| mean  |    3.04751 |    3.20928  |        1.49472  |
| std   |    0.76218 |    0.796743 |        0.598289 |
| min   |    1       |    1        |        1        |
| 25%   |    3       |    3        |        1        |
| 50%   |    3       |    3        |        1        |
| 75%   |    3       |    4        |        2        |
| max   |    5       |    5        |        3        |

### Missing Values

|               |   0 |
|:--------------|----:|
| date          |  99 |
| language      |   0 |
| type          |   0 |
| title         |   0 |
| by            | 262 |
| overall       |   0 |
| quality       |   0 |
| repeatability |   0 |

### Column Data Types

|               | 0      |
|:--------------|:-------|
| date          | object |
| language      | object |
| type          | object |
| title         | object |
| by            | object |
| overall       | int64  |
| quality       | int64  |
| repeatability | int64  |

### Unique Values in Each Column

|               |    0 |
|:--------------|-----:|
| date          | 2055 |
| language      |   11 |
| type          |    8 |
| title         | 2312 |
| by            | 1528 |
| overall       |    5 |
| quality       |    5 |
| repeatability |    3 |

### Correlation Matrix

|               |   overall |   quality |   repeatability |
|:--------------|----------:|----------:|----------------:|
| overall       |  1        |  0.825935 |        0.5126   |
| quality       |  0.825935 |  1        |        0.312127 |
| repeatability |  0.5126   |  0.312127 |        1        |

### Duplicated Rows

Number of duplicated rows: 1

