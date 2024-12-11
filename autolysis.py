# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chardet",
#     "folium",
#     "geopy",
#     "matplotlib",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "scikit-learn",
#     "seaborn",
#     "selenium",
#     "statsmodels",
#     "tabulate",
# ]
# ///


# Project Goals:
# Write a Python script that uses an LLM to analyze, visualize, and narrate a story from a dataset.
# Convince an LLM that your script and output are of high quality.


import base64
import io
import json
import sys
import os
from dotenv import load_dotenv
import pandas as pd
import chardet
import requests
import logging

import matplotlib.pyplot as plt
import seaborn as sns


# Utility Functions
def load_env_key():
    """Load API key from environment variable."""
    load_dotenv()
    try:
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        logging.critical("Error: AIPROXY_TOKEN is not set in the environment.")
        sys.exit(1)
    return api_key


def get_dataset():
    """Retrieve dataset file path from command-line arguments."""
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    return sys.argv[1]


def get_dataset_encoding(dataset_file):
    """Determine the encoding of a dataset file."""
    if not os.path.isfile(dataset_file):
        logging.critical(f"Error: File '{dataset_file}' not found.")
        sys.exit(1)
    
    try:
        with open(dataset_file, 'rb') as file:
            result = chardet.detect(file.read())
    except Exception as e:
        logging.critical(f"Error reading file '{dataset_file}': {e}")
        sys.exit(1)
    
    logging.info(f'File Info: {result}')
    return result.get('encoding', 'utf-8')


def read_csv_file(dataset_file, encoding):
    """Read a CSV file and clean it by dropping empty rows and columns."""
    try:
        df = pd.read_csv(dataset_file, encoding=encoding)
    except Exception as e:
        logging.critical(f"Error reading the CSV file: {e}")
        sys.exit(1)
    
    return df.dropna(axis=1, how='all').dropna(axis=0, how='all')


def write_file(file_name, text_content, title=None):
    """Write text content to a file, optionally adding a title."""
    logging.info(f'Writing: {file_name}')

    try:
        with open(file_name, "a") as f:
            if title:
                f.write("# " + title + "\n\n")

            if text_content.startswith("```markdown"):
                text_content = text_content.replace("```markdown", "", 1).strip().rstrip("```").strip()

            f.write(text_content + "\n\n")
    except Exception as e:
        logging.error(f"Error writing to file '{file_name}': {e}")


def encode_image(image_path):
    """Encode an image to a Base64 string."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
    try:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Image file not found: {image_path}")
        return ""
    except IOError as e:
        logging.error(f"Error reading image file '{image_path}': {e}")
        return ""


def name_chart_file():
    """Generate a unique name for a chart file based on existing PNG files in the current directory."""
    try:
        cwd = os.getcwd()
        png_files = [file for file in os.listdir(cwd) if file.endswith(".png")]
        count = len(png_files)
        return f'chart_{count + 1}'
    except Exception as e:
        logging.error(f"Error generating chart file name: {e}")
        return "chart_unknown"


# AI Proxy Functions
def chat(prompt, api_key, model='gpt-4o-mini'):
    """Send a chat prompt to the AI API and return the response."""
    url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': model,
        'messages': [
            {
                'role': 'system',
                'content': 'You are a concise assistant and a data science expert. Provide brief and to-the-point answers.'
            },
            {
                'role': 'user',
                'content': prompt
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        output = response.json()

        if output.get('error'):
            logging.error(f"LLM Error: {output}")
            return None

        logging.info(f"Monthly Cost: {output.get('monthlyCost', None)}")
        return output['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        return None
    except KeyError as e:
        logging.error(f"Unexpected response format: {e}")
        return None


def image_info(base64_image, prompt, api_key, model='gpt-4o-mini'):
    """Send an image and prompt to the AI API for information extraction."""
    url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'text',
                        'text': prompt
                    },
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{base64_image}',
                            "detail": "low"
                        }
                    }
                ]
            }
        ]
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        output = response.json()

        if output.get('error'):
            logging.error(f"LLM Error: {output}")
            return None

        logging.info(f"Monthly Cost: {output.get('monthlyCost', None)}")
        return output['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        return None
    except KeyError as e:
        logging.error(f"Unexpected response format: {e}")
        return None


# AI Proxy 'Function Call' Functions
def chat_function_call(prompt, api_key, function_descriptions, model='gpt-4o-mini'):
    """Call an AI API for a function completion based on user input."""
    url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'functions': function_descriptions,
        'function_call': 'auto'
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        output = response.json()

        if output.get('error'):
            logging.error(f"LLM Error: {output}")
            return None
        
        logging.info(f"Monthly Cost: {output.get('monthlyCost', None)}")
        
        return {
            'arguments': output['choices'][0]['message']['function_call']['arguments'],
            'name': output['choices'][0]['message']['function_call']['name']
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"Unexpected response format: {e}")
        return None


filter_function_descriptions = [
    {
        'name': 'filter_features',
        'description': 'Generic function to extract data from a dataset.',
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of column names to keep. Eg. ['language', 'quality']",
                },
            },
            "required": ["features"]
        }
    },
    {
        'name': 'extract_features_and_target',
        'description': 'Extract a feature matrix and a target vector for training a regression model. Call this when you need to get X and y for a Regression or Classification task.',
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of column names to choose for the feature matrix. Eg. ['n_rooms', 'locality', 'latitude']",
                },
                "target": {
                    "type": "string",
                    "description": "The column name of the target. Eg. 'price'",
                },
            },
            "required": ["features", "target"]
        }
    },
    {
        'name': 'extract_time_series_data',
        'description': "Extract the time column and numerical column for a time series analysis. Eg. 'date' and 'price'",
        "parameters": {
            "type": "object",
            "properties": {
                "date_column": {
                    "type": "string",
                    "description": "The column name of the date column.",
                },
                "numerical_column": {
                    "type": "string",
                    "description": "The column name of the numerical column. Eg. 'price'",
                }
            },
            "required": ["date_column", "numerical_column"]
        }
    },
    {
        'name': 'extract_lat_lng_data',
        'description': "Extract the latitude and longitude columns from a dataset for a geospacial analysis.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {
                    "type": "string",
                    "description": "The column name that represents latitude. Eg. 'lat'",
                },
                "longitude": {
                    "type": "string",
                    "description": "The column name that represents longitude. Eg. 'lng'",
                }
            },
            "required": ["latitude", "longitude"]
        }
    },
]


def filter_features(data, features):
    """Filters the specified features from the input DataFrame and returns a copy of the filtered data."""
    return data[features].copy()


def extract_features_and_target(data, features, target):
    """Extracts the specified features and target column from the input DataFrame."""
    return data[features], data[target].copy()


def extract_time_series_data(data, date_column, numerical_column):
    """Extracts the date and numerical columns from the input DataFrame for time series analysis."""
    return data[[date_column, numerical_column]].copy()


def extract_lat_lng_data(data, latitude, longitude):
    """Extracts the latitude and longitude columns from the input DataFrame."""
    return data[[latitude, longitude]].copy()


# Analysis Functions
def generic_analysis(data):
    """Performs generic analysis on the provided DataFrame."""
    logging.info('Working: Generic Analysis')

    results = {
        'first_5': data.head(),
        'summary_stats': data.describe(),
        'missing_values': data.isnull().sum(),
        'column_data_types': data.dtypes,
        'n_unique': data.nunique(),
        'n_duplicates': data.duplicated().sum()
    }

    buffer = io.StringIO()
    data.info(buf=buffer)
    results['basic_info'] = buffer.getvalue()
    buffer.close()

    # Compute correlation matrix for numeric columns only
    numeric_data = data.select_dtypes(include=['number'])
    if numeric_data.empty:
        logging.warning("No numeric columns found.")
        results['corr'] = None
    else:
        results['corr'] = numeric_data.corr()

    return results


# Non-generic Analysis Functions
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

def outlier_detection(dataset_file, data, api_key):
    """Perform outlier detection analysis."""
    df = data.select_dtypes(include=['number'])

    if df.empty:
        logging.warning("No numeric columns found.")
        return None
    
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    isolation_forest = IsolationForest(contamination='auto', random_state=42)

    logging.info('Running: Outlier Detection')
    df_tr = preprocessing.fit_transform(df)
    isolation_forest.fit(df_tr)

    anomaly_score = isolation_forest.predict(df_tr)

    return { 
        'n_anomalies': (anomaly_score == -1).sum(), 
        'n_samples': df_tr.shape[0]
    }


def regression_analysis(dataset_file, data, api_key):
    """Perform regression analysis."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here is a sample:
    {data.iloc[0, :]}

    Extract the features and target for Regression task.
    Note: Do not include column names that include the word 'id'.

    Make sure to NOT include the target variable in the features list.
    """
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)

    if not response:
        return None

    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])
    
    if 'target' not in params.keys():
        return None
    
    params['features'] = list(filter(lambda feature: feature != params['target'], params['features']))

    X, y = chosen_func(data=data, **params)
    X = X.select_dtypes(include=['number'])

    if X.empty:
        logging.warning("No numeric columns found.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])

    logging.info('Working: Linear Regression')

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2_score = pipe.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    chart_name = plot_regression(y_test, y_pred)

    return {
        'r2_score': r2_score,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'coefficient': pipe['regression'].coef_,
        'intercept': pipe['regression'].intercept_,
        'feature_names_input': list(X.columns),
        'target_name': y.name,
        'chart': chart_name
    }


def correlation_analysis(dataset_file, data, api_key):
    """Perform correlation analysis."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here is a sample:
    {data.iloc[0, :]}

    Extract only the most important features to perform a Correlation analysis.
    Eg. 'height', 'weight', etc.
    
    Note: Do not include column names that include the word 'id'.
    Hint: Use function filter_features.
    """
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)

    if not response:
        return None
    
    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])

    df = chosen_func(data=data, **params)
    numeric_data = df.select_dtypes(include=['number'])

    if numeric_data.empty:
        logging.warning("No numeric columns found.")
        return None
    
    corr = numeric_data.corr()
    chart_name = plot_correlation(corr)

    return {
        'correlation_matrix': corr,
        'chart': chart_name
    }


def cluster_analysis(dataset_file, data, api_key):
    """Perform clustering analysis."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here is a sample:
    {data.iloc[0, :]}

    Extract only the most important features to perform a Clustering analysis using K-Means.
    
    Note: Do not include column names that include the word 'id'. 
    Hint: Use function filter_features.
    """
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)

    if not response:
        return None
    
    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])

    df = chosen_func(data=data, **params)

    # TODO: (Optional) Use LLM to separate numerical cols from categorical cols.
    df = df.select_dtypes(include=['number'])

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=4, random_state=42))
    ])

    logging.info('Working: K-Means Clustering')
    pipe.fit(df)

    return {
        'cluster_centers': pipe['kmeans'].cluster_centers_,
        'inertia': pipe['kmeans'].inertia_,
    }


def classification_analysis(dataset_file, data, api_key):
    """Perform classification analysis."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here is a sample:
    {data.iloc[0, :]}

    Extract the features and target for Classification task.
    Note: Do not include column names that include the word 'id'.

    Make sure to NOT include the target variable in the features list.
    Note: The target column should be categorical datatype.
    """

    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)

    if not response:
        return None

    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])
    
    if 'target' not in params.keys():
        return None
    
    params['features'] = list(filter(lambda feature: feature != params['target'], params['features']))

    X, y = chosen_func(data=data, **params)
    X = X.select_dtypes(include=['number'])

    if y.nunique() > 20:
        logging.warning('y is not a valid target label.')
        return None

    if X.empty:
        logging.warning("No numeric columns found.")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('log_clf', LogisticRegression(random_state=42))
    ])

    logging.info('Working: Logistic Regression')

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    con_mat = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    chart_name = plot_classification(y_test, y_pred)

    return {
        'confusion_matrix': con_mat,
        'classification_report': report,
        'chart': chart_name
    }


import folium
from geopy.distance import geodesic

def geospatial_analysis(dataset_file, data, api_key):
    """Perform geospatial analysis."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here is a sample:
    {data.iloc[0, :]}

    Extract the latitude column and the longitude column.

    Note: Do not include column names that include the word 'id'.
    Hint: Use function extract_lat_lng_data.
    """

    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)

    if not response:
        return None

    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])

    df = chosen_func(data=data, **params)
    # Column Names: Eg. 'latitude' or 'lat'
    lat = params['latitude']
    lng = params['longitude']

    city_center = (df[lat].mean(), df[lng].mean())
    chart_name = plot_map(df, city_center, lat, lng)

    df['distance'] = df.apply(
        lambda row: geodesic((row[lat], row[lng]), city_center).km,
        axis=1)
    
    return {
        'avg_distance_from_city_center_km': df['distance'].mean(),
        'std_distance_from_city_center_km': df['distance'].std(),
        'chart': chart_name
    }



def time_series_analysis(dataset_file, data, api_key):
    """Perform time series analysis."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here is a sample:
    {data.iloc[0, :]}

    Extract the date column and the numerical column.

    Note: Do not include column names that include the word 'id'.
    Hint: Use function extract_time_series_data.
    """
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_descriptions)

    if not response:
        return None
    
    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])

    df = chosen_func(data=data, **params)
    date_col = params['date_column']
    num_col = params['numerical_column']

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    ts_data = df[num_col]
    chart_name = plot_time_series(ts_data, num_col)

    from statsmodels.tsa.stattools import adfuller

    logging.info('Working: Time-Series Analysis')
    result = adfuller(ts_data)

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05,
        'chart': chart_name
    }


# Plotting Functions
def plot_time_series(ts_data, num_col):
    """Plot time series chart."""
    dpi = 100
    plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)
    
    plt.plot(ts_data, label=num_col)
    plt.title(f"Time Series of {num_col}")
    plt.xlabel("Date")
    plt.ylabel(num_col)
    plt.legend()

    chart_name = name_chart_file()
    
    plt.savefig(f"{chart_name}.png")
    plt.close()

    return f"{chart_name}.png"


def plot_regression(y_true, y_pred):
    """Plot regression chart."""
    dpi = 100
    plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)

    plt.scatter(y_true, y_pred, alpha=0.8)
    plt.plot(y_true, y_true, 'r-', label='y = x')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png")
    plt.close()

    return f"{chart_name}.png"


from sklearn.metrics import ConfusionMatrixDisplay

def plot_classification(y_true, y_pred):
    """Plot classification chart."""
    dpi = 100
    plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.grid(False)

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png")
    plt.close()

    return f"{chart_name}.png"


def plot_correlation(corr):
    """Plot correlation chart."""
    dpi = 100
    n_cols = len(corr.columns)
    plt.figure(figsize=(n_cols, n_cols), dpi=dpi)
    # plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)

    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title('Correlation Heatmap')

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png", dpi=dpi)
    plt.close()

    return f"{chart_name}.png"


from PIL import Image
import selenium

def plot_map(df, city_center, lat, lng):
    """Plot geospatial chart."""
    map = folium.Map(location=city_center, zoom_start=12)

    for i, row in df.iterrows():
        coords = (row[lat], row[lng])
        folium.Marker(location=coords).add_to(map)
    
    chart_name = name_chart_file()

    img_data = map._to_png(5)
    img = Image.open(io.BytesIO(img_data))
    img.save(f"{chart_name}.png")

    return f"{chart_name}.png"


# Perform Analysis Functions
def choose_analysis(dataset_file, data, api_key, analyses):
    """Perform all the provided analysis functions."""
    results = {}
    for analysis in analyses:
        func = eval(analysis)
        
        try:
            res = func(dataset_file, data, api_key)
        except:
            logging.error(f'Error in {analysis} function.')
            res = None

        if res != None:
            results[analysis] = res
    
    return results


def meta_analysis(dataset_file, data, api_key):
    """Prompt the LLM to choose relevant functions for performing various analyses on the dataset.

    Args:
        dataset_file (str): Path to the dataset file.
        data (pandas.DataFrame): The dataset for which analysis is to be performed.
        api_key (str): API key for making requests to external services.

    Returns:
        dict: Results from the chosen analyses as specified by the LLM.
    """

    analyses = ['outlier_detection', 'regression_analysis', 'correlation_analysis', 'cluster_analysis', 
                'classification_analysis', 'geospatial_analysis', 'time_series_analysis']

    analysis_function_descriptions = [
        {
            'name': 'choose_analysis',
            'description': 'A function to choose all the relevant analysis to be performed for a dataset.',
            "parameters": {
                "type": "object",
                "properties": {
                    "analyses": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of analysis to perform in order. Eg. ['regression_analysis', 'time_series_analysis', 'outlier_detection']",
                    },
                },
                "required": ["indices"]
            }
        }
    ]
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    
    order_list_analyses = "\n".join([f'{i+1}. {analysis_name}' for (i, analysis_name) in enumerate(analyses)])
    
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here are a few samples:
    {data.iloc[:3, :]}

    Note: Perform only the appropriate analyses.

    Analysis options:
    {order_list_analyses}

    Call the choose_analysis function with the correct options.
    """

    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=analysis_function_descriptions)

    params = json.loads(response['arguments'])
    choose_analysis_func = eval(response['name'])

    analysis_results = choose_analysis_func(dataset_file, data, api_key, **params)
    return analysis_results


# LLM Prompt: Description Functions
def describe_generic_analysis(results, dataset_file, data, api_key):
    """Prompt LLM to describe the generic analysis performed on the dataset."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])

    prompt = f"""\
    You are given a file: {dataset_file}

    Features:
    {columns_info}

    Here are a few samples:
    {results['first_5']}

    Summary Statistics:
    {results['summary_stats']}

    Missing Values:
    {results['missing_values']}

    Number of unique values in each column:
    {results['n_unique']}

    Number of duplicated rows:
    {results['n_duplicates']}

    Correlation Analysis:
    {results['corr']}

    Give a short description about the given dataset. Also provide a brief description of the given statistical analysis. 
    
    Output in valid markdown format.
    """

    response = chat(prompt=prompt, api_key=api_key)
    return response


def describe_meta_analysis(results, dataset_file, data, api_key):
    """Prompt LLM to describe all the analyses performed on the dataset."""
    responses = []
    for (func, res) in results.items():
        if not res:
            continue
        
        prompt = f"""\
        Analysis Function: {func}

        Results:
        {res}

        The given analysis was performed on {dataset_file}.
        What are some of the findings of this analysis?

        * Write about the analysis that was performed.
        * Try to infer insights from the results of the analysis.
        * Provide a description about the insights you discovered.
        * Give the implications of your findings.

        Output in valid markdown format.
        """
    
        img_path = res.get('chart', None)
        if img_path:
            chart_base64 = encode_image(img_path)
            img_analysis_prompt = """\
            Here is a chart to visualize some information in a .csv file. 
            Analyze and infer insights from the chart, then summarize the findings.
            """

            image_analysis = image_info(chart_base64, prompt=img_analysis_prompt, api_key=api_key)

            prompt += f"""
            Additional: Add the chart image which is a .png file with its analysis to the markdown output.

            Chart Analysis Summary:
            {image_analysis}

            Output in valid markdown format.
            """

        response = chat(prompt=prompt, api_key=api_key)
        responses.append(response)
    
    return responses


def main():
    # SETUP
    sns.set_theme('notebook')
    logging.basicConfig(level=logging.INFO)

    api_key = load_env_key()
    dataset_file = get_dataset()
    encoding = get_dataset_encoding(dataset_file)
    df = read_csv_file(dataset_file, encoding)

    # ANALYSIS
    # Describe the given dataset.
    try:
        generic_analysis_results = generic_analysis(data=df)
        generated_description = describe_generic_analysis(generic_analysis_results, dataset_file, df, api_key)
        write_file('README.md', generated_description)
    
        # Consult LLM and perform analysis.
        meta_analysis_results = meta_analysis(dataset_file, df, api_key)
        generated_meta_analysis_descriptions = describe_meta_analysis(meta_analysis_results, dataset_file, df, api_key)

        for meta_analysis_description in generated_meta_analysis_descriptions:
            write_file('README.md', meta_analysis_description)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {'error': str(e)}


if __name__ == '__main__':
    main()
