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

# NOTES:
# * The eval() does not pose security risks as the LLM only responses with valid function names that is defined in this script.
# * The theme for all the data visualizations is consistent using the sns.set_theme() method.
# * The scipt is dynamic in nature as it prompts an LLM to perform the appropriate analysis to be peformed on the given dataset.
# * The analysis functions created do not call the LLM unless they are run, improviing efficiency.

import base64
import io
import json
import logging
import os
import sys

from dotenv import load_dotenv

import chardet
import numpy as np
import pandas as pd
import requests

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import adfuller

import folium
from geopy.distance import geodesic

from PIL import Image
import selenium


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


def name_chart_file(prefix=''):
    """Generate a unique name for a chart file based on existing PNG files in the current directory."""
    try:
        cwd = os.getcwd()
        png_files = [file for file in os.listdir(cwd) if file.endswith(".png")]
        count = len(png_files)
        return f'{prefix}chart_{count + 1}'
    except Exception as e:
        logging.error(f"Error generating chart file name: {e}")
        return "chart_unknown"


def handle_function_call(function_call, conversation_history, data):
    """Handles a function call based on the provided name and arguments."""
    function_name = function_call.get('name')
    try:
        args = json.loads(function_call.get('arguments', '{}'))
    except json.JSONDecodeError as e:
        raise ValueError("Invalid arguments format: LLM Function Call.") from e

    functions = {
        'generic_analysis': generic_analysis,
        'regression_analysis': regression_analysis,
        'classification_analysis': classification_analysis,
        'geospatial_analysis': geospatial_analysis, 
        'time_series_analysis': time_series_analysis,
    }

    func = functions.get(function_name)
    if not func:
        raise ValueError(f"Unknown function: {function_name}")
    
    function_result = func(data, **args)
    
    # TODO: DataFrame is not JSON serializable.
    conversation_history.append({
        "role": "function",
        "name": function_name,
        "content": str(function_result),
    })

    return function_result


# AI Proxy Functions
def chat(prompt, api_key, conversation_history, function_descriptions, model='gpt-4o-mini', base64_image=None):
    """Send a chat prompt to the LLM and return the response."""

    user_message = [{
        'role': 'user',
        'content': prompt
    }]

    if base64_image:
        content = [
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
        
        user_message[0]['content'] = content

    conversation_history += user_message

    url = 'https://aiproxy.sanand.workers.dev/openai/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': model,
        'messages': conversation_history,
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

        # Log LLM Costs
        monthly_cost = output.get('monthlyCost')
        if monthly_cost is not None:
            logging.info(f"Monthly Cost: {monthly_cost:.4f}")
            logging.info(f"Percentage Used: {(monthly_cost / 2) * 100:.4f}%")
        
        assistant_message = output['choices'][0]['message']
        conversation_history.append(assistant_message)

        assistant_content = assistant_message.get('content')
        function_call = assistant_message.get('function_call')

        return {
            'content': assistant_content,
            'function_call': function_call
        }
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Request failed: {e}")
        return None
    except KeyError as e:
        logging.error(f"Unexpected response format: {e}")
        return None


# Static Analysis Functions
def regression_analysis(data, target):
    """Perform regression analysis."""

    X = data.drop(target, axis=1)
    y = data[target]

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

def classification_analysis(data, label):
    """Perform classification analysis."""

    X = data.drop(label, axis=1)
    y = data[label]

    X = X.select_dtypes(include=['number'])
    if y.nunique() > 10:
        logging.warning('y is not a valid label.')
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

def geospatial_analysis(data, latitude, longitude):
    df = data[[latitude, longitude]].copy()

    city_center = (df[latitude].mean(), df[longitude].mean())
    chart_name = plot_map(df, city_center, latitude, longitude)

    df['distance'] = df.apply(
        lambda row: geodesic((row[latitude], row[longitude]), city_center).km,
        axis=1)
    
    return {
        'avg_distance_from_city_center_km': df['distance'].mean(),
        'std_distance_from_city_center_km': df['distance'].std(),
        'chart': chart_name
    }

def time_series_analysis(data, date_column, numerical_column):
    """Perform time series analysis."""

    df = data[[date_column, numerical_column]].copy()
    df = df.dropna(axis=0)

    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).sort_index()

    ts_data = df[numerical_column]
    chart_name = plot_time_series(ts_data, numerical_column)

    logging.info('Working: Time-Series Analysis')
    result = adfuller(ts_data)

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05,
        'chart': chart_name
    }


# Generic Analysis Functions
def outlier_detection(data):
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

def correlation_analysis(data):
    """Perform correlation analysis."""
    df = data.select_dtypes(include=['number'])

    if df.empty:
        logging.warning("No numeric columns found.")
        return None
    
    corr = df.corr()
    chart_name = plot_correlation(corr)

    return {
        'correlation_matrix': corr,
        'chart': chart_name
    }

def cluster_analysis(data):
    """Perform clustering analysis."""
    df = data.select_dtypes(include=['number'])

    if df.empty:
        logging.warning("No numeric columns found.")
        return None

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

def generic_analysis(data):
    """Performs generic analysis on the provided DataFrame."""

    # generic = ['outlier_detection', 'correlation_analysis', 'cluster_analysis']
    logging.info('Working: Generic Analysis')

    results = {
        'column_names': list(data.columns),
        'first_3': data.head(3),
        'summary_stats': data.describe(),
        'missing_values': data.isnull().sum(),
        'column_data_types': data.dtypes,
        'n_unique': data.nunique(),
        'n_duplicates': data.duplicated().sum(),
        'outlier_detection': outlier_detection(data),
        'correlation_analysis': correlation_analysis(data),
        'cluster_analysis': cluster_analysis(data),
    }

    buffer = io.StringIO()
    data.info(buf=buffer)
    results['column_info'] = buffer.getvalue()
    buffer.close()

    return results


# Plotting Functions
def plot_time_series(ts_data, num_col):
    """Plot time series chart."""
    sns.set_theme('notebook')
    dpi = 100

    time_periods = ['D', 'W', 'ME', 'QE', 'YE']
    selected_period = 'D'

    for period in time_periods:
        resampled_data = ts_data.resample(period).mean()
        if resampled_data.shape[0] >= 20:
            selected_period = period

    resampled_data = ts_data.resample(selected_period).mean()

    plt.figure(figsize=(20, 6), dpi=dpi)
    
    plt.plot(resampled_data, label=num_col)
    plt.title(f"Time Series of {num_col}")
    plt.xlabel(f"Time Period (in {selected_period})")
    plt.ylabel(f"Mean {num_col}")
    plt.legend()

    chart_name = name_chart_file()
    
    plt.savefig(f"{chart_name}.png", dpi=dpi)
    plt.close()

    return f"{chart_name}.png"

def plot_regression(y_true, y_pred):
    """Plot regression chart."""
    sns.set_theme('notebook')
    dpi = 100
    plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)

    plt.scatter(y_true, y_pred, alpha=0.8)
    plt.plot(y_true, y_true, 'r-', label='y = x')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png", dpi=dpi)
    plt.close()

    return f"{chart_name}.png"

def plot_classification(y_true, y_pred):
    """Plot classification chart."""
    sns.set_theme('notebook')
    dpi = 100
    n_cols = len(np.unique(y_true)) + 3
    plt.figure(figsize=(n_cols, n_cols), dpi=dpi)

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.grid(False)

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png", dpi=dpi)
    plt.close()

    return f"{chart_name}.png"

def plot_correlation(corr):
    """Plot correlation chart."""
    sns.set_theme('notebook')
    dpi = 100
    n_cols = len(corr.columns) + 3
    plt.figure(figsize=(n_cols, n_cols), dpi=dpi)

    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title('Correlation Heatmap')

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png", dpi=dpi)
    plt.close()

    return f"{chart_name}.png"

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


def dynamic_analysis():
    pass

# LLM Prompt: Description Functions
def describe_generic_analysis(results, dataset_file, data, api_key):
    """Prompt LLM to describe the generic analysis performed on the dataset."""
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])

    prompt = f"""\
    You are given a file: {dataset_file}

    Features:
    {columns_info}

    Here are a few samples:
    {results['first_3']}

    Summary Statistics:
    {results['summary_stats']}

    Missing Values:
    {results['missing_values']}

    Number of unique values in each column:
    {results['n_unique']}

    Number of duplicated rows:
    {results['n_duplicates']}

    Correlation Analysis:
    {results['correlation_matrix']}

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

            # image_analysis = image_info(chart_base64, prompt=img_analysis_prompt, api_key=api_key)

            prompt += f"""
            Additional: Add the chart image which is a .png file with its analysis to the markdown output.

            Output in valid markdown format.
            """

        response = chat(prompt=prompt, api_key=api_key)
        responses.append(response)
    
    return responses


def describe_dynamic_analysis():
    pass


def log_conversation_history(conversation_history):
    write_file('conversation_history.log', conversation_history, title='Conversation History')

def log_results(result):
    write_file('results.log', result)

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
    conversation_history = [
        {
            'role': 'system',
            'content': 'You are a concise assistant and a data science expert. Provide brief and to-the-point answers.'
        }
    ]

    function_descriptions = [
        {
            'name': 'generic_analysis',
            'description': 'Performs a generic analysis on a dataset. This includes analyzing number of unique data points, summary statistics, column data types etc.',
        },
        {
            'name': 'regression_analysis',
            'description': 'Performs a regression analysis on the DataFrame.',
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The column name of the target. Eg. 'price'",
                    },
                },
                "required": ["target"]
            }
        },
        {
            'name': 'classification_analysis',
            'description': 'Performs a classification analysis on the DataFrame.',
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "The column name of the label. Eg. 'has_disease'",
                    },
                },
                "required": ["label"]
            }
        },
        {
            'name': 'time_series_analysis',
            'description': "Performs a time series analysis on the DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {
                        "type": "string",
                        "description": "The column name of the date column. Eg. 'date' or 'year' etc.",
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
            'name': 'geospatial_analysis',
            'description': "Performs a geospacial analysis on the DataFrame.",
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
    
    try:
        params = {
            'api_key': api_key, 
            'conversation_history': conversation_history, 
            'function_descriptions': function_descriptions
        }

        # TODO: Remove this before deployment.
        print(chat('List all the functions that you can call?', **params)['content'])
        print('---'*10)

        generic_analysis_prompt = f'You are given a file: {dataset_file}. I have create a local DataFrame object, perform a generic analysis using a function call.'
        
        generic_analysis_function_call = chat(generic_analysis_prompt, **params)['function_call']
        generic_analysis_results = handle_function_call(generic_analysis_function_call, conversation_history, data=df)
        print('---'*10)

        print(chat('Write a short description about the dataset and the generic analysis result.', **params))
        print('---'*10)

        print(chat('This is the chart for the covariance matrix.', **params, base64_image=encode_image(generic_analysis_results['correlation_analysis']['chart'])))
        print('---'*10)

        static_analysis_functions = [
            'regression_analysis', 
            'classification_analysis', 
            'geospatial_analysis', 
            'time_series_analysis'
        ]

        order_list_analyses = "\n".join([f'{i+1}. {analysis_name}' for (i, analysis_name) in enumerate(static_analysis_functions)])

        static_analysis_prompt = f"""\
        Call the most appropriate analysis function from the given list below with the correct parameter.
        
        You have to figure out the correct parameters for the function call.

        Analysis functions:
        {order_list_analyses}

        Note: If an analysis from the list is already performed don't repeat it, select a different one.
        """

        static_analysis_results = {}

        for _ in range(2):
            static_analysis_function_call = chat(static_analysis_prompt, **params)['function_call']

            if not static_analysis_function_call:
                print('Function call was not called!')
                break

            analysis_func_name = static_analysis_function_call.get('name')
            
            static_analysis_results[analysis_func_name] = handle_function_call(static_analysis_function_call, conversation_history, data=df)

            print(chat(f'This is the chart for visualizing {analysis_func_name}. Use this and the analysis results computed earlier to write a short description.', **params, base64_image=encode_image(static_analysis_results[analysis_func_name]['chart'])))

            print('---'*10)

        # TODO: Dynamic Analysis: Use tenacity library to retry results.
        
        # TODO: Narrate a story.
        narrative_prompt = f"""\
        Write a story about the analysis. You are already aware of the dataset structure, the generic analysis results, static analysis results, and charts generated for visualizations.

        Describe the following:
        * The dataset you received, briefly
        * The analysis you carried out
        * The insights you discovered
        * The implications of your findings (i.e. what to do with the insights)

        Note: Output the story in valid Markdown.
        Note: Use the things learned in the context window.
        Note: Add charts in the correct locations in the markdown.
        """

        story = chat(narrative_prompt, **params)['content']
        write_file('README.md', story)

        log_conversation_history(conversation_history)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        log_conversation_history(conversation_history)
        return {'error': str(e)}
    
    print('\n\nResults\n')
    log_results(str(generic_analysis_results))
    log_results(str(static_analysis_results))


if __name__ == '__main__':
    main()
