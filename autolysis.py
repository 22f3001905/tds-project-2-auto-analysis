# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "scikit-learn",
#     "seaborn",
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

import matplotlib.pyplot as plt
import seaborn as sns


# Utility Functions
def load_env_key():
    load_dotenv()
    try:
        api_key = os.environ["AIPROXY_TOKEN"]
    except KeyError:
        print("Error: AIPROXY_TOKEN is not set in the environment.")
        sys.exit(1)
    return api_key


def get_dataset():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)
    return sys.argv[1]


def get_dataset_encoding(dataset_file):
    if not os.path.isfile(dataset_file):
        print(f"Error: File '{dataset_file}' not found.")
        sys.exit(1)
    
    with open(dataset_file, 'rb') as f:
        result = chardet.detect(f.read())
    
    print(f'File Info: {result}')
    return result.get('encoding', 'utf-8')


def write_file(file_name, text_content, title=None):
    with open(file_name, "a") as f:
        if title:
            f.write("# " + title + "\n\n")
        f.write(text_content)
        f.write('\n\n')


def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def name_chart_file():
    cwd = os.getcwd()
    png_files = [file for file in os.listdir(cwd) if file.endswith(".png")]
    count = len(png_files)

    return f'chart_{count + 1}.png'


# AI Proxy Functions
def text_embedding(query, api_key, model='text-embedding-3-small'):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "input": query,
        "model": model,
        "encoding_format": "float"
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()['data']


def chat(prompt, api_key, model='gpt-4o-mini'):
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
    response = requests.post(url, headers=headers, json=data)
    output = response.json()

    if output.get('error', None):
        print('LLM Error:\n', output)
        return None
    
    return output['choices'][0]['message']['content']


def image_info(base64_image, prompt, api_key, model='gpt-4o-mini'):
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

    response = requests.post(url, headers=headers, json=data)
    output = response.json()

    if output.get('error', None):
        print('LLM Error:\n', output)
        return None
    
    return output['choices'][0]['message']['content']


# AI Proxy 'Function Call' Functions
def chat_function_call(prompt, api_key, function_descriptions, model='gpt-4o-mini'):
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
    response = requests.post(url, headers=headers, json=data)    
    output = response.json()

    if output.get('error', None):
        print('LLM Error:\n', output)
        return None
    
    return {
        'arguments': output['choices'][0]['message']['function_call']['arguments'],
        'name': output['choices'][0]['message']['function_call']['name']
    }


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
        'description': 'Extract a feature matrix and a target vector for training a regression model. Call this when you need to get X and y for a Regression task.',
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
]


def filter_features(data, features):
    return data[features]


def extract_features_and_target(data, features, target):
    return data[features], data[target]


def extract_time_series_data(data, date_column, numerical_column):
    return data[[date_column, numerical_column]]


# Analysis Functions
def generic_analysis(data):
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
        print("\nNo numeric columns found. Cannot compute correlation matrix.")
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


def outlier_detection(dataset_file, data, api_key):
    df = data.select_dtypes(include=['number'])

    if df.empty:
        print("\nNo numeric columns found.")
        return None
    
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
    ])
    isolation_forest = IsolationForest(contamination='auto', random_state=42)

    df_tr = preprocessing.fit_transform(df)
    isolation_forest.fit(df_tr)

    anomaly_score = isolation_forest.predict(df_tr)
    return { 
        'n_anomalies': (anomaly_score == -1).sum(), 
        'n_samples': df_tr.shape[0]
    }


def regression_analysis(dataset_file, data, api_key):
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

    # print(response)
    if not response:
        return None

    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])
    
    if 'target' not in params.keys():
        return None
    
    # print('Regression Analysis')
    params['features'] = list(filter(lambda feature: feature != params['target'], params['features']))

    # print(params)

    X, y = chosen_func(data=data, **params)
    X = X.select_dtypes(include=['number'])

    if X.empty:
        print("\nNo numeric columns found.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('regression', LinearRegression())
    ])

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

    # print(response)
    if not response:
        return None
    
    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])

    print(params)
    
    df = chosen_func(data=data, **params)
    numeric_data = df.select_dtypes(include=['number'])

    if numeric_data.empty:
        print("\nNo numeric columns found. Cannot compute correlation matrix.")
        return None
    
    corr = numeric_data.corr()
    chart_name = plot_correlation(corr)

    return {
        'correlation_matrix': corr,
        'chart': chart_name
    }


def cluster_analysis(dataset_file, data, api_key):
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

    # print(response)
    if not response:
        return None
    
    params = json.loads(response['arguments'])
    chosen_func = eval(response['name'])

    print(params)
    
    df = chosen_func(data=data, **params)

    # TODO: (Optional) Use LLM to separate numerical cols from categorical cols.
    df = df.select_dtypes(include=['number'])

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=4, random_state=42))
    ])

    pipe.fit(df)

    return {
        'cluster_centers': pipe['kmeans'].cluster_centers_,
        'inertia': pipe['kmeans'].inertia_,
    }


# TODO: More analysis functions.
def classification_analysis(dataset_file, data, api_key):
    pass


def geographic_analysis(dataset_file, data, api_key):
    pass


def network_analysis(dataset_file, data, api_key):
    pass


def time_series_analysis(dataset_file, data, api_key):
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

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = adfuller(ts_data)

    # decompose_result = seasonal_decompose(ts_data, model='additive')
    # decompose_result.plot()
    # plt.show()

    chart_name = plot_time_series(ts_data, num_col)

    return {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05,
        'chart': chart_name
    }


# Plotting Functions
def plot_time_series(ts_data, num_col):
    dpi = 100
    plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)
    
    plt.plot(ts_data, label=num_col)
    plt.title(f"Time Series of {num_col}")
    plt.xlabel("Date")
    plt.ylabel(num_col)
    plt.legend()
    plt.show()

    chart_name = name_chart_file()
    
    plt.savefig(f"{chart_name}.png")
    plt.close()

    return chart_name


def plot_regression(y_true, y_pred):
    dpi = 100
    plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)

    plt.scatter(y_true, y_pred, alpha=0.8)
    plt.plot(y_true, y_true, 'r-', label='y = x')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png")
    plt.close()

    return chart_name


def plot_correlation(corr):
    dpi = 100
    plt.figure(figsize=(512 / dpi, 512 / dpi), dpi=dpi)

    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()

    chart_name = name_chart_file()

    plt.savefig(f"{chart_name}.png")
    plt.close()

    return chart_name


# Perform Analysis Functions
def choose_analysis(dataset_file, data, api_key, analyses):
    results = {}
    for analysis in analyses:
        func = eval(analysis)
        res = func(dataset_file, data, api_key)
        if res != None:
            results[analysis] = res
    
    return results


def meta_analysis(dataset_file, data, api_key):
    analyses = ['outlier_detection', 'regression_analysis', 'correlation_analysis', 
                'cluster_analysis', 'classification_analysis', 'geographic_analysis', 
                'network_analysis', 'time_series_analysis']

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
                        "description": "List of analysis to perform in order. Eg. ['Regression Analysis', 'Cluster Analysis']",
                    },
                },
                "required": ["indices"]
            }
        }
    ]
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    
    unorder_list_analyses = "\n".join([f'{i+1}. {analysis_name}' for (i, analysis_name) in enumerate(analyses)])
    
    prompt = f"""\
    You are given a file {dataset_file}.

    With features:
    {columns_info}

    Here are a few samples:
    {data.iloc[:3, :]}

    Perform only a few appropriate analyses. Make sure they are in correct order.

    Analysis options:
    {unorder_list_analyses}

    Call the choose_analysis function with the correct options.
    """

    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=analysis_function_descriptions)
    # print(response)

    params = json.loads(response['arguments'])
    choose_analysis_func = eval(response['name'])

    print(params)

    analysis_results = choose_analysis_func(dataset_file, data, api_key, **params)
    return analysis_results


# LLM Prompt: Description Functions
def describe_generic_analysis(results, dataset_file, data, api_key):
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

    print(prompt)
    response = chat(prompt=prompt, api_key=api_key)
    return response


# TODO: Add an plot for each analysis.
def describe_meta_analysis(results, dataset_file, data, api_key):
    responses = []
    for (func, res) in results.items():
        if res:
            prompt = f"""\
            Analysis Function: {func}

            Results:
            {res}

            The given analysis was performed on {dataset_file}.
            What are some of the findings of this analysis?

            * Write about the analysis that was performed.
            * Try to infer insights from the results of the analysis.
            * Provide a description about the insights you discovered.

            Output in valid markdown format.
            """

        print(prompt)
        response = chat(prompt=prompt, api_key=api_key)
        responses.append(response)
    
    return responses


def main():
    sns.set_theme('notebook')
    api_key = load_env_key()
    dataset_file = get_dataset()
    encoding = get_dataset_encoding(dataset_file)

    try:
        df = pd.read_csv(dataset_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # ANALYSIS
    # Describe the given dataset.
    generic_analysis_results = generic_analysis(data=df)
    generated_description = describe_generic_analysis(generic_analysis_results, dataset_file, df, api_key)
    write_file('README.md', generated_description)

    # Perform non-generic analysis.
    meta_analysis_results = meta_analysis(dataset_file, df, api_key)
    # print(meta_analysis_results)
    generated_meta_analysis_descriptions =  describe_meta_analysis(meta_analysis_results, dataset_file, df, api_key)

    for meta_analysis_description in generated_meta_analysis_descriptions:
        write_file('README.md', meta_analysis_description)

    # TODO: Describe the implications of your findings. (What to do with the insights?)


if __name__ == '__main__':
    main()
