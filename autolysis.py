# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chardet",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "scikit-learn",
#     "tabulate",
# ]
# ///


# Project Goals:
# Write a Python script that uses an LLM to analyze, visualize, and narrate a story from a dataset.
# Convince an LLM that your script and output are of high quality.


import io
import json
import sys
import os
from dotenv import load_dotenv
import pandas as pd
import chardet
import requests

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


def write_file(file_name, text_content, title=None):
    with open(file_name, "a") as f:
        if title:
            f.write("# " + title + "\n\n")
        f.write(text_content)
        f.write('\n\n')


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
    return response.json()

import requests

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

    res = requests.post(url, headers=headers, json=data)
    return res.json()['choices'][0]['message']['content']


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
    
    return output


filter_function_description = [
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
    }
]

def filter_features(data, features):
    return data[features]

def extract_features_and_target(data, features, target):
    return data[features], data[target]


# Non-generic Analysis
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


# TODO: Add classification_analysis

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
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_description)

    # print(response)
    if not response:
        return None

    params = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    chosen_func = eval(response['choices'][0]['message']['function_call']['name'])
    
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

    return {
        'r2_score': r2_score,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'coefficient': pipe['regression'].coef_,
        'intercept': pipe['regression'].intercept_,
        'feature_names_input': list(X.columns),
        'target_name': y.name
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
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_description)

    # print(response)
    if not response:
        return None
    
    params = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    chosen_func = eval(response['choices'][0]['message']['function_call']['name'])

    print(params)
    
    df = chosen_func(data=data, **params)
    numeric_data = df.select_dtypes(include=['number'])

    if numeric_data.empty:
        print("\nNo numeric columns found. Cannot compute correlation matrix.")
        return None
    else:
        return { 'correlation_matrix': numeric_data.corr() }


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
    
    response = chat_function_call(prompt=prompt, api_key=api_key, function_descriptions=filter_function_description)

    # print(response)
    if not response:
        return None
    
    params = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    chosen_func = eval(response['choices'][0]['message']['function_call']['name'])

    print(params)
    
    df = chosen_func(data=data, **params)

    # TODO: Use LLM to separate numerical cols from categorical cols.
    df = df.select_dtypes(include=['number'])

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=4, random_state=42))
    ])

    pipe.fit(df)

    # return {
    #     'cluster_labels': pipe['kmeans'].labels_,
    #     'cluster_centers': pipe['kmeans'].cluster_centers_
    # }
    return None


# TODO: Geographic Analysis 
def geographic_analysis(dataset_file, data, api_key):
    pass


# TODO: Network Analysis
def network_analysis():
    pass

# TODO: Time Series Analysis
def time_series_analysis():
    pass


def choose_analysis(dataset_file, data, api_key, analyses):
    results = {}
    for analysis in analyses:
        func = eval(analysis)
        res = func(dataset_file, data, api_key)
        if res != None:
            results[analysis] = res
    
    return results


def meta_analysis(dataset_file, data, api_key):
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
    
    analyses = ['outlier_detection', 'regression_analysis', 'correlation_analysis', 'cluster_analysis', 'geographic_analysis']
    
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

    params = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    choose_analysis_func = eval(response['choices'][0]['message']['function_call']['name'])

    print(params)

    analysis_results = choose_analysis_func(dataset_file, data, api_key, **params)
    return analysis_results


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

    Give a short description about the given dataset. Also provide a brief yet detailed description of the given statistical analysis. 
    
    Output in valid markdown format.
    """

    print(prompt)
    response = chat(prompt=prompt, api_key=api_key)
    return response['choices'][0]['message']['content']


# TODO: Describe the insights that were gained by this previous analysis.
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
        responses.append(response['choices'][0]['message']['content'])
    
    return responses

def main():
    api_key = load_env_key()
    dataset_file = get_dataset()
    encoding = get_dataset_encoding(dataset_file)

    try:
        df = pd.read_csv(dataset_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    df = df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # TESTING
    # n_outliers = outlier_detection(dataset_file, df, api_key)
    # print(f'# outliers: {n_outliers} out of {df.shape[0]}')

    # reg_results = regression_analysis(dataset_file, df, api_key)
    # print(reg_results)

    # corr_result = correlation_analysis(dataset_file, df, api_key)
    # print(corr_result)

    # cluster_analysis(dataset_file, df, api_key)

    # ANALYSIS
    # Describe the given dataset.
    generic_analysis_results = generic_analysis(data=df)
    generated_description = describe_generic_analysis(generic_analysis_results, dataset_file, df, api_key)
    write_file('README.md', generated_description)

    # Perform non-generic analysis.
    meta_analysis_results = meta_analysis(dataset_file, df, api_key)
    print(meta_analysis_results)

    generated_meta_analysis_descriptions =  describe_meta_analysis(meta_analysis_results, dataset_file, df, api_key)
    for meta_analysis_description in generated_meta_analysis_descriptions:
        write_file('README.md', meta_analysis_description)

    # TODO: Describe the implications of your findings. (What to do with the insights?)


if __name__ == '__main__':
    main()
