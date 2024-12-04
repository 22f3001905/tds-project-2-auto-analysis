# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chardet",
#     "json",
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


# def write_generic_analysis_md(md_file, results):
#     with open(md_file, "w") as f:
#         f.write("## Generic Data Analysis\n\n")
        
#         f.write("### First 5 Rows\n\n")
#         f.write(results['first_5'].to_markdown(index=False) + "\n\n")
        
#         f.write("### Dataset Info\n\n")
#         f.write("```\n" + results['basic_info'] + "\n```\n\n")
        
#         f.write("### Summary Statistics\n\n")
#         f.write(results['summary_stats'].to_markdown() + "\n\n")
        
#         f.write("### Missing Values\n\n")
#         f.write(results['missing_values'].to_markdown() + "\n\n")
        
#         f.write("### Column Data Types\n\n")
#         f.write(results['column_data_types'].to_markdown() + "\n\n")
        
#         f.write("### Unique Values in Each Column\n\n")
#         f.write(results['n_unique'].to_markdown() + "\n\n")
        
#         f.write("### Correlation Matrix\n\n")
#         f.write(results['corr'].to_markdown() + "\n\n")
        
#         f.write("### Duplicated Rows\n\n")
#         f.write(f"Number of duplicated rows: {results['n_duplicates']}\n\n")

#     print(f"Analysis written to {md_file}")


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
    return response.json()


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

function_description = []

# Non-generic Analysis
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def outlier_detection(data, columns):
    df = data[columns]
    df = df.select_dtypes(include=['number'])

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
    return (anomaly_score == -1).sum()


def regression_analysis(dataset_file, data, api_key):
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    message = f'You are given a file {dataset_file}.\n\nWith features:\n\n{columns_info}\n\nHere is a sample:\n\n{data.iloc[0, :]}'
    
    response = chat_function_call(prompt=message + "\n\nExtract the features and target for Regression task.", api_key=api_key, function_descriptions=filter_function_description)

    if response.get('error', None):
        exit()

    # print(response)

    params = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    chosen_func = eval(response['choices'][0]['message']['function_call']['name'])
    
    X, y = chosen_func(data=data, **params)

    # TODO
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
        'model': pipe,
        'r2_score': r2_score,
        'mae': mae,
        'mse': mse,
        'rmse': rmse
    }


def correlation_analysis(dataset_file, data, api_key):
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    message = f'You are given a file {dataset_file}.\n\nWith features:\n\n{columns_info}\n\nHere is a sample:\n\n{data.iloc[0, :]}'
    
    response = chat_function_call(prompt=message + "\n\nExtract only the most important features to perform a Correlation analysis. (Use filter_features)", api_key=api_key, function_descriptions=filter_function_description)

    if response.get('error', None):
        exit()

    # print(response)
    
    params = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    chosen_func = eval(response['choices'][0]['message']['function_call']['name'])
    
    df = chosen_func(data=data, **params)
    numeric_data = df.select_dtypes(include=['number'])

    if numeric_data.empty:
        print("\nNo numeric columns found. Cannot compute correlation matrix.")
        return None
    else:
        return numeric_data.corr()


from sklearn.cluster import KMeans

def clustering_analysis(dataset_file, data, api_key):
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
    message = f'You are given a file {dataset_file}.\n\nWith features:\n\n{columns_info}\n\nHere is a sample:\n\n{data.iloc[0, :]}'
    
    response = chat_function_call(prompt=message + "\n\nExtract only the most important features to perform a Clustering analysis using K-Means. (Use filter_features)", api_key=api_key, function_descriptions=filter_function_description)

    if response.get('error', None):
        exit()

    # print(response)
    
    params = json.loads(response['choices'][0]['message']['function_call']['arguments'])
    chosen_func = eval(response['choices'][0]['message']['function_call']['name'])
    
    df = chosen_func(data=data, **params)
    # TODO
    df = df.select_dtypes(include=['number'])

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=4, random_state=42))
    ])

    pipe.fit(df)

    return {
        'cluster_labels': pipe['kmeans'].labels_,
        'cluster_centers': pipe['kmeans'].cluster_centers_
    }

# TODO: Geographic Analysis 
def geographic_analysis(dataset_file, data, api_key):
    pass


# def describe_data(dataset_file, data, api_key):
#     columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])
#     message = (
#         f"You are given a file {dataset_file}.\n\n"
#         "With features:\n"
#         f"{columns_info}\n\n"
#         "Here are a few samples:\n"
#         f"{data.iloc[:3, :]}\n\n"
#     )
#     prompt = message + (
#         "Give a short description about the given dataset. Also give bullet points describing each attribute in the dataset. Just output the description in markdown format."
#     )

#     response = chat(prompt=prompt, api_key=api_key)

#     print(response)
#     # print(response['choices'][0]['message']['content'])
#     return response


def describe_generic_analysis(results, dataset_file, data, api_key):
    columns_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])

    message = (
        f"You are given a file: {dataset_file}\n\n"
        "Features:\n"
        f"{columns_info}\n\n"
        "Here are a few samples:\n"
        f"{results['first_5']}\n\n"
        "Summary Statistics:\n"
        f"{results['summary_stats']}\n\n"
        "Missing Values:\n"
        f"{results['missing_values']}\n\n"
        "Number of unique values in each column:\n"
        f"{results['n_unique']}\n\n"
        "Number of duplicated rows:\n"
        f"{results['n_duplicates']}\n\n"
        "Correlation Analysis:\n"
        f"{results['corr']}\n\n"
    )
    prompt = message + (
        "Give a short description about the given dataset. Also provide a brief yet detailed description of the given statistical analysis. Output in valid markdown format."
    )

    print(prompt)
    response = chat(prompt=prompt, api_key=api_key)
    return response
    # response['choices'][0]['message']['content']


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

    generic_analysis_results = generic_analysis(data=df)
    # write_generic_analysis_md('README.md', results=generic_analysis_results)

    # n_outliers = outlier_detection(df, df.columns)
    # print(f'# outliers: {n_outliers} out of {df.shape[0]}')

    # reg_results = regression_analysis(dataset_file, df, api_key)

    # corr_result = correlation_analysis(dataset_file, df, api_key)
    # print(corr_result)

    # clustering_analysis(dataset_file, df, api_key)

    # description_basic = describe_data(dataset_file, df, api_key)
    description_gen = describe_generic_analysis(generic_analysis_results, dataset_file, df, api_key)


if __name__ == '__main__':
    main()
