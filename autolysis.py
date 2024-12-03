# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chardet",
#     "pandas",
#     "python-dotenv",
#     "requests",
#     "tabulate",
# ]
# ///


# Project Goals:
# Write a Python script that uses an LLM to analyze, visualize, and narrate a story from a dataset.
# Convince an LLM that your script and output are of high quality.


import io
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


def perform_generic_analysis(data):
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


def write_generic_analysis_md(md_file, results):
    with open(md_file, "w") as f:
        f.write("## Generic Data Analysis\n\n")
        
        f.write("### First 5 Rows\n\n")
        f.write(results['first_5'].to_markdown(index=False) + "\n\n")
        
        f.write("### Dataset Info\n\n")
        f.write("```\n" + results['basic_info'] + "\n```\n\n")
        
        f.write("### Summary Statistics\n\n")
        f.write(results['summary_stats'].to_markdown() + "\n\n")
        
        f.write("### Missing Values\n\n")
        f.write(results['missing_values'].to_markdown() + "\n\n")
        
        f.write("### Column Data Types\n\n")
        f.write(results['column_data_types'].to_markdown() + "\n\n")
        
        f.write("### Unique Values in Each Column\n\n")
        f.write(results['n_unique'].to_markdown() + "\n\n")
        
        f.write("### Correlation Matrix\n\n")
        f.write(results['corr'].to_markdown() + "\n\n")
        
        f.write("### Duplicated Rows\n\n")
        f.write(f"Number of duplicated rows: {results['n_duplicates']}\n\n")

    print(f"Analysis written to {md_file}")


def main():
    api_key = load_env_key()
    dataset_file = get_dataset()
    encoding = get_dataset_encoding(dataset_file)

    try:
        df = pd.read_csv(dataset_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    generic_analysis_results = perform_generic_analysis(data=df)
    write_generic_analysis_md('README.md', results=generic_analysis_results)


if __name__ == '__main__':
    main()
