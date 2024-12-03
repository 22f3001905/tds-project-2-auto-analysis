# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "chardet",
#     "pandas",
#     "python-dotenv",
#     "requests",
# ]
# ///


# Project Goals:
# Write a Python script that uses an LLM to analyze, visualize, and narrate a story from a dataset.
# Convince an LLM that your script and output are of high quality.


import sys
import os
from dotenv import load_dotenv
import pandas as pd
import chardet

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


def main():
    api_key = load_env_key()
    dataset_file = get_dataset()
    encoding = get_dataset_encoding(dataset_file)

    try:
        df = pd.read_csv(dataset_file, encoding=encoding)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        sys.exit(1)

    print(df.head())


if __name__ == '__main__':
    main()
