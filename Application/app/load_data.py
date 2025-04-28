import os
from google.cloud import storage
import pandas as pd
import json
from io import StringIO


def load_data_from_bucket(file_name, file_type):
    """
    Load a CSV or JSON file from Google Cloud Storage.

    Args:
        file_name (str): Name of the file in the bucket.
        file_type (str): "csv" or "json".

    Returns:
        pandas.DataFrame if csv, dict if json.
    """
    bucket_name = os.getenv('BUCKET_NAME')
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_text()

    if file_type == "csv":
        return pd.read_csv(StringIO(data))
    elif file_type == "json":
        return json.loads(data)
    else:
        raise ValueError(
            f"Unsupported file_type '{file_type}'. Must be 'csv' or 'json'.")
