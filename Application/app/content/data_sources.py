import pandas as pd
import json
import requests
import os

from load_data import load_data_from_bucket

# true for local testing, env variable set to false in yaml file
IS_LOCAL = os.getenv('IS_LOCAL', 'true').lower() == 'true'

if IS_LOCAL:
    # -----------   FOR LOCAL TESTING  ------------------

    # analytics.py data
    cv_metrics = pd.read_csv(
        "https://raw.githubusercontent.com/nicolenadine/SeniorProject/refs/heads/main/metric_and_testing_data/cross_validation_metrics.csv")

    conf_json_url = 'https://raw.githubusercontent.com/nicolenadine/SeniorProject/refs/heads/main/metric_and_testing_data/seg1_ensemble_metrics.json'
    conf_data = requests.get(conf_json_url).json()

    VARIANCE_CSV = 'https://raw.githubusercontent.com/nicolenadine/SeniorProject/refs/heads/main/metric_and_testing_data/segment_variance_data_cleaned.csv'
    PREDICTION_CSV = 'https://raw.githubusercontent.com/nicolenadine/SeniorProject/refs/heads/main/metric_and_testing_data/sample_level_predictions_rounded.csv'

    variance_df = pd.read_csv(VARIANCE_CSV)
    prediction_df = pd.read_csv(PREDICTION_CSV)

    url = 'https://raw.githubusercontent.com/nicolenadine/SeniorProject/refs/heads/main/metric_and_testing_data/mcnemar_comparison_results.json'
    mcnemar_data = requests.get(url).json()

else:
    # -------  FOR DEPLOYMENT IN GOOGLE APP ENGINE ---------

    # analytics.py data
    cv_metrics = load_data_from_bucket('cross_validation_metrics.csv', 'csv')
    conf_data = load_data_from_bucket('seg1_ensemble_metrics.json', 'json')
    variance_df = load_data_from_bucket('segment_variance_data_cleaned.csv',
                                        'csv')
    prediction_df = load_data_from_bucket(
        'sample_level_predictions_rounded.csv', 'csv')
    mcnemar_data = load_data_from_bucket('mcnemar_comparison_results.json',
                                         'json')

# --- images (always from GitHub, no need to switch)
calibration_curves_img = "https://github.com/nicolenadine/SeniorProject/blob/main/plots/calibration_curves.png?raw=true"
benign_prob_dist_img = "https://github.com/nicolenadine/SeniorProject/blob/main/plots/benign_probability_distribution.png?raw=true"
malware_prob_dist_img = "https://github.com/nicolenadine/SeniorProject/blob/main/plots/malware_probability_distribution.png?raw=true"

opcode_embeddings_tsne_img = "https://github.com/nicolenadine/SeniorProject/blob/main/plots/opcode_embeddings_tsne.png?raw=true"
sample_images_img = "https://github.com/nicolenadine/SeniorProject/blob/main/plots/sample_images.png?raw=true"

benign_file_count_img = "https://github.com/nicolenadine/SeniorProject/blob/main/plots/benign_file_count.png?raw=true"
top_malware_families_img = "https://github.com/nicolenadine/SeniorProject/blob/main/plots/Top_20_malware_families_v077.png?raw=true"
