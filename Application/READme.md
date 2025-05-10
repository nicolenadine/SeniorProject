## Application File Tree
```
app/
├── app.py                     
├── load_data.py           
├── assets/  
│   ├── random_variance_heatmaps/   #contains a sample of 16 variance heatmaps for each class
│   └── styles.css
├── content/           # Defines string variables containing paragraph text to keep layout clean
│   ├── __init__.py
│   ├── analytics_text.py
│   ├── data_sources.py    # defines resource paths for deployed application vs. local (for testing) 
│   ├── findings_text.py
│   ├── home_text.py
│   ├── preprocessing_text.py
│   └── objectives_text.py
├── components/              
│   ├── __init__.py
│   ├── confusion_matrix.py
│   ├── data_tables.py
│   ├── metrics_plots.py
│   ├── statistical_tests.py
│   └── variance_plots.py
├── utils/                     
│   ├── __init__.py
│   └── data_processing.py
├── callbacks/               
│   ├── __init__.py
│   └── analytics_callbacks.py
└── pages/                      
    ├── __init__.py             
    ├── analytics.py          
    ├── findings.py           
    ├── home.py               
    ├── preprocessing.py      
    └── project_objective.py    
```
**Notes**

**data_sources.py** 
Contains links to data sources and images that are externally hosted but used within the app.
During local development GitHub links are used whereas the deployed version uses Google's Storage Buckets. 
```IS_LOCAL = os.getenv('IS_LOCAL', 'true').lower() == 'true'``` 
defaults to true inside data_sources.py which causes the application to the GitHub source links
the app.yaml file sets this environment variable to False so that the deployed application loads
external sources through the load_data_from_bucket method inside load_data.py

## Python Package Requirements

| Package | Version | Documentation |
|---------|---------|---------------|
| dash | ~=2.12.1 | [Dash Documentation](https://dash.plotly.com/) |
| plotly | ~=5.18.0 | [Plotly Documentation](https://plotly.com/python/) |
| pandas | ~=2.0.3 | [Pandas Documentation](https://pandas.pydata.org/) |
| Flask | Latest | [Flask Documentation](https://flask.palletsprojects.com/) |
| gunicorn | Latest | [Gunicorn Documentation](https://gunicorn.org/) |
| google-cloud-storage | Latest | [Google Cloud Storage Documentation](https://cloud.google.com/python/docs/reference/storage/latest) |
| dash_bootstrap_components | ==1.4.1 | [Dash Bootstrap Components Documentation](https://dash-bootstrap-components.opensource.faculty.ai/) |
| numpy | ~=1.24.3 | [NumPy Documentation](https://numpy.org/doc/) |
| requests | ~=2.32.3 | [Requests Documentation](https://requests.readthedocs.io/) |
| scipy | ~=1.10.1 | [SciPy Documentation](https://docs.scipy.org/) |
| matplotlib | ~=3.7.5 | [Matplotlib Documentation](https://matplotlib.org/stable/index.html) |
| seaborn | ~=0.13.2 | [Seaborn Documentation](https://seaborn.pydata.org/) |
| pyarrow | ~=14.0.2 | [PyArrow Documentation](https://arrow.apache.org/docs/python/) |

# Installation and Deployment Instructions

This section guides you through setting up the application locally and deploying it to Google App Engine.

## Prerequisites

- Python 3.9 or higher
- Git
- Google Cloud SDK installed ([installation guide](https://cloud.google.com/sdk/docs/install))
- Google Cloud account with billing enabled

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nicolenadine/SeniorProject.git
   cd SeniorProject
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application locally:
   ```bash
   python app.py
   ```

   The app should now be running at `http://localhost:8080`

## Deploying to Google App Engine

1. Login to Google Cloud:
   ```bash
   gcloud auth login
   ```

2. Create a new project (if you haven't already):
   ```bash
   gcloud projects create [YOUR_PROJECT_ID] --name="[YOUR_PROJECT_NAME]"
   ```

3. Upload files contained in metrics_and_testing_data direcrory into google storage bucket (in Google Cloud Console)

4. Update ```BUCKET_NAME``` environment variable inside ```app/app.yaml``` file with the address of storage bucket containing uploaded files.

3. Set the current project (ensure you have billing account attached):
   ```bash
   gcloud config set project [YOUR_PROJECT_ID]
   ```

4. Enable the App Engine Admin API:
   ```bash
   gcloud services enable appengine.googleapis.com
   ```

5. Enable the Cloud Build API:
   ```bash
   gcloud services enable cloudbuild.googleapis.com
   ```

6. Create an App Engine application:
   ```bash
   gcloud app create --region=[REGION]
   ```
   (Choose a region close to your users for better performance)

7. Deploy your application:
   ```bash
   gcloud app deploy
   ```

8. View your deployed application:
   ```bash
   gcloud app browse
   ```


## Troubleshooting

- If deployment fails, check that both the App Engine Admin API and Cloud Build API are enabled
- You may need to set IAM roles 'Artifact Registry Writer' and 'Storage Admin'
- Ensure your `app.yaml` file is properly configured
- Check logs for any errors:
  ```bash
  gcloud app logs tail
  ```

## Additional Resources

- [Google App Engine Documentation](https://cloud.google.com/appengine/docs)
- [Python on App Engine](https://cloud.google.com/appengine/docs/standard/python3)
