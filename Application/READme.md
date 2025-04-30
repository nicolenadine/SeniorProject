**Application File Tree**
```
.
├── .gcloudignore
├── app.yaml
├── requirements.txt
├── app
│   ├── app.py
│   ├── load_data.py
│   ├── assets
│   │   ├── mathjax_loader.js
│   │   ├── pipeline_graphic_part1.png
│   │   ├── pipeline_graphic_part2.png
│   │   ├── styles.css
│   │   └── random_gradcam
│   ├── content
│   │   ├── analytics_text.py
│   │   ├── data_sources.py
│   │   ├── findings_text.py
│   │   ├── home_text.py
│   │   ├── objectives_text.py
│   │   └── preprocessing_text.py
│   └── pages
│       ├── analytics.py
│       ├── findings.py
│       ├── home.py
│       ├── preprocessing.py
│       └── project_objective.py

```

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


**Notes**

**data_sources.py** 
Contains links to data sources and images that are externally hosted but used within the app.
During local development GitHub links are used whereas the deployed version uses Google's Storage Buckets. 
```IS_LOCAL = os.getenv('IS_LOCAL', 'true').lower() == 'true'``` 
defaults to true inside data_sources.py which causes the application to the GitHub source links
the app.yaml file sets this environment variable to False so that the deployed application loads
external sources through the load_data_from_bucket method inside load_data.py
