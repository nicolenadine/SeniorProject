# Dash Website 

## Overview
This section of the repository contains files related to the published dash application for this project. The site can be found [here](https://senior-project-457222.wl.r.appspot.com).

## File Tree
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

## Notes

### data_sources.py 
Contains links to data sources and images that are externally hosted but used within the app.
During local development GitHub links are used whereas the deployed version uses Google's Storage Buckets. 
```IS_LOCAL = os.getenv('IS_LOCAL', 'true').lower() == 'true'``` 
defaults to true inside data_sources.py which causes the application to the GitHub source links
the app.yaml file sets this environment variable to False so that the deployed application loads
external sources through the load_data_from_bucket method inside load_data.py



