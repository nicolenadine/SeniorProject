#!/usr/bin/env python3
"""
Interactive Visualization Module for Malware Classification System
Provides interactive components for exploring model behavior and results
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
from PIL import Image
import json
import io
import base64
import pandas as pd
import html


class InteractiveVisualizer:
    def __init__(self, model=None, data_handler=None, output_dir=None):
        """
        Initialize the interactive visualizer.

        Args:
            model: Trained TensorFlow model
            data_handler: DataHandler instance containing the data
            output_dir: Directory to save visualizations
        """
        self.model = model
        self.data_handler = data_handler
        self.output_dir = output_dir

        if output_dir:
            os.makedirs(os.path.join(output_dir, 'interactive_visualizations'), exist_ok=True)

    def create_saliency_slider_dashboard(self, X_data, y_true, indices=None, n_samples=5):
        """
        Create an interactive dashboard with saliency threshold sliders.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            indices: Specific indices to visualize (if None, samples will be randomly selected)
            n_samples: Number of samples to select if indices is None

        Returns:
            Path to the generated HTML dashboard
        """
        if self.model is None:
            raise ValueError("Model is required for saliency visualization")

        from visualization import GradCAMGenerator

        # Create a GradCAM generator
        gradcam_gen = GradCAMGenerator(
            model=self.model,
            output_dir=None  # We'll handle saving manually
        )

        # Select samples to visualize
        if indices is None:
            # Randomly select samples (ensure balanced class representation)
            class_indices = {}
            for i, label in enumerate(y_true):
                label = int(label)
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(i)

            # Select samples from each class
            indices = []
            for label, idx_list in class_indices.items():
                selected = np.random.choice(idx_list, min(n_samples, len(idx_list)), replace=False)
                indices.extend(selected)

        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model")

        # Generate GradCAM heatmaps for each sample
        heatmaps = []
        for idx in indices:
            cam = gradcam_gen.compute_gradcam(X_data[idx], last_conv_layer)
            heatmaps.append(cam)

        # Generate HTML with interactive sliders
        html_content = self._generate_saliency_slider_html(X_data, y_true, indices, heatmaps)

        # Save HTML file
        if self.output_dir:
            output_path = os.path.join(
                self.output_dir,
                'interactive_visualizations',
                'saliency_slider_dashboard.html'
            )
            with open(output_path, 'w') as f:
                f.write(html_content)
            print(f"Saliency slider dashboard saved to {output_path}")
            return output_path
        else:
            return html_content

    def _generate_saliency_slider_html(self, X_data, y_true, indices, heatmaps):
        """
        Generate HTML for interactive saliency slider dashboard.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            indices: Indices of samples to visualize
            heatmaps: GradCAM heatmaps for each sample

        Returns:
            HTML content as a string
        """
        # Make predictions for selected samples
        if self.model is not None:
            y_pred_prob = self.model.predict(X_data[indices])
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        else:
            y_pred_prob = np.zeros(len(indices))
            y_pred = np.zeros(len(indices), dtype=int)

        # Create HTML string
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Malware Classification - Saliency Slider Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ display: flex; flex-wrap: wrap; justify-content: center; }}
                .sample-card {{ 
                    margin: 15px; 
                    padding: 15px; 
                    background-color: white; 
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    width: 550px;
                }}
                .sample-header {{ 
                    display: flex; 
                    justify-content: space-between; 
                    margin-bottom: 10px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }}
                .sample-title {{ font-size: 16px; font-weight: bold; }}
                .prediction {{
                    padding: 5px 10px;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                .correct {{ background-color: #d4edda; color: #155724; }}
                .incorrect {{ background-color: #f8d7da; color: #721c24; }}
                .visualization {{ 
                    display: flex; 
                    justify-content: space-between;
                    margin-top: 10px;
                }}
                .image-container {{ width: 256px; height: 256px; position: relative; }}
                .controls {{ 
                    width: 250px; 
                    padding: 10px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                }}
                .slider-container {{ margin-bottom: 15px; }}
                .slider {{ width: 100%; }}
                .slider-label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
                .overlay-image {{ 
                    position: absolute; 
                    top: 0; 
                    left: 0; 
                    mix-blend-mode: multiply;
                    opacity: 0.7;
                }}
            </style>
        </head>
        <body>
            <h1>Malware Classification - Saliency Slider Dashboard</h1>
            <p>Adjust the sliders to change the visualization of the GradCAM heatmaps.</p>

            <div class="container">
        """

        # Add a card for each sample
        for i, idx in enumerate(indices):
            # Convert image to base64
            img = X_data[idx]
            if len(img.shape) == 3 and img.shape[2] == 1:
                img = img[:, :, 0]  # Remove channel dimension for grayscale

            # Normalize to [0, 255] for display
            img_display = (img * 255).astype(np.uint8)

            # Convert to PIL Image and then to base64
            pil_img = Image.fromarray(img_display)
            if len(img.shape) == 2:  # If grayscale, specify mode
                pil_img = Image.fromarray(img_display, 'L')

            buffer = io.BytesIO()
            pil_img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Convert heatmap to base64
            heatmap = heatmaps[i]
            heatmap_display = (heatmap * 255).astype(np.uint8)
            heatmap_rgb = np.zeros((heatmap.shape[0], heatmap.shape[1], 3), dtype=np.uint8)

            # Apply jet colormap (simple version)
            for r in range(heatmap.shape[0]):
                for c in range(heatmap.shape[1]):
                    val = heatmap[r, c]
                    if val < 0.25:
                        # Blue to cyan
                        heatmap_rgb[r, c, 2] = 255
                        heatmap_rgb[r, c, 1] = int(val * 4 * 255)
                    elif val < 0.5:
                        # Cyan to green
                        heatmap_rgb[r, c, 1] = 255
                        heatmap_rgb[r, c, 2] = int((0.5 - val) * 4 * 255)
                    elif val < 0.75:
                        # Green to yellow
                        heatmap_rgb[r, c, 1] = 255
                        heatmap_rgb[r, c, 0] = int((val - 0.5) * 4 * 255)
                    else:
                        # Yellow to red
                        heatmap_rgb[r, c, 0] = 255
                        heatmap_rgb[r, c, 1] = int((1.0 - val) * 4 * 255)

            heatmap_pil = Image.fromarray(heatmap_rgb)
            heatmap_buffer = io.BytesIO()
            heatmap_pil.save(heatmap_buffer, format='PNG')
            heatmap_str = base64.b64encode(heatmap_buffer.getvalue()).decode('utf-8')

            # Determine if prediction is correct
            true_label = int(y_true[idx])
            pred_label = int(y_pred[i])
            prob = float(y_pred_prob[i])
            is_correct = true_label == pred_label

            html_content += f"""
            <div class="sample-card" id="sample-{i}">
                <div class="sample-header">
                    <div class="sample-title">Sample #{idx} (True: {true_label})</div>
                    <div class="prediction {('correct' if is_correct else 'incorrect')}">
                        Pred: {pred_label} ({prob:.3f})
                    </div>
                </div>

                <div class="visualization">
                    <div class="image-container">
                        <img src="data:image/png;base64,{img_str}" width="256" height="256">
                        <img src="data:image/png;base64,{heatmap_str}" class="overlay-image" 
                             id="heatmap-{i}" width="256" height="256" style="opacity: 0.7;">
                    </div>

                    <div class="controls">
                        <div class="slider-container">
                            <label class="slider-label" for="opacity-{i}">Opacity: <span id="opacity-value-{i}">0.7</span></label>
                            <input type="range" id="opacity-{i}" class="slider" min="0" max="1" step="0.05" value="0.7"
                                onInput="updateOpacity({i}, this.value)">
                        </div>

                        <div class="slider-container">
                            <label class="slider-label" for="threshold-{i}">Threshold: <span id="threshold-value-{i}">0.0</span></label>
                            <input type="range" id="threshold-{i}" class="slider" min="0" max="1" step="0.05" value="0.0"
                                onInput="updateThreshold({i}, this.value)">
                        </div>

                        <div class="slider-container">
                            <label class="slider-label">Blend Mode:</label>
                            <select id="blend-mode-{i}" onchange="updateBlendMode({i}, this.value)">
                                <option value="multiply">Multiply</option>
                                <option value="screen">Screen</option>
                                <option value="overlay">Overlay</option>
                                <option value="darken">Darken</option>
                                <option value="lighten">Lighten</option>
                            </select>
                        </div>
                    </div>
                </div>
            </div>
            """

        # Add JavaScript for interactivity
        html_content += """
            </div>

            <script>
                function updateOpacity(index, value) {
                    document.getElementById(`heatmap-${index}`).style.opacity = value;
                    document.getElementById(`opacity-value-${index}`).innerText = value;
                }

                function updateThreshold(index, value) {
                    // This is a visual approximation of thresholding
                    // In a real application, you would recompute the heatmap
                    const heatmap = document.getElementById(`heatmap-${index}`);
                    heatmap.style.filter = `brightness(${1 + parseFloat(value)}) contrast(${1 + parseFloat(value) * 3})`;
                    document.getElementById(`threshold-value-${index}`).innerText = value;
                }

                function updateBlendMode(index, value) {
                    document.getElementById(`heatmap-${index}`).style.mixBlendMode = value;
                }
            </script>
        </body>
        </html>
        """

        return html_content

    def create_comparative_dashboard(self, results_paths, model_names=None):
        """
        Create an interactive dashboard for comparing models side by side.

        Args:
            results_paths: List of paths to model results directories
            model_names: Optional list of model names for display

        Returns:
            Path to the generated HTML dashboard
        """
        if not results_paths:
            raise ValueError("At least one results path must be provided")

        # Set default model names if not provided
        if model_names is None:
            model_names = [f"Model {i + 1}" for i in range(len(results_paths))]
        elif len(model_names) != len(results_paths):
            raise ValueError("Number of model names must match number of results paths")

        # Load metrics for each model
        models_data = []
        for i, path in enumerate(results_paths):
            model_data = {
                'name': model_names[i],
                'path': path
            }

            # Try to find metrics JSON file
            metrics_path = None
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('_metrics.json'):
                        metrics_path = os.path.join(root, file)
                        break
                if metrics_path:
                    break

            if metrics_path:
                try:
                    with open(metrics_path, 'r') as f:
                        model_data['metrics'] = json.load(f)
                except:
                    print(f"Could not load metrics from {metrics_path}")
                    model_data['metrics'] = None
            else:
                print(f"No metrics file found for {model_names[i]}")
                model_data['metrics'] = None

            # Try to find sample images
            samples_dir = None
            for root, dirs, files in os.walk(path):
                if 'error_analysis' in root and 'difficult_examples' in root:
                    samples_dir = root
                    break
            model_data['samples_dir'] = samples_dir

            models_data.append(model_data)

        # Generate HTML dashboard
        html_content = self._generate_comparative_dashboard_html(models_data)

        # Save HTML file
        if self.output_dir:
            output_path = os.path.join(
                self.output_dir,
                'interactive_visualizations',
                'comparative_dashboard.html'
            )
            with open(output_path, 'w') as f:
                f.write(html_content)
            print(f"Comparative dashboard saved to {output_path}")
            return output_path
        else:
            return html_content

    def _generate_comparative_dashboard_html(self, models_data):
        """
        Generate HTML for interactive comparative dashboard.

        Args:
            models_data: List of dictionaries containing model information

        Returns:
            HTML content as a string
        """
        # Create HTML string
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Malware Classification - Model Comparison Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .tabs {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                .tabs button {{ 
                    background-color: inherit; 
                    float: left; 
                    border: none; 
                    outline: none; 
                    cursor: pointer; 
                    padding: 14px 16px; 
                    transition: 0.3s; 
                    font-size: 16px; 
                }}
                .tabs button:hover {{ background-color: #ddd; }}
                .tabs button.active {{ background-color: #ccc; }}
                .tabcontent {{ display: none; padding: 20px; border: 1px solid #ccc; border-top: none; }}
                .model-comparison {{ display: flex; flex-wrap: wrap; margin-top: 20px; }}
                .model-card {{ 
                    margin: 15px; 
                    padding: 15px; 
                    background-color: white; 
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    width: 300px;
                }}
                .model-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .metric-container {{ 
                    display: flex; 
                    justify-content: space-between;
                    margin-bottom: 5px;
                }}
                .metric-name {{ font-weight: bold; }}
                .metric-value {{ }}
                .chart-container {{ 
                    width: 100%; 
                    height: 400px; 
                    margin-top: 20px; 
                    background-color: white; 
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    padding: 20px;
                }}
                .example-container {{ 
                    display: flex; 
                    flex-wrap: wrap; 
                    justify-content: center; 
                    margin-top: 20px; 
                }}
                .example-card {{ 
                    margin: 10px; 
                    padding: 10px; 
                    background-color: white; 
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    width: 300px;
                }}
                .model-selector {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .model-selector select {{
                    padding: 10px;
                    font-size: 16px;
                    border-radius: 5px;
                }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>Malware Classification - Model Comparison Dashboard</h1>

            <div class="tabs">
                <button class="tablinks active" onclick="openTab(event, 'metrics')">Performance Metrics</button>
                <button class="tablinks" onclick="openTab(event, 'curves')">ROC & PR Curves</button>
                <button class="tablinks" onclick="openTab(event, 'examples')">Example Comparisons</button>
            </div>

            <div id="metrics" class="tabcontent" style="display: block;">
                <h2>Performance Metrics Comparison</h2>
                <div class="model-comparison">
        """

        # Add model metrics cards
        for model_data in models_data:
            html_content += f"""
            <div class="model-card">
                <div class="model-title">{model_data['name']}</div>
            """

            if model_data['metrics'] and 'metrics' in model_data['metrics']:
                threshold_metrics = model_data['metrics']['metrics'].get('threshold_0.5', {})
                roc_auc = model_data['metrics']['roc'].get('auc', 'N/A')
                pr_auc = model_data['metrics']['precision_recall'].get('average_precision', 'N/A')

                # Add key metrics
                metrics_to_show = {
                    'Accuracy': threshold_metrics.get('accuracy', 'N/A'),
                    'Precision': threshold_metrics.get('precision', 'N/A'),
                    'Recall': threshold_metrics.get('recall', 'N/A'),
                    'F1 Score': threshold_metrics.get('f1_score', 'N/A'),
                    'AUC-ROC': roc_auc,
                    'AUC-PR': pr_auc
                }

                for name, value in metrics_to_show.items():
                    if isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = value

                    html_content += f"""
                    <div class="metric-container">
                        <div class="metric-name">{name}:</div>
                        <div class="metric-value">{formatted_value}</div>
                    </div>
                    """

                # Add inference time if available
                if 'inference_time' in model_data['metrics']:
                    inference_time = model_data['metrics']['inference_time']
                    html_content += f"""
                    <div class="metric-container">
                        <div class="metric-name">Samples/second:</div>
                        <div class="metric-value">{inference_time.get('samples_per_second', 'N/A'):.2f}</div>
                    </div>
                    <div class="metric-container">
                        <div class="metric-name">ms/sample:</div>
                        <div class="metric-value">{inference_time.get('ms_per_sample', 'N/A'):.2f}</div>
                    </div>
                    """
            else:
                html_content += "<p>No metrics available</p>"

            html_content += "</div>"  # End model-card

        html_content += """
                </div>

                <!-- Add a bar chart comparing key metrics -->
                <div class="chart-container">
                    <canvas id="metricsChart"></canvas>
                </div>
            </div>

            <div id="curves" class="tabcontent">
                <h2>ROC and Precision-Recall Curves</h2>

                <div style="display: flex; justify-content: space-around; margin-bottom: 20px;">
                    <div style="width: 45%;">
                        <canvas id="rocChart"></canvas>
                    </div>
                    <div style="width: 45%;">
                        <canvas id="prChart"></canvas>
                    </div>
                </div>
            </div>

            <div id="examples" class="tabcontent">
                <h2>Example Comparisons</h2>

                <div class="model-selector">
                    <select id="exampleType" onchange="updateExamples()">
                        <option value="difficult">Difficult Examples</option>
                        <option value="confident">Confident Examples</option>
                        <option value="boundary">Boundary Examples</option>
                    </select>
                </div>

                <div class="example-container" id="exampleContainer">
                    <!-- Examples will be loaded here dynamically -->
                </div>
            </div>

            <script>
                // Tab functionality
                function openTab(evt, tabName) {
                    var i, tabcontent, tablinks;
                    tabcontent = document.getElementsByClassName("tabcontent");
                    for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                    }
                    tablinks = document.getElementsByClassName("tablinks");
                    for (i = 0; i < tablinks.length; i++) {
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                    }
                    document.getElementById(tabName).style.display = "block";
                    evt.currentTarget.className += " active";
                }

                // Parse model data
                const modelData = {
        """

        # Add JavaScript data for models
        for i, model_data in enumerate(models_data):
            html_content += f"'{model_data['name']}': {{"

            if model_data['metrics'] and 'metrics' in model_data['metrics']:
                threshold_metrics = model_data['metrics']['metrics'].get('threshold_0.5', {})
                roc_data = model_data['metrics']['roc']
                pr_data = model_data['metrics']['precision_recall']

                html_content += f"""
                    accuracy: {threshold_metrics.get('accuracy', 'null')},
                    precision: {threshold_metrics.get('precision', 'null')},
                    recall: {threshold_metrics.get('recall', 'null')},
                    f1_score: {threshold_metrics.get('f1_score', 'null')},
                    auc_roc: {roc_data.get('auc', 'null')},
                    auc_pr: {pr_data.get('average_precision', 'null')},
                    roc_fpr: {json.dumps(roc_data.get('fpr', []))},
                    roc_tpr: {json.dumps(roc_data.get('tpr', []))},
                    pr_precision: {json.dumps(pr_data.get('precision', []))},
                    pr_recall: {json.dumps(pr_data.get('recall', []))},
                """

                if 'inference_time' in model_data['metrics']:
                    inference_time = model_data['metrics']['inference_time']
                    html_content += f"""
                    samples_per_second: {inference_time.get('samples_per_second', 'null')},
                    ms_per_sample: {inference_time.get('ms_per_sample', 'null')},
                    """

            html_content += "},"

        html_content += """
                };

                // Create metrics chart
                const metricsCtx = document.getElementById('metricsChart').getContext('2d');
                const modelNames = Object.keys(modelData);

                const metricsChart = new Chart(metricsCtx, {
                    type: 'bar',
                    data: {
                        labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'AUC-PR'],
                        datasets: modelNames.map((name, index) => {
                            const colors = ['rgba(75, 192, 192, 0.7)', 'rgba(255, 99, 132, 0.7)', 
                                          'rgba(54, 162, 235, 0.7)', 'rgba(255, 206, 86, 0.7)'];
                            return {
                                label: name,
                                data: [
                                    modelData[name].accuracy, 
                                    modelData[name].precision, 
                                    modelData[name].recall, 
                                    modelData[name].f1_score,
                                    modelData[name].auc_roc,
                                    modelData[name].auc_pr
                                ],
                                backgroundColor: colors[index % colors.length],
                                borderColor: colors[index % colors.length].replace('0.7', '1'),
                                borderWidth: 1
                            };
                        })
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { position: 'top' },
                            title: { display: true, text: 'Performance Metrics Comparison' }
                        },
                        scales: {
                            y: { beginAtZero: true, max: 1 }
                        }
                    }
                });

                // Create ROC chart
                const rocCtx = document.getElementById('rocChart').getContext('2d');
                const rocChart = new Chart(rocCtx, {
                    type: 'line',
                    data: {
                        datasets: modelNames.map((name, index) => {
                            const colors = ['rgb(75, 192, 192)', 'rgb(255, 99, 132)', 
                                          'rgb(54, 162, 235)', 'rgb(255, 206, 86)'];
                            return {
                                label: `${name} (AUC = ${modelData[name].auc_roc?.toFixed(4) || 'N/A'})`,
                                data: modelData[name].roc_fpr?.map((fpr, i) => ({ x: fpr, y: modelData[name].roc_tpr[i] })),
                                borderColor: colors[index % colors.length],
                                borderWidth: 2,
                                fill: false,
                                pointRadius: 0
                            };
                        })
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { position: 'top' },
                            title: { display: true, text: 'ROC Curves' }
                        },
                        scales: {
                            x: { 
                                title: { display: true, text: 'False Positive Rate' },
                                beginAtZero: true,
                                max: 1
                            },
                            y: { 
                                title: { display: true, text: 'True Positive Rate' },
                                beginAtZero: true,
                                max: 1
                            }
                        }
                    }
                });

                // Create PR chart
                const prCtx = document.getElementById('prChart').getContext('2d');
                const prChart = new Chart(prCtx, {
                    type: 'line',
                    data: {
                        datasets: modelNames.map((name, index) => {
                            const colors = ['rgb(75, 192, 192)', 'rgb(255, 99, 132)', 
                                          'rgb(54, 162, 235)', 'rgb(255, 206, 86)'];
                            return {
                                label: `${name} (AP = ${modelData[name].auc_pr?.toFixed(4) || 'N/A'})`,
                                data: modelData[name].pr_recall?.map((recall, i) => ({ x: recall, y: modelData[name].pr_precision[i] })),
                                borderColor: colors[index % colors.length],
                                borderWidth: 2,
                                fill: false,
                                pointRadius: 0
                            };
                        })
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { position: 'top' },
                            title: { display: true, text: 'Precision-Recall Curves' }
                        },
                        scales: {
                            x: { 
                                title: { display: true, text: 'Recall' },
                                beginAtZero: true,
                                max: 1
                            },
                            y: { 
                                title: { display: true, text: 'Precision' },
                                beginAtZero: true,
                                max: 1
                            }
                        }
                    }
                });
                
                // Function to update examples tab
                function updateExamples() {
                    const exampleType = document.getElementById('exampleType').value;
                    const container = document.getElementById('exampleContainer');
                    container.innerHTML = '<p>Loading examples...</p>';
                    
                    // In a real implementation, you would load examples based on the type
                    // For this HTML prototype, we'll just show a message
                    container.innerHTML = `<p>Example comparisons would be loaded here based on type: ${exampleType}</p>`;
                }
                
                // Initialize the page
                document.addEventListener('DOMContentLoaded', function() {
                    // Default to first tab
                    document.getElementsByClassName('tablinks')[0].click();
                    
                    // Initialize examples
                    updateExamples();
                });
            </script>
        </body>
        </html>
        """

        return html_content

    def create_feature_map_explorer(self, X_data, indices=None, n_samples=5):
        """
        Create an interactive feature map explorer to visualize activations across layers.

        Args:
            X_data: Input data
            indices: Specific indices to visualize (if None, samples will be randomly selected)
            n_samples: Number of samples to select if indices is None

        Returns:
            Path to the generated HTML dashboard
        """
        if self.model is None:
            raise ValueError("Model is required for feature map exploration")

        # Select samples to visualize
        if indices is None:
            indices = np.random.choice(len(X_data), min(n_samples, len(X_data)), replace=False)

        # Find convolutional layers
        conv_layers = []
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                conv_layers.append({'index': i, 'name': layer.name})

        if not conv_layers:
            raise ValueError("No convolutional layers found in the model")

        # Generate feature maps for each sample and layer
        feature_maps = {}
        for idx in indices:
            # Prepare input
            img = X_data[idx:idx+1]  # Add batch dimension if needed

            feature_maps[idx] = {}
            for layer_info in conv_layers:
                # Create a model that outputs the layer activations
                layer_model = tf.keras.models.Model(
                    inputs=self.model.inputs,
                    outputs=self.model.layers[layer_info['index']].output
                )

                # Get layer activations
                activations = layer_model.predict(img)
                feature_maps[idx][layer_info['name']] = activations[0]  # Remove batch dimension

        # Generate HTML with interactive feature map explorer
        html_content = self._generate_feature_map_explorer_html(X_data, indices, conv_layers, feature_maps)

        # Save HTML file
        if self.output_dir:
            output_path = os.path.join(
                self.output_dir,
                'interactive_visualizations',
                'feature_map_explorer.html'
            )
            with open(output_path, 'w') as f:
                f.write(html_content)
            print(f"Feature map explorer saved to {output_path}")
            return output_path
        else:
            return html_content

    def _generate_feature_map_explorer_html(self, X_data, indices, conv_layers, feature_maps):
        """
        Generate HTML for interactive feature map explorer.

        Args:
            X_data: Input data
            indices: Indices of samples to visualize
            conv_layers: List of convolutional layer information
            feature_maps: Dictionary of feature maps for each sample and layer

        Returns:
            HTML content as a string
        """
        # Create HTML string
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Malware Classification - Feature Map Explorer</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ display: flex; flex-direction: column; align-items: center; }}
                .controls {{ 
                    display: flex; 
                    justify-content: center;
                    margin: 20px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .control-group {{ margin: 0 15px; }}
                .label {{ font-weight: bold; margin-bottom: 5px; }}
                select {{ padding: 8px; border-radius: 4px; border: 1px solid #ddd; }}
                .visualization {{ 
                    display: flex; 
                    flex-wrap: wrap;
                    justify-content: center;
                    max-width: 1200px;
                }}
                .sample-container {{ 
                    margin: 15px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .sample-header {{ font-weight: bold; margin-bottom: 10px; }}
                .original-image {{ 
                    width: 256px; 
                    height: 256px; 
                    object-fit: contain;
                    margin-bottom: 10px;
                }}
                .feature-maps {{ 
                    display: flex; 
                    flex-wrap: wrap;
                    justify-content: center;
                    max-width: 800px;
                }}
                .feature-map {{ 
                    margin: 5px;
                    width: 64px;
                    height: 64px;
                    object-fit: contain;
                }}
                .feature-map-canvas {{ 
                    width: 64px;
                    height: 64px;
                }}
            </style>
        </head>
        <body>
            <h1>Malware Classification - Feature Map Explorer</h1>
            <p>Explore feature maps (activations) across different layers of the model.</p>
            
            <div class="container">
                <div class="controls">
                    <div class="control-group">
                        <div class="label">Sample:</div>
                        <select id="sample-selector" onchange="updateVisualization()">
        """

        # Add sample options
        for i, idx in enumerate(indices):
            html_content += f'<option value="{idx}">Sample #{idx}</option>\n'

        html_content += """
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <div class="label">Layer:</div>
                        <select id="layer-selector" onchange="updateVisualization()">
        """

        # Add layer options
        for layer_info in conv_layers:
            html_content += f'<option value="{layer_info["name"]}">{layer_info["name"]}</option>\n'

        html_content += """
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <div class="label">Colormap:</div>
                        <select id="colormap-selector" onchange="updateVisualization()">
                            <option value="viridis">Viridis</option>
                            <option value="plasma">Plasma</option>
                            <option value="inferno">Inferno</option>
                            <option value="magma">Magma</option>
                            <option value="cividis">Cividis</option>
                            <option value="jet">Jet</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <div class="label">Filter Count:</div>
                        <select id="filter-count" onchange="updateVisualization()">
                            <option value="16">16</option>
                            <option value="32">32</option>
                            <option value="64">64</option>
                            <option value="all">All</option>
                        </select>
                    </div>
                </div>
                
                <div class="visualization" id="visualization">
                    <!-- Feature maps will be loaded here dynamically -->
                </div>
            </div>
            
            <script>
                // Store feature map data
                const featureMaps = {
        """

        # Add JavaScript data for feature maps
        for idx in indices:
            html_content += f"'{idx}': {{\n"

            for layer_name, activations in feature_maps[idx].items():
                # Convert numpy array to list
                activations_list = activations.tolist()
                html_content += f"'{layer_name}': {json.dumps(activations_list)},\n"

            html_content += "},\n"

        html_content += """
                };
                
                // Function to apply colormap
                function applyColormap(value, colormap) {
                    // Normalize value to [0, 1]
                    const v = Math.max(0, Math.min(1, value));
                    
                    if (colormap === 'jet') {
                        // Simple jet colormap
                        let r, g, b;
                        if (v < 0.25) {
                            // Blue to cyan
                            r = 0;
                            g = 4 * v;
                            b = 1;
                        } else if (v < 0.5) {
                            // Cyan to green
                            r = 0;
                            g = 1;
                            b = 1 - 4 * (v - 0.25);
                        } else if (v < 0.75) {
                            // Green to yellow
                            r = 4 * (v - 0.5);
                            g = 1;
                            b = 0;
                        } else {
                            // Yellow to red
                            r = 1;
                            g = 1 - 4 * (v - 0.75);
                            b = 0;
                        }
                        return [r * 255, g * 255, b * 255];
                    } else if (colormap === 'viridis') {
                        // Simplified viridis colormap
                        const r = 0.267 + 0.3 * Math.sin((1.2 + v) * Math.PI);
                        const g = 0.267 + 0.5 * Math.sin((0.4 + v) * Math.PI);
                        const b = 0.267 + 0.9 * Math.sin((0.1 + v) * Math.PI);
                        return [r * 255, g * 255, b * 255];
                    } else if (colormap === 'plasma') {
                        // Simplified plasma colormap
                        const r = 0.5 + 0.5 * Math.sin((0.9 + v) * Math.PI);
                        const g = 0.3 + 0.5 * Math.sin((0.35 + v) * Math.PI);
                        const b = 0.5 + 0.5 * Math.sin((v - 0.2) * Math.PI);
                        return [r * 255, g * 255, b * 255];
                    } else if (colormap === 'inferno') {
                        // Simplified inferno colormap
                        const r = 0.3 + 0.7 * v;
                        const g = 0.1 + 0.7 * Math.sin(v * Math.PI);
                        const b = 0.4 * (1 - v);
                        return [r * 255, g * 255, b * 255];
                    } else if (colormap === 'magma') {
                        // Simplified magma colormap
                        const r = 0.2 + 0.8 * v;
                        const g = 0.1 + 0.5 * Math.sin(v * Math.PI);
                        const b = 0.3 * (1 - v) + 0.4 * Math.pow(v, 3);
                        return [r * 255, g * 255, b * 255];
                    } else if (colormap === 'cividis') {
                        // Simplified cividis colormap
                        const r = 0.3 + 0.5 * v;
                        const g = 0.3 + 0.5 * v;
                        const b = 0.5 - 0.3 * v;
                        return [r * 255, g * 255, b * 255];
                    }
                    
                    // Default grayscale
                    return [v * 255, v * 255, v * 255];
                }
                
                // Function to draw feature map
                function drawFeatureMap(canvas, featureMap, colormap) {
                    const ctx = canvas.getContext('2d');
                    const width = featureMap.length;
                    const height = featureMap[0].length;
                    
                    // Find min and max for normalization
                    let min = Infinity;
                    let max = -Infinity;
                    
                    for (let x = 0; x < width; x++) {
                        for (let y = 0; y < height; y++) {
                            const value = featureMap[x][y];
                            min = Math.min(min, value);
                            max = Math.max(max, value);
                        }
                    }
                    
                    // Create image data
                    const imageData = ctx.createImageData(width, height);
                    
                    for (let x = 0; x < width; x++) {
                        for (let y = 0; y < height; y++) {
                            // Normalize value
                            const value = (featureMap[x][y] - min) / (max - min);
                            
                            // Apply colormap
                            const [r, g, b] = applyColormap(value, colormap);
                            
                            // Set pixel color (rgba)
                            const idx = (y * width + x) * 4;
                            imageData.data[idx] = r;
                            imageData.data[idx + 1] = g;
                            imageData.data[idx + 2] = b;
                            imageData.data[idx + 3] = 255; // Alpha
                        }
                    }
                    
                    // Resize canvas to match feature map dimensions
                    canvas.width = width;
                    canvas.height = height;
                    
                    // Draw image data
                    ctx.putImageData(imageData, 0, 0);
                }
                
                // Function to update visualization
                function updateVisualization() {
                    const sampleIdx = document.getElementById('sample-selector').value;
                    const layerName = document.getElementById('layer-selector').value;
                    const colormap = document.getElementById('colormap-selector').value;
                    const filterCount = document.getElementById('filter-count').value;
                    
                    // Get feature maps for selected sample and layer
                    const activations = featureMaps[sampleIdx][layerName];
                    
                    // Create visualization container
                    const container = document.getElementById('visualization');
                    container.innerHTML = '';
                    
                    // Create sample container
                    const sampleContainer = document.createElement('div');
                    sampleContainer.className = 'sample-container';
                    
                    // Add sample header
                    const header = document.createElement('div');
                    header.className = 'sample-header';
                    header.innerText = `Sample #${sampleIdx} - Layer: ${layerName}`;
                    sampleContainer.appendChild(header);
                    
                    // Add original image
                    // In a real implementation, you would include the original image
                    
                    // Add feature maps container
                    const featureMapsContainer = document.createElement('div');
                    featureMapsContainer.className = 'feature-maps';
                    
                    // Determine how many filters to show
                    const numFilters = activations.length;
                    let filtersToShow = numFilters;
                    
                    if (filterCount !== 'all') {
                        filtersToShow = Math.min(parseInt(filterCount), numFilters);
                    }
                    
                    // Add canvases for feature maps
                    for (let i = 0; i < filtersToShow; i++) {
                        const canvas = document.createElement('canvas');
                        canvas.className = 'feature-map-canvas';
                        canvas.title = `Filter ${i}`;
                        featureMapsContainer.appendChild(canvas);
                        
                        // Draw feature map
                        drawFeatureMap(canvas, activations[i], colormap);
                    }
                    
                    sampleContainer.appendChild(featureMapsContainer);
                    container.appendChild(sampleContainer);
                }
                
                // Initialize visualization when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    updateVisualization();
                });
            </script>
        </body>
        </html>
        """

        return html_content

    def create_side_by_side_comparison(self, model_paths, X_data, y_true, indices=None, n_samples=5):
        """
        Create a side-by-side comparison of model predictions and visualizations.

        Args:
            model_paths: List of paths to model files
            X_data: Input data
            y_true: Ground truth labels
            indices: Specific indices to visualize (if None, samples will be randomly selected)
            n_samples: Number of samples to select if indices is None

        Returns:
            Path to the generated HTML dashboard
        """
        if not model_paths:
            raise ValueError("At least one model path must be provided")

        # Select samples to visualize
        if indices is None:
            indices = np.random.choice(len(X_data), min(n_samples, len(X_data)), replace=False)

        # Load models
        models = []
        model_names = []
        for i, path in enumerate(model_paths):
            try:
                model = tf.keras.models.load_model(path, compile=False)
                models.append(model)
                model_names.append(f"Model {i+1}")
            except Exception as e:
                print(f"Error loading model from {path}: {e}")

        if not models:
            raise ValueError("Could not load any models")

        # Generate predictions and visualizations
        model_results = []
        for i, model in enumerate(models):
            # Make predictions
            y_pred = model.predict(X_data[indices])
            y_pred = (y_pred > 0.5).astype(int).flatten()
            y_pred_prob = model.predict(X_data[indices]).flatten()

            # Find the last convolutional layer
            last_conv_layer = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer = layer.name
                    break

            # Generate GradCAM if convolutional layers exist
            gradcams = None
            if last_conv_layer:
                from visualization import GradCAMGenerator
                gradcam_gen = GradCAMGenerator(model=model, output_dir=None)
                gradcams = []
                for idx in indices:
                    cam = gradcam_gen.compute_gradcam(X_data[idx], last_conv_layer)
                    gradcams.append(cam)

            model_results.append({
                'name': model_names[i],
                'y_pred': y_pred,
                'y_pred_prob': y_pred_prob,
                'gradcams': gradcams
            })

        # Generate HTML with side-by-side comparison
        html_content = self._generate_side_by_side_comparison_html(X_data, y_true, indices, model_results)

        # Save HTML file
        if self.output_dir:
            output_path = os.path.join(
                self.output_dir,
                'interactive_visualizations',
                'side_by_side_comparison.html'
            )
            with open(output_path, 'w') as f:
                f.write(html_content)
            print(f"Side-by-side comparison saved to {output_path}")
            return output_path
        else:
            return html_content

    def _generate_side_by_side_comparison_html(self, X_data, y_true, indices, model_results):
        """
        Generate HTML for side-by-side comparison.

        Args:
            X_data: Input data
            y_true: Ground truth labels
            indices: Indices of samples to visualize
            model_results: List of dictionaries containing model results

        Returns:
            HTML content as a string
        """
        # Create HTML string
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Malware Classification - Side-by-Side Comparison</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .container {{ display: flex; flex-direction: column; align-items: center; }}
                .controls {{ 
                    display: flex; 
                    justify-content: center;
                    margin: 20px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                .control-group {{ margin: 0 15px; }}
                .label {{ font-weight: bold; margin-bottom: 5px; }}
                select {{ padding: 8px; border-radius: 4px; border: 1px solid #ddd; }}
                .sample-container {{ 
                    margin: 15px;
                    padding: 15px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    width: 90%;
                    max-width: 1400px;
                }}
                .sample-header {{ 
                    font-weight: bold; 
                    margin-bottom: 10px; 
                    padding-bottom: 10px;
                    border-bottom: 1px solid #eee;
                }}
                .model-comparison {{ 
                    display: flex; 
                    flex-wrap: wrap;
                    justify-content: center;
                }}
                .model-result {{ 
                    margin: 10px;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    width: 280px;
                }}
                .model-header {{ 
                    font-weight: bold; 
                    text-align: center;
                    margin-bottom: 10px;
                }}
                .prediction {{ 
                    text-align: center;
                    padding: 5px;
                    margin-bottom: 10px;
                    border-radius: 4px;
                    font-weight: bold;
                }}
                .correct {{ background-color: #d4edda; color: #155724; }}
                .incorrect {{ background-color: #f8d7da; color: #721c24; }}
                .visualization-container {{ 
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .original-image {{ 
                    width: 100%;
                    height: auto;
                    max-width: 256px;
                    margin-bottom: 10px;
                }}
                .gradcam-image {{ 
                    width: 100%;
                    height: auto;
                    max-width: 256px;
                }}
                .image-overlay {{
                    position: relative;
                    width: 256px;
                    height: 256px;
                }}
                .overlay-image {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    mix-blend-mode: multiply;
                    opacity: 0.7;
                }}
            </style>
        </head>
        <body>
            <h1>Malware Classification - Side-by-Side Model Comparison</h1>
            <p>Compare predictions and visualizations from different models.</p>
            
            <div class="container">
                <div class="controls">
                    <div class="control-group">
                        <div class="label">Sample:</div>
                        <select id="sample-selector" onchange="updateVisualization()">
        """

        # Add sample options
        for i, idx in enumerate(indices):
            html_content += f'<option value="{i}">Sample #{idx}</option>\n'

        html_content += """
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <div class="label">Visualization:</div>
                        <select id="viz-type" onchange="updateVisualization()">
                            <option value="separate">Separate Images</option>
                            <option value="overlay">Overlay</option>
                        </select>
                    </div>
                    
                    <div class="control-group">
                        <div class="label">Opacity:</div>
                        <input type="range" id="opacity-slider" min="0" max="1" step="0.05" value="0.7" onInput="updateOpacity(this.value)">
                        <span id="opacity-value">0.7</span>
                    </div>
                </div>
            </div>
            
            <div id="sample-containers">
                <!-- Sample containers will be created dynamically -->
            </div>
            
            <script>
                // Store results data
                const indices = ${json.dumps(indices.tolist())};
                const trueLabels = ${json.dumps(y_true[indices].tolist())};
                const modelResults = [
        """

        # Add JavaScript data for model results
        for result in model_results:
            html_content += "{\n"
            html_content += f"name: '{result['name']}',\n"
            html_content += f"predictions: {json.dumps(result['y_pred'].tolist())},\n"
            html_content += f"probabilities: {json.dumps(result['y_pred_prob'].tolist())},\n"

            if result['gradcams'] is not None:
                # Convert numpy arrays to lists for JSON serialization
                gradcams_list = [cam.tolist() for cam in result['gradcams']]
                html_content += f"gradcams: {json.dumps(gradcams_list)},\n"
            else:
                html_content += "gradcams: null,\n"

            html_content += "},\n"

        html_content += """
                ];
                
                // Function to create image data URL
                function createImageDataURL(imageArray, width, height, colormap = null) {
                    const canvas = document.createElement('canvas');
                    canvas.width = width;
                    canvas.height = height;
                    const ctx = canvas.getContext('2d');
                    const imageData = ctx.createImageData(width, height);
                    
                    // Find min and max for normalization
                    let min = Infinity;
                    let max = -Infinity;
                    
                    for (let i = 0; i < imageArray.length; i++) {
                        for (let j = 0; j < imageArray[i].length; j++) {
                            const value = imageArray[i][j];
                            min = Math.min(min, value);
                            max = Math.max(max, value);
                        }
                    }
                    
                    // Create image data
                    for (let y = 0; y < height; y++) {
                        for (let x = 0; x < width; x++) {
                            const idx = (y * width + x) * 4;
                            
                            if (colormap === 'jet') {
                                // Normalize value
                                const value = (imageArray[y][x] - min) / (max - min);
                                
                                // Apply jet colormap
                                let r, g, b;
                                if (value < 0.25) {
                                    // Blue to cyan
                                    r = 0;
                                    g = 4 * value;
                                    b = 1;
                                } else if (value < 0.5) {
                                    // Cyan to green
                                    r = 0;
                                    g = 1;
                                    b = 1 - 4 * (value - 0.25);
                                } else if (value < 0.75) {
                                    // Green to yellow
                                    r = 4 * (value - 0.5);
                                    g = 1;
                                    b = 0;
                                } else {
                                    // Yellow to red
                                    r = 1;
                                    g = 1 - 4 * (value - 0.75);
                                    b = 0;
                                }
                                
                                imageData.data[idx] = r * 255;
                                imageData.data[idx + 1] = g * 255;
                                imageData.data[idx + 2] = b * 255;
                                imageData.data[idx + 3] = 255; // Alpha
                            } else {
                                // Grayscale
                                const value = (imageArray[y][x] - min) / (max - min) * 255;
                                imageData.data[idx] = value;
                                imageData.data[idx + 1] = value;
                                imageData.data[idx + 2] = value;
                                imageData.data[idx + 3] = 255; // Alpha
                            }
                        }
                    }
                    
                    ctx.putImageData(imageData, 0, 0);
                    return canvas.toDataURL();
                }
                
                // Function to update visualization
                function updateVisualization() {
                    const sampleIndex = parseInt(document.getElementById('sample-selector').value);
                    const vizType = document.getElementById('viz-type').value;
                    const opacity = parseFloat(document.getElementById('opacity-slider').value);
                    
                    // Get original sample index
                    const originalIndex = indices[sampleIndex];
                    const trueLabel = trueLabels[sampleIndex];
                    
                    // Create container for sample
                    const containersDiv = document.getElementById('sample-containers');
                    containersDiv.innerHTML = '';
                    
                    const sampleContainer = document.createElement('div');
                    sampleContainer.className = 'sample-container';
                    sampleContainer.innerHTML = `
                        <div class="sample-header">
                            Sample #${originalIndex} - True Label: ${trueLabel}
                        </div>
                        <div class="model-comparison" id="model-comparison-${sampleIndex}">
                        </div>
                    `;
                    containersDiv.appendChild(sampleContainer);
                    
                    // Add model results
                    const modelComparison = document.getElementById(`model-comparison-${sampleIndex}`);
                    
                    for (let i = 0; i < modelResults.length; i++) {
                        const result = modelResults[i];
                        const prediction = result.predictions[sampleIndex];
                        const probability = result.probabilities[sampleIndex];
                        const isCorrect = prediction === trueLabel;
                        
                        const modelResult = document.createElement('div');
                        modelResult.className = 'model-result';
                        modelResult.innerHTML = `
                            <div class="model-header">${result.name}</div>
                            <div class="prediction ${isCorrect ? 'correct' : 'incorrect'}">
                                Prediction: ${prediction} (${probability.toFixed(3)})
                            </div>
                            <div class="visualization-container" id="viz-container-${i}-${sampleIndex}">
                            </div>
                        `;
                        modelComparison.appendChild(modelResult);
                        
                        // Add visualization
                        const vizContainer = document.getElementById(`viz-container-${i}-${sampleIndex}`);
                        
                        // In a real implementation, you would include the actual images
                        // For this demonstration, we'll just show placeholders
                        if (vizType === 'separate') {
                            // Show original image and GradCAM separately
                            vizContainer.innerHTML = `
                                <div>Original Image:</div>
                                <div>[Original image would be shown here]</div>
                                <div>GradCAM:</div>
                                <div>[GradCAM visualization would be shown here]</div>
                            `;
                            
                            // If we have GradCAM data, render it
                            if (result.gradcams) {
                                const gradcam = result.gradcams[sampleIndex];
                                const gradcamURL = createImageDataURL(gradcam, gradcam.length, gradcam[0].length, 'jet');
                                
                                vizContainer.innerHTML = `
                                    <div>Original Image:</div>
                                    <div>[Original image would be shown here]</div>
                                    <div>GradCAM:</div>
                                    <img src="${gradcamURL}" class="gradcam-image">
                                `;
                            }
                        } else {
                            // Show overlay
                            vizContainer.innerHTML = `
                                <div class="image-overlay">
                                    <div>[Original image with GradCAM overlay would be shown here]</div>
                                </div>
                            `;
                            
                            // If we have GradCAM data, render it as an overlay
                            if (result.gradcams) {
                                const gradcam = result.gradcams[sampleIndex];
                                const gradcamURL = createImageDataURL(gradcam, gradcam.length, gradcam[0].length, 'jet');
                                
                                vizContainer.innerHTML = `
                                    <div class="image-overlay">
                                        <div>[Original image with GradCAM overlay would be shown here]</div>
                                        <img src="${gradcamURL}" class="overlay-image" style="opacity: ${opacity};">
                                    </div>
                                `;
                            }
                        }
                    }
                }
                
                // Function to update opacity
                function updateOpacity(value) {
                    document.getElementById('opacity-value').innerText = value;
                    const overlayImages = document.querySelectorAll('.overlay-image');
                    overlayImages.forEach(img => {
                        img.style.opacity = value;
                    });
                }
                
                // Initialize visualization when page loads
                document.addEventListener('DOMContentLoaded', function() {
                    updateVisualization();
                });
            </script>
        </body>
        </html>
        """

        return html_content