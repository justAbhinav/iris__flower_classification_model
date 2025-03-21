# Iris Flower Classification Project

[![Python Version](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
    - [Training and Evaluation](#training-and-evaluation)
    - [Testing](#testing)
    - [Web Interface](#web-interface)
    - [API Deployment](#api-deployment)
- [API Example](#api-example)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This project demonstrates an end-to-end machine learning pipeline for classifying Iris flowers into three species. It covers the following:
- **Data Loading and Exploratory Data Analysis (EDA):** Understanding the Iris dataset.
- **Data Preprocessing:** Cleaning, splitting, and scaling data.
- **Model Training and Evaluation:** Building, comparing, and tuning various classifiers.
- **Model Interpretability:** Using SHAP to explain model predictions.
- **Deployment:** Serving the model as a RESTful API using Flask.

## Features
- **Interactive Data Visualization:** Histograms, pairplots, and interactive Plotly charts.
- **Multiple Machine Learning Models:** Includes Random Forest, Logistic Regression, SVM, KNN, Gradient Boosting, and ensemble methods.
- **Hyperparameter Tuning:** GridSearchCV and RandomizedSearchCV for optimal model parameters.
- **Model Interpretability:** SHAP-based visualizations to explain feature contributions.
- **RESTful API Deployment:** Flask-based API to serve predictions.

## Project Structure
```
Iris-Classification/
├── notebooks/
│   ├── Iris_Training.ipynb  # Main notebook for training and evaluation
│   └── Iris_Testing.ipynb   # Notebook for testing the saved model
├── src/
│   ├── api.py             # Flask API for model deployment
│   ├── static/           # Static files for web interface (CSS, JS, images)
│   └── templates/        # HTML templates for web interface
├── requirements.txt       # List of required packages
├── README.md              # Project overview and instructions
```

## Installation
1. **Clone the repository:**
     ```bash
     git clone https://github.com/justAbhinav/Iris-Classification.git
     cd Iris-Classification
     ```

2. **Create and activate a virtual environment (recommended):**
    
    on Unix or MacOS:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
    
    on Windows:
     ```bash
    python -m venv venv
     venv\Scripts\activate 
     ```

3. **Install dependencies:**
     ```bash
     pip install -r requirements.txt
     ```

## Usage

### Training and Evaluation
1. Open the `notebooks/Iris_Training.ipynb` notebook in Jupyter.
2. Run each cell sequentially to:
     - Load and explore the Iris dataset.
     - Preprocess the data and perform EDA.
     - Train various models and compare their performance.
     - Use SHAP for model interpretability.
     - Save the best model and scaler to disk.

### Testing
1. Open the `notebooks/Iris_Testing.ipynb` notebook.
2. This notebook loads the saved model and scaler, runs predictions on the test data, and evaluates performance using accuracy, confusion matrices, and classification reports.

### Web Interface
1. The project includes a user-friendly web interface for making predictions.
2. To run the web interface locally:
     ```bash
     python src/api.py
     ```
3. Open your web browser and navigate to `http://127.0.0.1:5000`
4. The web interface provides:
   - An intuitive form to input Iris flower measurements
   - Real-time predictions
   - Visual feedback and results display
   - Responsive design for all devices

### API Deployment
1. The deployment code is located in the `src/api.py` file.
2. To run the API locally:
     ```bash
     python src/api.py
     ```
3. The API will be available at `http://127.0.0.1:5000/predict`. Where you can use Postman or curl to send a POST request with the input features.
4. The API uses the saved model and scaler to make predictions.

## API Example
You can test the API using Postman or curl. Below is an example using curl:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }'
```

Expected JSON response:
```json
{
    "predicted_species": "setosa"
}
```

## License
This project is licensed under the MIT License.

## Acknowledgements
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- Special thanks to the contributors and community that have inspired this project.
