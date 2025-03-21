from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved model and scaler
model_filename = 'iris_model.pkl'
with open(model_filename, 'rb') as file:
    saved_objects = pickle.load(file)

model = saved_objects['model']
scaler = saved_objects['scaler']

@app.route('/')
def home():
    # Render the homepage with the prediction form
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects a JSON payload with feature values.
    Example:
      {
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
      }
    Returns the predicted iris species.
    """
    try:
        # Retrieve JSON data from the request
        data = request.get_json(force=True)
        
        feature_df = pd.DataFrame([data])
        
        # Preprocess the features using the saved scaler
        features_scaled = scaler.transform(feature_df)
        
        # Make the prediction using the loaded model
        prediction = model.predict(features_scaled)
        
        # Return the predicted species as JSON
        return jsonify({'predicted_species': prediction[0]})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Flask app runs on the default port (5000)
    app.run(debug=True)
