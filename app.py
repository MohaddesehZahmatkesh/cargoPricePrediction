from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import math
import os
import pandas as pd
import warnings
import numpy as np
from datetime import datetime
from waitress import serve

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://api.1000bar.ir"}})

# Set your secret API key
SECRET_API_KEY = "iw3WCklmQ6rJVnKLtTAYRn5A7bSovK"  # Replace with your secret key

# Get the current working directory
current_dir = os.path.dirname(os.path.realpath(__file__))

model_path = os.path.join(current_dir, 'best_model.pkl')
model = pickle.load(open(model_path, 'rb'))

def extract_datetime_features(datetime_str):
    # Convert string to datetime object
    dt_object = pd.to_datetime(datetime_str)

    # Extract year, month, day, hour, and minute
    year = dt_object.year
    month = dt_object.month
    day = dt_object.day
    hour = dt_object.hour
    minute = dt_object.minute

    return year, month, day, hour, minute

def verify_api_key(api_key):
    # Verify the API key against the secret key
    return api_key == SECRET_API_KEY

@app.route('/predict', methods=['POST'])
def predict():
    
    # Check for the presence of the 'ApiKey' header in the request
    api_key = request.headers.get('ApiKey')
    
    # Verify the API key
    if not verify_api_key(api_key):
        return jsonify({'error': 'Invalid API key'}), 401  # Unauthorized

    data = request.get_json()

    # Extract variables from the request JSON
    product_id = data.get('ProductId')
    source_id = data.get('SourceId')
    destination_id = data.get('DestinationId')
    tonnage = data.get('Tonnage')

    # Validate input data
    if any(v is None for v in [product_id, source_id, destination_id, tonnage]):
        return jsonify({'error': 'Missing required fields'}), 400  # Bad request
    
    # Get the current date and time
    current_datetime = datetime.now()

    # Extract datetime features
    year, month, day, hour, minute = extract_datetime_features(current_datetime)
    inputs=(product_id , source_id, destination_id, tonnage, year, month, day, hour, minute)
    input_data_as_numpy_array = np.asarray(inputs)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Make a prediction using the machine learning model
    predictions = model.predict(input_data_reshaped)

    # Extract the first element from the array and round up
    prediction = math.ceil(predictions[0])

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Specify the port to run the Flask app on (e.g., port 5090 for your case)
    port = 5090
    
    # Run the Flask app using Waitress, specifying the port
    serve(app, port=port)
