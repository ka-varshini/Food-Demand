import pandas as pd
import numpy as np
import pickle
import os
import logging
from flask import Flask, request, render_template, jsonify

app = Flask(__name__, template_folder="templates")
app.config['SECRET_KEY'] = '2ece243aa5bfad295dca55d8b38cdbcd'

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/', methods=['GET'])
def index():
    return render_template('index1.html')

@app.route('/home', methods=['GET'])
def about():
    return render_template('index1.html')

@app.route('/pred', methods=['GET'])
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        logging.info("[INFO] Loading model...")
        model_path = "foodemand1.pkl"
        if not os.path.exists(model_path):
            logging.error(f"Model file {model_path} not found.")
            return jsonify({"error": "Model file not found."})
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if request.method == 'POST':
            y = request.form.values()
            input_features = []
            logging.debug(f"Form values: {y}")
            for x in y:
                try:
                    logging.debug(f"Processing value: {x}")
                    input_features.append(float(x))
                except ValueError:
                    logging.error(f"Invalid input value: {x}")
                    return jsonify({"error": f"Invalid input value: {x}"})

            logging.debug(f"Input features: {input_features}")
            if len(input_features) == 0:
                logging.error("No valid input features provided.")
                return jsonify({"error": "No valid input features provided."})

            features_value = [np.array(input_features)]
            logging.debug(f"Features value array: {features_value}")

            prediction = model.predict(features_value)
            output = prediction[0]
            logging.debug(f"Prediction output: {output}")
            return jsonify({"prediction": float(output)})

        return render_template('upload.html')

    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": "An error occurred during prediction."})
    
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    app.run(debug=False)
