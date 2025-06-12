from flask import Flask, render_template, request
import joblib
import pandas as pd
# import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('decisionTree.pkl')

# Define the feature columns used during training
# NOTE: You should replace this list with the actual columns used when training the model
feature_columns = ['month', 'origin', 'departure', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4',
                   'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7']  # Example

# Define airport mapping
airport_map = {
    "ATL": 1,
    "DTW": 2,
    "SEA": 3,
    "MSP": 4,
    "JFK": 5
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        month = float(request.form['month'])
        day = int(request.form['day'])
        quarter = int(request.form['quarter'])
        origin = airport_map.get(request.form['origin'], 0)
        departure = airport_map.get(request.form['departure'], 0)

        # Prepare DataFrame
        user_data = pd.DataFrame({
            'month': [month],
            'day': [day],
            'quarter': [quarter],
            'origin': [origin],
            'departure': [departure]
        })

        # One-hot encode 'day' and 'quarter'
        user_data_encoded = pd.get_dummies(user_data, columns=['quarter', 'day'])

        # Reindex to match training columns
        X = user_data_encoded.reindex(columns=feature_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(X)[0]

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
