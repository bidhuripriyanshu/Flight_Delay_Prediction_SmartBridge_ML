from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Define the feature columns used during training
feature_columns = [
    'month', 'origin', 'departure', 'quarter_1', 'quarter_2', 'quarter_3', 'quarter_4',
    'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6', 'day_7'
]

# Define airport mapping
airport_map = {
    "ATL": 1,
    "DTW": 2,
    "SEA": 3,
    "MSP": 4,
    "JFK": 5
}

# Load the trained model safely
model = None
model_path = 'decisionTree.pkl'
if os.path.exists(model_path):
    try:
        model = joblib.load("decisionTree.pkl")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Model file '{model_path}' not found. Please ensure it exists and is compatible.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Please contact the administrator."

    try:
        # Get form data
        month = float(request.form.get('month', 0))
        day = int(request.form.get('day', 1))
        quarter = int(request.form.get('quarter', 1))
        origin = airport_map.get(request.form.get('origin', ''), 0)
        departure = airport_map.get(request.form.get('departure', ''), 0)

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
        return f"An error occurred during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
