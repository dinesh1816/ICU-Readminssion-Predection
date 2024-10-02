from flask import Flask, request, jsonify, render_template
import joblib


app = Flask(__name__)

# Load the best model (assuming it includes the preprocessor)
model = joblib.load('model/best_model.pkl')

@app.route('/')
def index():
    # Serve the HTML interface (ensure 'index.html' is properly set up in the 'templates' directory)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the POST request
        data = request.get_json(force=True)
        features = [
            float(data.get('hematocrit', 0)),  # Defaulting to 0 if the key does not exist
            float(data.get('neutrophils', 0)),
            float(data.get('sodium', 0)),
            float(data.get('glucose', 0)),
            float(data.get('bloodureanitro', 0)),
            float(data.get('creatinine', 0)),
            float(data.get('bmi', 0)),
            float(data.get('pulse', 0)),
            float(data.get('respiration', 0)),
            float(data.get('lengthofstay', 0))
        ]

        # Transform the data using the loaded model's preprocessor and make predictions
        prediction = model.predict([features])
        probability = model.predict_proba([features])[0, 1]  # Probability of the positive class

        # Return prediction and probability as JSON
        return jsonify({'prediction': int(prediction[0]), 'probability': float(probability)})
    except Exception as e:
        # Return error message if something goes wrong
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Turn off debug in production