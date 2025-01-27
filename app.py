from flask import Flask, request, jsonify
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")

# Define a route for predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get features from the request
    data = request.get_json()
    
    # Make a prediction using the model
    prediction = model.predict([data["features"]])
    
    # Return the prediction as JSON
    return jsonify({"prediction": prediction.tolist()})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
