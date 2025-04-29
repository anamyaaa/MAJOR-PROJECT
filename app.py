
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline (which includes preprocessing and the model)
model = joblib.load("nids_pipeline.pkl")

@app.route("/")
def home():
    return "NIDS Model is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    # Get the incoming JSON data
    data = request.json

    # Convert it into a DataFrame (same format used during training)
    df = pd.DataFrame(data)

    # Make predictions using the model
    preds = model.predict(df)  # Model will automatically handle preprocessing if in pipeline
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
