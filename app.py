
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("nids_pipeline.pkl")

@app.route("/")
def home():
    return "NIDS Model is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    preds = model.predict(df)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
