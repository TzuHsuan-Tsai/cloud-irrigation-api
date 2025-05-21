# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("irrigation_model.pkl")  # 只載入一次

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temp = float(request.form['temperature'])
        humi = float(request.form['humidity'])
        X = np.array([[temp, humi]])
        pred = model.predict(X)
        return jsonify(round(pred[0], 2))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
