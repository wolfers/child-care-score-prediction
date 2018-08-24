from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import requests

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predictERS', methods=["POST"])
def predictERS():
    df = pd.read_csv('data/fake_ers_X.csv')
    df['time_delta'] = 150
    prediction = ers_model.predict(df)
    return jsonify({'prediction': prediction, 'test_type': 'ers'})

@app.route('/predictCLASS', methods=["POST"])
def predictCLASS():
    df = pd.read_csv('data/fake_class_X.csv')
    df['time_delta'] = 150
    prediction = class_model.predict()
    return jsonify({'prediction': prediction, 'test_type': 'class'})

if __name__ == '__main__':
    #load pickled models for use in prediction
    with open('static/ers_model.pkl', 'rb') as f:
        ers_model = pickle.load(f)
    with open('static/class_model.pkl', 'rb') as f:
        class_model = pickle.load(f)
    app.run(host='0.0.0.0', port=8080, debug=True)