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
    prediction = ers_model.predict()
    return prediction

@app.route('/predictCLASS', methods=["POST"])
def predictCLASS():
    prediction = class_model.predict()
    return prediction

if __name__ == '__main__':
    #load pickled models for use in prediction
    with open('static/ers_model.pkl', 'rb') as f:
        ers_model = pickle.load(f)
    with open('static/class_model.pkl', 'rb') as f:
        class_model = pickle.load(f)
    app.run(host='0.0.0.0', port=8080, debug=True)