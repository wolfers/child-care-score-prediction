from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/ers')
def ers():
    return render_template('ers.html')

@app.route('/class')
def class_page():
    return render_template('class.html')

@app.route('/predictERS', methods=["POST"])
def predictERS():
    df = pd.read_csv('data/fake_ers_X.csv', sep='|')
    df_times = create_times(df)
    predictions = ers_model.predict(df_times)
    create_graph(df_times['time_delta'].values, predictions, 'static/images/ers_graph.jpg')
    return jsonify({'test_type': 'ers'})

@app.route('/predictCLASS', methods=["POST"])
def predictCLASS():
    df = pd.read_csv('data/fake_class_X.csv', sep='|')
    df_times = create_times(df)
    predictions = class_model.predict(df_times)
    create_graph(df_times['time_delta'].values, predictions, 'static/images/class_graph.jpg')
    return jsonify({'test_type': 'class'})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

def create_times(df):
    row_list = []
    for time_delta in range(200, 1500, 10):
        temp_x = df.to_dict('records')[0]
        temp_x['time_delta'] = time_delta
        row_list.append(temp_x)
    return pd.DataFrame(row_list)

def create_graph(times, data, filename):
    _, ax = plt.subplots(figsize=(16,7))
    ax.plot(times, data)
    plt.xlabel('Days from Baseline')
    plt.ylabel('Average Score')
    plt.savefig(filename)

if __name__ == '__main__':
    #load pickled models for use in prediction
    with open('static/ers_model.pkl', 'rb') as f:
        ers_model = pickle.load(f)
    with open('static/class_model.pkl', 'rb') as f:
        class_model = pickle.load(f)
    app.run(host='0.0.0.0', port=8080, debug=True)