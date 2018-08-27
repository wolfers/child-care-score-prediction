from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestRegressor
import sys
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
from DataCleaning import CleanClass, CleanErs

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('index.html')

@app.route('/ers')
def ers():
    ers_html = create_html('ers')
    return render_template('ers.html', value=ers_html)

@app.route('/class')
def class_page():
    class_html = create_html('class')
    return render_template('class.html', value=class_html)

@app.route('/predictfakeERS', methods=["POST"])
def predictfakeERS():
    df = pd.read_csv('data/fake_ers_X.csv', sep='|')
    df_times = create_times(df)
    predictions = ers_model.predict(df_times)
    create_graph(df_times['time_delta'].values, predictions, 'static/images/demo/ers_graph.jpg')
    return jsonify({'test_type': 'ers'})

@app.route('/predictfakeCLASS', methods=["POST"])
def predictfakeCLASS():
    df = pd.read_csv('data/fake_class_X.csv', sep='|')
    df_times = create_times(df)
    predictions = class_model.predict(df_times)
    create_graph(df_times['time_delta'].values, predictions, 'static/images/demo/class_graph.jpg')
    return jsonify({'test_type': 'class'})

@app.route('/predictERS', methods=["POST"])
def predictERS():
    df = create_df_from_form(request.json)
    df_cleaned = ers_transformer(df)
    df_times = create_times(df_cleaned)
    predictions = ers_model.predict(df_times)
    create_graph(df_times['time_delta'].values, predictions, 'static/images/prediction-graphs/ers_graph.jpg')
    return jsonify({'test_type': 'ers'})

@app.route('/predictCLASS', methods=["post"])
def predictCLASS():
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

def create_html(test_type):
    final_html = ''
    html_string = '<label>{0}</label><input type="text" name="{0}">'
    if test_type == "ers":
        transformer = ers_transformer
    else:
        transformer = class_transformer
    for score_col in transformer.score_col_names:
        final_html += (html_string.format(score_col))
    return final_html

def create_df_from_form(json_form):
    to_be_df = []
    for form_input in json_form:
        to_be_df.append({form_input['name']: form_input['value']})
    return pd.DataFrame(to_be_df)

if __name__ == '__main__':
    with open('static/ers_model.pkl', 'rb') as f:
        ers_model = pickle.load(f)
    with open('static/class_model.pkl', 'rb') as f:
        class_model = pickle.load(f)
    with open('static/ers_trans_model.pkl', 'rb') as f:
        ers_transformer = pickle.load(f)
    with open('static/class_trans_model.pkl', 'rb') as f:
        class_transformer = pickle.load(f)
    app.run(host='0.0.0.0', port=8080, debug=True)