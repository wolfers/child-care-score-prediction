from flask import Flask, request, render_template, jsonify
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import pickle
import requests
from src.DataCleaning2 import CleanErs, CleanClass
import src.ModelFunctions as mf

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


@app.route('/predictERS', methods=["POST"])
def predictERS():
    df = create_df_from_form(request.json)
    df_cleaned = ers_transformer.transform(df)
    top_list = mf.process_ccqb(models_ers, df_cleaned)
#   top_html = create_top_html(top_list)
    return jsonify({"test_type": "ers", "top": top_list})


@app.route('/predictCLASS', methods=["post"])
def predictCLASS():
    df = create_df_from_form(request.json)
    df_cleaned = class_transformer.transform(df)
    top_list = mf.process_ccqb(models_class, df_cleaned)
#    top_html = create_top_html(top_list)
    return jsonify({"test_type": "class", "top": top_list})


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


def create_html(test_type):
    final_html = '<div class="row">'
    html_string = '''
    <div class="col">
    <label for={0}>{0}</label>
    <input id={0} class="form-control" type="text" name="{0}">
    </div>
    '''
    if test_type == "ers":
        transformer = ers_transformer
    else:
        transformer = class_transformer
    for count, score_col in enumerate(transformer.score_col_names):
        if count % 3 == 0:
            final_html += '</div><div class="row">'
        final_html += (html_string.format(score_col))
    final_html += "</div>"
    return final_html


def create_df_from_form(json_form):
    to_be_df = {}
    for form_dict in json_form:
        if 'Score' in form_dict['name'] or 'Number' in form_dict['name'] or 'Observation Time' == form_dict['name']:
            if form_dict['value'] == '':
                to_be_df[form_dict['name']] = np.nan
            else:
                to_be_df[form_dict['name']] = float(form_dict['value'])
        else:
            to_be_df[form_dict['name']] = form_dict['value']
    return pd.DataFrame(to_be_df, index=[0])


def create_top_html(top_list):
    '''
    Does not work with jquery like I had hoped.
    A project for later
    '''
    html = ""
    html_empty = "<p>{0}</p>"
    for top in top_list:
        html += html_empty.format(top)
    return html


if __name__ == '__main__':
    models_ers = mf.load_pickled_models("ers")
    models_class = mf.load_pickled_models("class")
    ers_transformer, class_transformer = mf.load_pickled_transformers()
    app.run(host='0.0.0.0', port=8080, debug=True)