from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_boston

app = Flask(__name__)

@app.route("/")

def hello_world():
    request_type_str=request.method
    if request_type_str=='GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        # random_string = uuid.uuid4().hex
        # path = "static/"+random_string+'.svg'
        # model = load('model.joblib')
        # np_arr = floats_string_to_np_arr(text)
        # make_picture('AgesAndHeight.pkl', model, np_arr, path)

    return render_template("index.html", href=path) 
#    return "<p>Hello, World!</p>"