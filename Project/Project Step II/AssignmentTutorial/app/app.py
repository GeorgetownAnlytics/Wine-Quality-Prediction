from flask import Flask, render_template,request
import numpy as np
import pandas as pd
from utils import *

app = Flask(__name__)

# @app.route("/")
# def hello_world():
#     request_type_str=request.method
#     if request_type_str=='GET':
#         return render_template('index.html', href='static/base_pic.png')
#     else:
#         text = request.form['text']
#         # random_string = uuid.uuid4().hex
#         # path = "static/"+random_string+'.svg'
#         # model = load('model.joblib')
#         # np_arr = floats_string_to_np_arr(text)
#         # make_picture('AgesAndHeight.pkl', model, np_arr, path)
#         return render_template("index.html", href=path) 
# #    return "<p>Hello, World!</p>"

@app.route('/', methods = ["GET","POST"])
def form():
    if request.method == "GET": # get data from server
        return render_template("index.html")
    else:
        fixed_acidity = float(request.form["fixed_acidity"])
        volatile_acidity = float(request.form["volatile_acidity"])
        citric_acid = float(request.form["citric_acid"])
        residual_sugar = float(request.form["residual_sugar"])
        chlorides = float(request.form["chlorides"])
        free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
        total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
        density = float(request.form["density"])
        pH = float(request.form["pH"])
        sulphates = float(request.form["sulphates"])
        alcohol = float(request.form["alcohol"])

        data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,chlorides,
                  free_sulfur_dioxide, total_sulfur_dioxide, density,pH, sulphates, alcohol]]

        model = load_model()
        y_pred = model.predict(data)[0]   # [val]
        y_pred = get_label(y_pred)


        return render_template("index.html",pred = y_pred)

        





if __name__ == "__main__":
    app.run(debug = True)    