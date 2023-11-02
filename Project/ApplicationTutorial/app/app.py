from flask import Flask, render_template,request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
import plotly.express as px
import plotly.graph_objs as go
import uuid



app = Flask(__name__)

@app.route("/", methods=["GET", "POST"] )

def hello_world():
    request_type_str=request.method
    if request_type_str=='GET':
        path = "static/static_image.svg"
        return render_template('index.html', href=path)
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static_image/"+random_string+'.svg'
        # model = load('model.joblib')
        # np_arr = floats_string_to_np_arr(text)
        # make_picture('AgesAndHeight.pkl', model, np_arr, path)
    
        # load dataset
        gr8_wq= pd.read_csv("dataset/winequality-red.csv",header=0)

        # split the data
        gr8_X = gr8_wq.drop(['quality'], axis = 1)
        gr8_y = gr8_wq['quality']
        gr8_X_train, gr8_X_test, gr8_y_train, gr8_y_test = train_test_split(gr8_X, gr8_y, stratify=gr8_y,test_size=0.20)

        # load the model with details
        np_arr = floatsome_to_np_array(text).reshape(1,-1)
        pkl_filename='TrainedModel/StackedPickle.pkl'
        with open(pkl_filename, 'rb') as file:
            pickle_model=pickle.load(file)
        plot_graphs(model=pickle_model, new_input_arr=np_arr, output_file=path)
        return render_template('index.html', href=path)

# plot for the model with output
def plot_graphs(model, new_input_arr, output_file_static, output_file_interactive):
    df = gr8_wq
    fig = px.scatter_3d(df, x='alcohol', y="volatile acidity", z='quality', color='quality',color_continuous_scale='bluered')
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend=dict(
    yanchor="bottom",
    xanchor="right",
))
    new_preds = model.predict(new_input_arr)
    alcohol_input = np.array(new_input_arr[0][10])
    volatile_acidity_input  = np.array(new_input_arr[0][1])

    # print wine features and wine quality prediction output
    print("Alcohol: ", alcohol_input)
    print("Volatile Acidity: ",  volatile_acidity_input)
    print("Predicted Wine Quality:", new_preds[0])
    
    # graph the wine quality and predicted quality based on the input features
    fig.add_trace(
        go.Scatter3d(
            x=alcohol_input,
            y=volatile_acidity_input,
            z=new_preds,
            mode='markers', name = 'Predicted Output',
            marker=dict(
                color='#FFCC00', size=13),
            line=dict(color='#FFCC00', width=1)
            ))
    fig.update_layout(height=1000, width=1800, 
    title={
        'text': "Wine features and wine quality",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    font=dict(
        size=15
    )
    )
    print('\n' + "Wine Quality vs. Predict Wine Quality (yellow) Graph For The Input Wine Features Is Displayed Below")
    fig.write_image(output_file_static, engine='kaleido')
    fig.write_html(output_file_interactive)
    fig.show()

    # output feature importance - using Random Forest 
    gr8_X_train.columns
    model.estimators_[1].feature_importances_
    feat_df = pd.DataFrame({'Feature':gr8_X_train.columns.array,'Importance':model.estimators_[1].feature_importances_})

    # graph feature important
    plt.figure(figsize=(20,8))
    sns.barplot(y=feat_df['Importance'], x=feat_df['Feature'], palette='flare').set(title='Features Importance to predict Wine Quality')
    plt.show()



# Define input function
def floatsome_to_np_array(floats_str):

    floats = np.array([float(x) for x in floats_str.split(',')])
    return floats.reshape(len(floats),1)

floatsome_to_np_array("1, 222, 3")