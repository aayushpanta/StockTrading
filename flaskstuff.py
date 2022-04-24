# from cProfile import label
# from enum import unique
# from operator import index
from flask import Flask, render_template, redirect, request, session, flash
from flask_sqlalchemy import SQLAlchemy
import os
import json
from matplotlib.font_manager import json_dump
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime

#import gviz_api

from trading import training, prediction, graph, graph2


with open("config.json", 'r') as c:
    params = json.load(c)["params"]

local_server = True
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['upload_folder'] = params['upload_location']
app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# db schema
class traindb(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(12), nullable=True)
    modelname = db.Column(db.String(80), nullable=False, unique=True)
    algorithm = db.Column(db.String(120), nullable=False)
    dataset = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f"{self.date},{self.modelname},{self.algorithm},{self.dataset}"

models = os.listdir('static\saved models')
datas = os.listdir('static\dataset')


@app.route("/")
def home():
    return render_template('home.html', params=params)


@app.route("/train", methods=['POST', 'GET'])
def train():
    if (request.method == "POST"):      #runs after user clicks 'Train' button
        # get dataset name, algo etc
        modelname = request.form.get('modelname')
        algorithm = request.form.get('algorithm')
        dataset = request.form.get('dataset')
        start = request.form.get('startdate')
        end = request.form.get('enddate')
        modelname = modelname+'_'+algorithm
        # print(f'{modelname}+{algorithm}+{dataset}')
        entry = traindb(date=datetime.now(), modelname=modelname,
                        algorithm=algorithm, dataset=dataset)
        db.session.add(entry)
        db.session.commit()
        #flash("Training in process....Please Wait!!!", "primary")
        training(modelname, algorithm, dataset, start, end)
        flash("Training done !!!", "primary")
        # call training method and pass modelname, algoritm and dataset as arguments
        # train(modelname,algorithm,dataset)
        
        #return redirect('/renderPredictions/train:'+modelname+':'+dataset+':'+start+':'+end)
        return redirect('/getPredictions')

    return render_template('train.html', params=params, datas=datas)


@app.route("/getPredictions", methods=['POST', 'GET'])
def getPredictions():
    models = os.listdir('static\saved models')
    if (request.method == "POST"):      #runs after user clicks 'Predict' button
        modelname = request.form.get('modelname')
        dataset = request.form.get('dataset')
        start = request.form.get('startdate')
        end = request.form.get('enddate')
        # call prediction method
        # render graphs and result
        # todo: find the specific modelname and dataset and pass to prediction method and return graphs which should be displayed in web

        #item = traindb.query.filter_by(dataset=dataset).order_by(traindb.date.desc(
        #)).first_or_404(description='There is no data with modelname {} and dataset {}'.format(modelname, dataset))
        if(True): # cnange True to item
            flash("Predicting....Please Wait!!!", "Primary")
            env4, name = prediction(modelname, dataset, start, end)
            plt.figure(figsize=(12,6))
            graph(env4, name, modelname)
            plt.figure(2)
            graph2(env4, name)
            plt.show()
            # print(item.algorithm)
            # prediction(modelname,item[algorithm],dataset)
            #return redirect('/renderPredictions/prediction:'+modelname+':'+dataset+':'+start+':'+end)
            return redirect('/getPredictions')

        print(f'{item}')

    return render_template('prediction.html', params=params, datas=datas, models=models)

# if needed to reder graph in new page


@app.route("/renderPredictions/<string:mode>:<string:modelname>:<string:dataset>:<string:start>:<string:end>")
def renderPredictions(mode, modelname, dataset, start, end):
    col_list = ["Date", "Close", "High", "Low"]
    if (mode == 'prediction'):
        item = traindb.query.filter_by(dataset=dataset).order_by(traindb.date.desc()).first_or_404(
            description='There is no data with modelname {} and dataset {}'.format(modelname, dataset))
        dataset = item.dataset
        df = pd.read_csv('static/Banks/'+dataset, usecols=col_list)
        # print(df)
        datas = []
        labels = []
        values = []
        short = []
        long = []
        for l in df.Date:
            labels.append(l)
        for v in df.Close:
            values.append(v)
        for s in df.High:
            short.append(s)
        for l in df.Low:
            long.append(l)

        for i in range(1, len(values)):
            if (df.Date[i] >= start) & (df.Date[i] <= end):
                datas.append([labels[i], values[i], short[i], long[i]])

        # print(msg)
        # jsondata = json.dumps(datas[0:20])
        return render_template('graph.html', params=params, jsondata=datas, dataset=dataset.split('.')[0], modelname=modelname)
    df = pd.read_csv('static/dataset/'+dataset)
    datas = []
    labels = []
    values = []
    short = []
    long = []
    for l in df.Date:
        labels.append(l)
    for v in df.Close:
        values.append(v)
    for s in df.High:
        short.append(s)
    for l in df.Low:
        long.append(l)

    for i in range(1, len(values)):
        if (df.Date[i] >= start) & (df.Date[i] <= end):
            datas.append([labels[i], values[i], short[i], long[i]])
    return render_template('graph.html', params=params, jsondata=datas, bankname=dataset.split('.')[0], modelname=modelname)


@app.route("/team")
def team():
    return render_template('team.html', params=params)


if __name__ == '__main__':
    app.run(debug=True)
