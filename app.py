import sys
import os
import glob
import re
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import math  
import pandas as pd
from flask_cors import cross_origin
from flask import send_file
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import joblib
from gevent.pywsgi import WSGIServer
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = r'uploads'
ALLOWED_EXTENSIONS = {'csv'}

STATIC_DIR = os.path.abspath('static')

truckLoad = joblib.load(open("approach2.2.pkl","rb"))
reciptUnits = joblib.load(open("approach3.1.pkl","rb"))
salesUnit = joblib.load(open("approach1.1.pkl","rb"))



@app.route("/")
def home():
    return render_template("index.html")

@app.route("/pred")
def prediction():
    return render_template("prediction.html")

@app.route("/pred2")
def prediction2():
    return render_template("prediction2.html")

@app.route("/pred3")
def prediction3():
    return render_template("prediction3.html")


@app.route('/download') 
def download():
    return send_file("Forecast.csv",
                     mimetype='text/csv',
                      attachment_filename='forecastResult.csv',
                     as_attachment=True,
                     cache_timeout=0)

@app.route("/pred", methods=['GET', 'POST'])
def upload_file():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)    

        df = pd.read_csv(file_path)
        X = df.copy()
        test = truckLoad.predict(X)
        data = {'Inbound Truckloads':test} 
        df_sub = pd.DataFrame(data = data)
        result = pd.concat([df, df_sub], axis=1)
        result.head(5)
        result.drop(['SALES_UNITS'],1,inplace = True)

        result.to_csv(r"Forecast.csv",index=False)



    return render_template("result.html")

@app.route("/pred2", methods=['GET', 'POST'])
def upload_file1():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)    

        df = pd.read_csv(file_path)
        X = df.copy()
        test = reciptUnits.predict(X)
        data = {'receiptUnits':test**2}  
        df_sub = pd.DataFrame(data = data)
        result = pd.concat([df, df_sub], axis=1)
        result.head(5)

        result.to_csv(r"Forecast.csv",index=False)

    return render_template("result.html")


@app.route("/pred3", methods=['GET', 'POST'])
def upload_file2():

    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)    

        df = pd.read_csv(file_path)
        X = df.copy()
        test = salesUnit.predict(X)
        data = {'SALES_UNITS':test} 
        df_sub = pd.DataFrame(data = data)
        result = pd.concat([df, df_sub], axis=1)
        result.head(5)
        result.to_csv(r"Forecast.csv",index=False)
    
    return render_template("result.html")


@app.route("/aboutus")
def aboutus():
    return render_template("about.html")

@app.route("/res")
def result():
    return render_template("result.html")


if __name__ == "__main__":
    app.run(debug=False)