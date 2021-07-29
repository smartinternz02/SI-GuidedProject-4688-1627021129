from flask import Flask, render_template, request
app = Flask(__name__)

from tensorflow.keras.models import load_model
import joblib
# import numpy as np

model = load_model("VisaPredModel.h5")
ct = joblib.load("columntransform")

@app.route('/',methods=["GET"])

def index():
    return render_template("home.html")

@app.route('/',methods=["POST"])

def predict():
    soc = request.form["soc"]
    pos = request.form["pos"]
    wage = float(request.form["wage"])
    year = float(request.form["year"])
    data = [[soc,pos,wage,year]]
    #data = np.asarray(data).astype(np.float32)
    pred = model.predict(ct.transform(data))
    prediction = {0 : 'CERTIFIED' , 1: 'CERTIFIED-WITHDRAWN' , 2: 'DENIED' , 3:'INVALIDATED' ,4:'PENDING QUALITY AND COMPLIANCE REVIEW - UNASSIGNED' , 5:'REJECTED' , 6: 'WITHDRAWN'}
    return render_template("home.html",value=str(prediction[pred.argmax()]))

app.run(host="localhost",port="5000",debug=False)


