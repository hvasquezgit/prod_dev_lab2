#from crypt import methods
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from feature_engine.imputation import AddMissingIndicator, MeanMedianImputer, CategoricalImputer
from datetime import datetime
from termcolor import colored

app = Flask(__name__)

modelo_entrenado_prod = joblib.load('C:/Users/hevas/titanic_pipeline.pkl')

FEATURES = joblib.load('C:/Users/hevas/FEATURES.pkl')

def generatLog(message, logType):
    f = open("logsData.log","a")
    message = message + datetime.today().strftime('%Y-%m-%d %H:%M:%S')+";" +"\n"
    f.write(message)
    if(logType==10):
        strColor = "yellow"
    elif(logType==30):
        strColor="red"
    elif(logType==90):
        strColor="green"
    print(colored(message,strColor))
    f.close()

@app.route("/predictOne", methods=['POST'])
def predictOne():
    data = request.get_json()
    dataframe = pd.json_normalize(data)

    #informar sobre transformación de JSON.
    #f = open("logsData.log","a")
    #logStr = "OX10-INFO - JSON transformación exitosa "
    #generatLog(logStr,10)
    #f.write(logStr)
    #print(colored(logStr,"yellow"))
    #f.close()
    #dataframe = pd.DataFrame.from_dict(data, orient="index")
    #dataframe = dataframe.T
    ids = dataframe['PassengerId']
    dataframe = dataframe[FEATURES]
    #prediccion
    try:
        nomr_preds = modelo_entrenado_prod.predict(dataframe)
        outPredict = np.exp(nomr_preds)
        
        out = {}
        for index,item in enumerate(outPredict):
            out[str(ids[index])] = round(item,2)
        logStr = "OX90-INFO - Exito-  se genero una prediccion exitosa -"
        generatLog(logStr,90)
        print(out)
        return jsonify(out)
    except ValueError:
        logStr = "OX30-INFO - PredictError-  se genero un problema en la prediccion -"
        generatLog(logStr,30)
        return jsonify({'mensaje: ':logStr})
    #return jsonify({'Prediccion ':str(outPredict)})