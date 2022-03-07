
# Import all packages and libraries
import pandas as pd
import numpy as np
from flask import Flask, render_template, url_for, request
import pickle
import math
import base64
import joblib
from lightgbm import LGBMClassifier


dataframe = pd.read_csv(('clientsdata.csv'), encoding ='utf-8')
all_id_client = list(dataframe['SK_ID_CURR'].unique())
model = joblib.load('model/model.jol')
        
seuil = 0.6

app= Flask(__name__)


@app.route('/')
def home():
    return "Prédiction rapide de l'acceptation ou non d'un prêt pour l'entreprise 'Prêt à dépenser' "




@app.route('/predict', methods = ['GET'])
def predict():
    
    '''
    For rendering results on HTML GUI
    '''

    ID = request.args.get('id_client')
    ID = int(ID)
    if ID not in all_id_client:
        prediction="Ce client n'est pas répertorié"
    else :

	

        X = dataframe[dataframe['SK_ID_CURR'] == ID]
        X = X.drop(['TARGET'], axis=1)

        


        probability_default_payment = model.predict(X)[:, 1]
        if probability_default_payment >= seuil:
            prediction = "Prêt NON Accordé"
        else:
            prediction = "Prêt Accordé"

    return str(prediction)

# Define endpoint for flask
app.add_url_rule('/predict', 'predict', predict)


# Run app.
# Note : comment this line if you want to deploy on heroku
app.run()
app.run(debug=True)
