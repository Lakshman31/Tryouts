import os
from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('random_forest_pickle.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    Fuel_Type_Diesel=0
    if request.method == 'POST':
        Year = int(request.form['Year'])
        Kms_Driven=int(request.form['Kms_Driven'])
        Kms_Driven2=np.log(Kms_Driven)
        Owner=(request.form['Owner'])
        if (Owner == 'Four and above'):
            Owner_fourplus= 1
            Owner_third = 0
            Owner_second = 0
            Owner_testdrive = 0
        if (Owner == 'Three'):
            Owner_fourplus= 0
            Owner_third = 1
            Owner_second = 0
            Owner_testdrive = 0
        if (Owner == 'Two'):
            Owner_fourplus= 0
            Owner_third = 0
            Owner_second = 1
            Owner_testdrive = 0
        if (Owner >= 'Test Drive Car'):
            Owner_fourplus= 0
            Owner_third = 0
            Owner_second = 0
            Owner_testdrive = 1
        else:
            Owner_fourplus= 0
            Owner_third = 0
            Owner_second = 0
            Owner_testdrive = 0
        Fuel_Type_Petrol=request.form['Fuel_Type_Petrol']
        if(Fuel_Type_Petrol=='Petrol'):
                Fuel_Type_Petrol=1
                Fuel_Type_Diesel=0
                Fuel_Type_Electric =0
                Fuel_Type_LPG = 0
        elif (Fuel_Type_Petrol=='CNG'):
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0
            Fuel_Type_Electric = 0
            Fuel_Type_LPG = 0
        elif (Fuel_Type_Petrol=='Electric'):
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0
            Fuel_Type_Electric = 1
            Fuel_Type_LPG = 0
        elif (Fuel_Type_Petrol=='LPG'):
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 0
            Fuel_Type_Electric = 0
            Fuel_Type_LPG = 1
        else:
            Fuel_Type_Petrol=0
            Fuel_Type_Diesel=1
            Fuel_Type_Electric = 0
            Fuel_Type_LPG = 0
        Year=2022-Year
        Seller_Type_Individual=request.form['Seller_Type_Individual']
        if(Seller_Type_Individual=='Individual'):
            Seller_Type_Individual=1
            Seller_Type_TDealer = 0
        elif (Seller_Type_Individual == 'Trustmark Dealer'):
            Seller_Type_Individual = 0
            Seller_Type_TDealer = 1
        else:
            Seller_Type_Individual=0
            Seller_Type_TDealer = 0
        Transmission_Mannual=request.form['Transmission_Mannual']
        if(Transmission_Mannual=='Mannual'):
            Transmission_Mannual=1
        else:
            Transmission_Mannual=0

        prediction=model.predict([[Kms_Driven2,Fuel_Type_Diesel,Fuel_Type_Electric,Fuel_Type_LPG,Fuel_Type_Petrol,Seller_Type_Individual,Seller_Type_TDealer,Transmission_Mannual,Owner_fourplus,Owner_second,Owner_testdrive,Owner_third,Year]])
        output=round(prediction[0],2)
        if output<0:
            return render_template('index.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index.html',prediction_text="You Can Sell The Car at {}".format(output))
    else:
        return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)