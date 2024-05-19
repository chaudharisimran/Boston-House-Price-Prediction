import pickle
import numpy as np
import pandas as pd
from flask import Flask,render_template,request,app,jsonify,url_for

app=Flask(__name__)

#Load the model
regmodel=pickle.load(open('house_price_regmodel.pkl','rb'))
scaler=pickle.load(open('scaler_data.pkl','rb'))
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("result.html",prediction_text="The House price prediction is: {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)
