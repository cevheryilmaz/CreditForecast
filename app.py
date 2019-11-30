# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 22:37:53 2019

@author: Cevher
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    if output>=1.0 and output<1.5:
        return jsonify('Kredi Verilebilir')
    elif output>=1.5:
       return jsonify('Kredi Verilemez')
    else:
       return jsonify(output)

if __name__ == "__main__":
    app.run(debug=False)

