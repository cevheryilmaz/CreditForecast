# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 23:12:09 2019

@author: Cevher
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'krediMiktari':1169, 'yas':67, 'evDurumu':1,'aldigi_kredi_sayi':2,'telefonDurumu':1})

print(r.json())