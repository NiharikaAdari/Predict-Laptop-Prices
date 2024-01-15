from flask import Flask, request, jsonify, render_template
import util
from sklearn.base import BaseEstimator,TransformerMixin
import numpy as np
app = Flask(__name__, template_folder='../client/templates',  static_url_path='/static')

class LogScaling(BaseEstimator, TransformerMixin):

    def fit(self, X):
        return self   

    def transform(self, X):
        return np.log1p(X)
    
@app.route('/')
def home():
    return render_template('app.html')

@app.route('/predict_laptop_price', methods=['POST'])
def predict_laptop_price():
    try:
        data = request.json  # Use request.json to access JSON data

        if data is None:
            print("No JSON data received")
            return jsonify({'error': 'No JSON data received'}), 400

        brand = str(data.get('brand'))
        processor_brand = str(data.get('processor_brand'))
        processor_name = str(data.get('processor_name'))
        processor_gnrtn = str(data.get('processor_gnrtn'))
        ram_gb = str(data.get('ram_gb'))
        ram_type = str(data.get('ram_type'))
        ssd = str(data.get('ssd'))
        hdd = str(data.get('hdd'))
        os = str(data.get('os'))
        os_bit = str(data.get('os_bit'))
        graphic_card_gb = str(data.get('graphic_card_gb'))
        weight = str(data.get('weight'))
        warranty = str(data.get('warranty'))
        touchscreen = str(data.get('touchscreen'))
        msoffice = str(data.get('msoffice'))
        rating = str(data.get('rating'))
        # number_of_reviews = int(data.get('number_of_reviews', 0))
        # number_of_ratings = int(data.get('number_of_ratings', 0))

        estimated_price = util.predict_price(
           brand = brand, processor_brand = processor_brand, processor_name = processor_name, processor_gnrtn = processor_gnrtn, ram_gb = ram_gb, ram_type = ram_type, ssd = ssd, hdd = hdd, os = os, os_bit = os_bit, graphic_card_gb = graphic_card_gb, weight = weight, warranty = warranty, Touchscreen = touchscreen, msoffice = msoffice, rating = rating)

        print("Estimated Price:", estimated_price * 0.012) 

        response = jsonify({
            'estimated_price': estimated_price
        })
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == "__main__":
    util.load_saved_artifacts()
    print("Starting Python Flask Server for Laptop Price Prediction...")
    app.run()
