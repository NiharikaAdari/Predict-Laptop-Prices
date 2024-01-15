from flask import Flask, request, jsonify, render_template
import util

app = Flask(__name__, template_folder='../client/templates')

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

        brand = data.get('brand')
        processor_brand = data.get('processor_brand')
        processor_name = data.get('processor_name')
        processor_gnrtn = data.get('processor_gnrtn')
        ram_gb = data.get('ram_gb')
        ram_type = data.get('ram_type')
        ssd = data.get('ssd')
        hdd = data.get('hdd')
        os = data.get('os')
        os_bit = data.get('os_bit')
        graphic_card_gb = data.get('graphic_card_gb')
        weight = data.get('weight')
        warranty = data.get('warranty')
        touchscreen = data.get('touchscreen')
        msoffice = data.get('msoffice')
        rating = data.get('rating')
        number_of_reviews = int(data.get('number_of_reviews', 0))
        number_of_ratings = int(data.get('number_of_ratings', 0))

        estimated_price = util.predict_price(
            brand, processor_brand, processor_name, processor_gnrtn, ram_gb, ram_type, ssd, hdd, os, os_bit,
            graphic_card_gb, weight, warranty, touchscreen, msoffice, rating, number_of_ratings, number_of_reviews)

        print("Estimated Price:", estimated_price[0])  # Added print statement

        response = jsonify({
            'estimated_price': estimated_price
        })
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': 'An error occurred'}), 500

if __name__ == "__main__":
    print("Starting Python Flask Server for Laptop Price Prediction...")
    app.run()
