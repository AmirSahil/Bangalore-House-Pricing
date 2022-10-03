from flask import Flask, render_template, request, app, jsonify
import json
import pickle
import numpy as np

app = Flask(__name__)

__locations = None
__data_columns = None
__model = None

def load_saved_artifacts():
    print("Loading Artifacts...")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/columns.json", 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open("./artifacts/bhp.pickle", 'rb') as f:
        __model = pickle.load(f)

def get_estimated_price(location, sqft, bhk, bath):
    load_saved_artifacts()
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    output = get_estimated_price(data['location'], data['sqft'], data['bhk'], data['bath'])
    return jsonify(output * 100000)

@app.route('/predict', methods=['GET','POST'])
def predict():
    data = [x for x in request.form.values()]
    request.form.get('location')
    print(data)
    response = jsonify(get_estimated_price(data[0], data[0], data[1], data[2]))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return render_template("home.html", prediction = response * 100000, sqft=data[0], bed=data[1], bath=data[2], loc=data[3].title())

if __name__ == "__main__":
    app.run(debug=True)