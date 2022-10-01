# import pickle
# import Flask, request, app, jsonify, url_for, render_template

# app = Flask(__name__)
# model = pickle.load(open('bhp.pkl','rb'))

# @app.route('/')
# def home():
#     return render_template('home.html')

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     total_sqft = float(request.form['total_sqft'])
#     location = request.form['location']
#     bhk = int(request.form['bhk'])
#     bath = int(request.form['bath'])


from flask import Flask, request, jsonify
import util
app = Flask(__name__)

@app.route('/home')
def home():
    return 'Home Page'

@app.route('/get_location_names')
def get_location_names():
    response = jsonify({
        'locations': util.get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])

    response = jsonify({
        'estimated_price': util.get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == "__main__":
    print("starting python flask server")
    app.run()