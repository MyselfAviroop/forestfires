from flask import Flask, render_template, request
import numpy as np
import pickle

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    result = None
    if request.method == 'POST':
        # Get data from form
        temperature = float(request.form.get('temperature'))
        relative_humidity = float(request.form.get('relative_humidity'))
        wind_speed = float(request.form.get('wind_speed'))
        rain = float(request.form.get('rain'))
        ffmc = float(request.form.get('ffmc'))
        dmc = float(request.form.get('dmc'))
        dc = float(request.form.get('dc'))
        isi = float(request.form.get('isi'))
        region = float(request.form.get('region'))
        classes = float(request.form.get('classes'))

        # Scale and predict
        # Only include features used in training the scaler/model
        new_data = standard_scaler.transform([[temperature, relative_humidity, wind_speed, rain,
                                       ffmc, dmc, dc, isi, region]])

        result = ridge_model.predict(new_data)[0]

    return render_template('home.html', result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
