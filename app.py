from flask import Flask, request, render_template
import joblib
import pandas as pd
from datetime import datetime

app = Flask(__name__)
model = joblib.load('machine_failure_model.pkl')  # Make sure this file exists

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction')
def prediction_form():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        location = request.form['location']
        machine_age = int(request.form['machine_age'])
        operating_hours = int(request.form['operating_hours'])
        temperature = float(request.form['temperature'])
        pressure = float(request.form['pressure'])
        vibration = float(request.form['vibration'])
        last_maintenance_date = request.form['last_maintenance_date']

        last_maintenance_date = datetime.strptime(last_maintenance_date, '%Y-%m-%d')

        user_data = pd.DataFrame({
            'Location': [location],
            'Machine Age': [machine_age],
            'Operating Hours': [operating_hours],
            'Temperature': [temperature],
            'Pressure': [pressure],
            'Vibration': [vibration],
            'Last Maintenance Date': [last_maintenance_date]
        })

        prediction = model.predict(user_data)
        probability = model.predict_proba(user_data)

        return render_template('result.html',
                               prediction=prediction[0],
                               probability=probability[0][1])
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
