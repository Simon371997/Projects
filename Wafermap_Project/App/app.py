# Imports
from flask import Flask, render_template, url_for, request
import numpy as np
from keras.models import load_model

# Initialize App
app = Flask(__name__)

# Load Model
model = load_model('model.h5')

# routes
@app.route('/')
def home():
    return render_template('base.html')


@app.route('/predict', methods = ['POST'])
def predict():
    if request == 'POST':
        input_data = request.form['input_data']
        prediction = make_prediction(input_data)
        return render_template('predict.html')


def make_prediction(input_data):
    input_array = np.array([input_data])
    
    result = model.predict(input_array)
    return result










if __name__ == '__main__':
    app.run(debug=True)
