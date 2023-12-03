# Imports
from flask import Flask, render_template, url_for, request
import numpy as np
from tensorflow.keras.models import load_model



app = Flask(__name__)












if __name__ == '__main__':
    app.run(debug=True)
