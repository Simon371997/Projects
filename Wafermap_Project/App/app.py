# Imports
from flask import Flask, render_template, url_for, request
import numpy as np
from keras.models import load_model
from keras.preprocessing import image 

# Initialize App
app = Flask(__name__)

# Load Model
model = load_model('model.h5')

# routes
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/get_prediction')
def get_prediction():
    return render_template('get_prediction.html')


@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        # Get the Imagge
        uploaded_file = request.files['image']
        # Save the Image
        image_path = 'uploads/uploaded_image.jpg'
        uploaded_file.save(image_path)

        # make prediction
        prediction = make_prediction(image_path)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]
        return render_template('predict.html', predicted_class=predicted_class, confidence=confidence)




# Functions
def preprocess_data(input_data):
    image_path = 'uploads/uploaded_image.jpg'

    # Load image and prepare for prediction
    img = image.load_img(image_path, target_size=(64, 65), color_mode='rgba')  # Zielgröße und Farbmodus anpassen
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


def make_prediction(input_data):
    input_array = preprocess_data(input_data)
    result = model.predict(input_array)
    return result








if __name__ == '__main__':
    app.run(debug=True)
# target_names=['Center', 'Donut', 'Edge-loc', 'Edge-ring', 'Loc', 'Near-Full', 'None', 'Random', 'Scratch']