""" Run this programm to start the Web-App"""


# Imports
from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Initialize App
app = Flask(__name__)

# Load Model
tf_model = load_model("./model/models/tensorflowCNN.h5")


# routes
@app.route("/")
def home():
    """
    Home - Route
    """
    return render_template("home.html")


@app.route("/get_prediction")
def get_prediction():
    """
    Route to get the prediction
    """
    return render_template("get_prediction.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Route to display the prediction
    """
    if request.method == "POST":
        # Get the Imagge
        uploaded_file = request.files["image"]
        # Save the Image
        image_path = "uploads/uploaded_image.jpg"
        uploaded_file.save(image_path)

        # make prediction
        target_names = [
            "Center",
            "Donut",
            "Edge-loc",
            "Edge-ring",
            "Loc",
            "Near-Full",
            "None",
            "Random",
            "Scratch",
        ]
        prediction = make_prediction(image_path)  # result: [[0.03, 0.82, 0.05, .....]]
        dict_result = {}

        for i, target_name in enumerate(target_names):
            dict_result[prediction[0][i]] = target_name

        prediction = prediction[0]
        predictions = sorted(prediction, reverse=True)
        probabilities = predictions[:3]

        probability_result = []
        class_result = []

        for i in range(3):
            probability_result.append((probabilities[i] * 100).round(2))
            class_result.append(dict_result[probabilities[i]])

        return render_template(
            "predict.html",
            top3_classes=class_result,
            top3_probabilities=probability_result,
        )
    return None


# Functions
def preprocess_data(image_path:str):
    """
    Preprocesses an image for prediction.

    Parameters:
        image_path (str): The path to the input image file.

    Returns:
        numpy.ndarray: A preprocessed image array suitable for prediction.

    Raises:
        FileNotFoundError: If the specified image file is not found.
        ValueError: If the image cannot be loaded or preprocessed.
    """
    # Load image and prepare for prediction
    img = image.load_img(
        image_path, target_size=(64, 65), color_mode="rgba"
    )  # Zielgröße und Farbmodus anpassen
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array


def make_prediction(input_data:str):
    """
    Makes a prediction using a pre-trained TensorFlow model.

    Parameters:
        input_data (str): The path to the input image file.

    Returns:
        numpy.ndarray: The prediction result as an array.

    Raises:
        FileNotFoundError: If the specified image file is not found.
        ValueError: If there is an error in image preprocessing or model prediction.
    """
    input_array = preprocess_data(input_data)
    result = tf_model.predict(input_array)  # result: [[0.03, 0.82, 0.05, .....]]
    return result


if __name__ == "__main__":
    app.run(debug=True)
