from flask import Flask, render_template, request
import numpy as np
from keras import models
from keras import preprocessing

app = Flask(__name__)

# Load the pre-trained TensorFlow model
tf_model = models.load_model("./model/models/tensorflowCNN.h5")

@app.route("/")
def home():
    """
    Route for the home page.
    """
    return render_template("home.html")

@app.route("/get_prediction")
def get_prediction():
    """
    Route for the get_prediction page.
    """
    return render_template("get_prediction.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Route for making predictions. It accepts POST requests with an image, 
    makes a prediction using the pre-trained model, and returns the top 3 predictions.
    """
    if request.method == "POST":
        # Save the uploaded image
        uploaded_file = request.files["image"]
        image_path = "uploads/uploaded_image.jpg"
        uploaded_file.save(image_path)

        # Define the target classes
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

        # Make a prediction
        prediction = make_prediction(image_path) 
        dict_result = {}

        # Map the prediction results to the target classes
        for i, target_name in enumerate(target_names):
            dict_result[prediction[0][i]] = target_name

        # Get the top 3 predictions
        prediction = prediction[0]
        predictions = sorted(prediction, reverse=True)
        probabilities = predictions[:3]

        # Prepare the results for rendering
        probability_result = []
        class_result = []

        for i in range(3):
            probability_result.append((probabilities[i] * 100).round(2))
            class_result.append(dict_result[probabilities[i]])

        # Render the results
        return render_template(
            "predict.html",
            top3_classes=class_result,
            top3_probabilities=probability_result,
        )
    return None

def preprocess_data(image_path:str):
    """
    Preprocess the input image for prediction.
    """
    # Load the image and resize it
    img = preprocessing.image.load_img(
        image_path, target_size=(64, 65), color_mode="rgba"
    ) 

    # Convert the image to a numpy array
    img_array = preprocessing.image.img_to_array(img)

    # Add a new axis to match the model's input shape
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the pixel values
    img_array /= 255.0  

    return img_array

def make_prediction(input_data:str):
    """
    Make a prediction using the pre-trained model.
    """
    # Preprocess the input data
    input_array = preprocess_data(input_data)

    # Make a prediction
    result = tf_model.predict(input_array)  

    return result

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)