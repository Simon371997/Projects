from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Laden des gespeicherten Modells
model = load_model('model.h5')

# Pfad zum Testbild
image_path = '/Users/simon/Documents/Master/Sonstiges/Python/Projects/Wafermap_Project/App/model/data/ImageData/test/Center/center_48.jpg'

# Bild für die Vorhersage laden und vorverarbeiten
img = image.load_img(image_path, target_size=(64, 65), color_mode='rgba')  # Zielgröße und Farbmodus anpassen
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalisierung

# Vorhersage machen
prediction = model.predict(img_array)

# Index des vorhergesagten Labels
predicted_class = np.argmax(prediction)

print(f"Vorhersage: {predicted_class}")
print(f"Konfidenz: {prediction[0][predicted_class]}")
