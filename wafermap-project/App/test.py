from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import torch

# Laden des gespeicherten Modells
model = torch.load('./model/models/torchCNN.pth')

# Pfad zum Testbild
image_path = '/Users/simon/Documents/Master/Sonstiges/Python/Projects/Wafermap_Project/App/model/data/ImageData/test/Center/center_49.jpg'

# Bild für die Vorhersage laden und vorverarbeiten
img = image.load_img(image_path, target_size=(64, 65), color_mode='rgba')  # Zielgröße und Farbmodus anpassen
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalisierung

# Vorhersage machen
prediction = model.predict(img_array)

# Index des vorhergesagten Labels
target_names=['Center', 'Donut', 'Edge-loc', 'Edge-ring', 'Loc', 'Near-Full', 'None', 'Random', 'Scratch']
dict_result = {}
for i in range(len(target_names)):
    dict_result[prediction[0][i]] = target_names[i]
print(dict_result)

prediction = prediction[0]
prediction = sorted(prediction, reverse=True)
probabilty = prediction[:3]
print(probabilty)

probability_result = []
class_result = []

for i in range(3):
    probability_result.append((probabilty[i]*100).round(2))
    class_result.append(dict_result[probabilty[i]])
    