# Step 1: Imports

    # general imports
import warnings
warnings.filterwarnings("ignore")
import os

    # Data Handling/Transformation
import pandas as pd
import numpy as np

    # Data Visualization
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns

    # Machine/Deep Learning
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Step 2: Read Data
train_image_folders = os.listdir('./data/ImageData/balanced') # 9 folders with 409 Images each
for folder in train_image_folders:
    print(folder, len(os.listdir('./data/ImageData/balanced/'+folder)))

sample_wafer = []
train_path = './data/ImageData/balanced/'
test_path = './data/ImageData/test'

    # get 1 sample per failure-class
for folder in train_image_folders:
    sample_wafer.append(train_path+folder+'/'+os.listdir(train_path+folder+'/')[0])


    # Function to plot one example per class
def plot_samples():
    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(10, 6))
    m = 0
    for i in range(3):
        for j in range(3):
            ax[i][j].imshow(imread(sample_wafer[m]))            
            ax[i][j].set_title(os.path.basename(sample_wafer[m]), fontsize=8)
            m+=1
    plt.subplots_adjust(wspace=0.4, hspace=0.4) 
    plt.show()


# Step 3: Prepare / Manipulate Data
image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.05, # Shift the pic width by a max of 5%
                               height_shift_range=0.05, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest') # Fill in missing pixels with the nearest filled value


batch_size = 16
img_shape = (64, 65, 4)


test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=img_shape[:2],
                                                color_mode='rgba',
                                               batch_size=batch_size,
                                               class_mode='categorical',
                                              shuffle=False)


train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=img_shape[:2],
                                                color_mode='rgba',
                                               batch_size=batch_size,
                                               class_mode='categorical')




# Step 4: Create & Train Model
model = keras.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(4,4), input_shape=img_shape, activation='relu')) #Conv Layer
model.add(layers.MaxPool2D(pool_size=(2,2))) #Pool Layer

model.add(layers.Conv2D(filters=32, kernel_size=(4,4), activation='relu')) #2.Conv Layer
model.add(layers.MaxPool2D(pool_size=(2,2))) #2.Pool Layer

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(9, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model.summary())

history = model.fit_generator(train_image_gen, epochs=10, validation_data=test_image_gen)

# Step 5 Evaluate Model
def model_evaluation(model):
    target_names=['Center', 'Donut', 'Edge-loc', 'Edge-ring', 'Loc', 'Near-Full', 'None', 'Random', 'Scratch']
    Y_pred=model.predict_generator(test_image_gen, 855)
    y_pred=np.argmax(Y_pred,axis=1)

    con_matrix=confusion_matrix(test_image_gen.classes,y_pred)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(con_matrix, cmap=plt.cm.Greens, annot=True, square=True, xticklabels=target_names, yticklabels=target_names)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_ylabel('Actual', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.show()
    report=classification_report(test_image_gen.classes,y_pred,target_names=target_names,output_dict=True)
    report = pd.DataFrame(report).transpose()
    report.to_csv('./data_exploration/classification_report.csv')


     #Plot Loss über die Epochen
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy über die Epochen
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

model_evaluation(model)

