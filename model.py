import os  
import PIL  # Import Python Imaging Library
import shutil  # Import file operation utilities
import pathlib  # Import object-oriented filesystem paths
import numpy as np  
import matplotlib.pyplot as plt  

import tensorflow as tf 
from tensorflow import keras  
from tensorflow.keras import layers  
from tensorflow.keras.models import Sequential, save_model  


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"  #
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)  
data_dir = pathlib.Path(data_dir)
image_count=len(list(data_dir.glob('*/*.jpg')))
print('Total No of image ',image_count)

batch_size=32
img_height=180
img_width=180

train_ds=tf.keras.preprocessing.image_dataset_from_directory(  # Create training dataset
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_height, img_width),
            batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(  # Create validation dataset
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
)

class_names = train_ds.class_names  # Get class names from training dataset
print("Class Names :", class_names)  # Print class names

train_ds = train_ds.cache().shuffle(1000)  # Cache and shuffle training dataset
val_ds = val_ds.cache()  # Cache validation dataset

num_classes = len(class_names)  # Determine number of classes



model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    # Rescaling layer: Normalizes pixel values to [0,1] by dividing by 255.
    # 'input_shape' specifies the input image dimensions (height, width, channels). Here, it's (img_height, img_width, 3) for RGB images.

    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    # Convolutional layer with 16 filters, each using a 3x3 kernel.
    # 'padding='same'' adds padding to ensure the output has the same spatial dimensions as the input.
    # 'activation='relu'' applies the Rectified Linear Unit activation function.

    tf.keras.layers.MaxPooling2D(),
    # MaxPooling layer: Performs max pooling operation using a 2x2 window and a stride of 2 to downsample the feature maps.

    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    # Another Convolutional layer with 32 filters, 3x3 kernel, 'padding='same'', and 'activation='relu''.

    tf.keras.layers.MaxPooling2D(),
    # Another MaxPooling layer.

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    # Another Convolutional layer with 64 filters, 3x3 kernel, 'padding='same'', and 'activation='relu''.

    tf.keras.layers.MaxPooling2D(),
    # Another MaxPooling layer.

    tf.keras.layers.Flatten(),
    # Flatten layer: Flattens the 3D output to 1D to prepare for fully connected layers.

    tf.keras.layers.Dense(128, activation='relu'),
    # Dense (fully connected) layer with 128 neurons and 'activation='relu''.

    tf.keras.layers.Dense(num_classes)
    # Output Dense layer with 'num_classes' neurons (number of output classes).
    # No activation specified here; it's common in multi-class classification to omit activation to get logits.
])


model.compile(optimizer='adam',  # Compile model with optimizer, loss function, and metrics
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
print(model.summary())  

print("Model Training....")  
epochs=10 
history = model.fit(  
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)
print("Training Complete!")  

acc = history.history['accuracy']  
val_acc = history.history['val_accuracy']  

loss = history.history['loss']  
val_loss = history.history['val_loss'] 

epochs_range = range(epochs)  

plt.figure(figsize=(8, 8))  
plt.subplot(1, 2, 1) 
plt.plot(epochs_range, acc, label='Training Accuracy') 
plt.plot(epochs_range, val_acc, label='Validation Accuracy') 
plt.legend(loc='lower right')  
plt.title('Training and Validation Accuracy')  


plt.subplot(1, 2, 2)  
plt.plot(epochs_range, loss, label='Training Loss')  
plt.plot(epochs_range, val_loss, label='Validation Loss')  
plt.legend(loc='upper right')  
plt.title('Training and Validation Loss') 
plt.show()  

# Save the trained model to a file
save_model(model, 'flower_model_trained.hdf5') 
print("Model Saved")  