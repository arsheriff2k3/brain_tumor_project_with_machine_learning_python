Brain Tumor Classification using CNN

This repository contains a deep learning model built with TensorFlow and Keras for classifying brain tumor images into different categories. The model is trained using a Convolutional Neural Network (CNN) architecture.
Dataset

The dataset used for training and testing the model consists of brain tumor images categorized into four classes:

    Glioma
    Meningioma
    Normal
    Pituitary Tumor

The dataset is divided into a training set and a testing set. The training set contains 3162 images, while the testing set contains 1577 images.
Model Architecture

The CNN model architecture used for brain tumor classification consists of the following layers:

    Convolutional layer with 64 filters, kernel size of (3, 3), and ReLU activation function.
    Batch Normalization layer for normalization of the output from the convolutional layer.
    Activation layer with ReLU activation function.
    MaxPooling layer with a pool size of (2, 2) for downsampling.
    Dropout layer with a dropout rate of 0.25 to prevent overfitting.
    Three additional sets of convolutional, batch normalization, activation, max pooling, and dropout layers.
    Flatten layer to convert the output from the convolutional layers into a 1D vector.
    Two fully connected dense layers with 256 and 512 units, respectively.
    Batch Normalization and Activation layers for normalization and activation of the output from the dense layers.
    Dropout layers with a dropout rate of 0.25.
    Final dense layer with 4 units and softmax activation function for multi-class classification.

Training and Evaluation

The model is trained using the Adam optimizer with a learning rate of 0.0005. The loss function used is categorical cross-entropy, and the evaluation metric is accuracy.

The training is performed for 22 epochs with a batch size of 64. The training and validation accuracy and loss values are recorded for each epoch.
Results

After training the model, the following results were achieved on the validation set:

    Final validation accuracy: 96.09%
    Final validation loss: 0.1032

The trained model is saved as 'brain_tumor.h5'.
Usage

To use the trained model for predicting brain tumor images, you can load the saved model using TensorFlow and Keras, and then pass the images through the model to obtain the predictions.

Example code for loading the model and making predictions:

python

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = tf.keras.models.load_model('brain_tumor.h5')

# Load and preprocess an example image
image_path = 'path/to/your/image.jpg'
image = load_img(image_path, target_size=(48, 48))
image = img_to_array(image)
image = image / 255.0  # Normalize the image

# Reshape the image to match the model's input shape
image = image.reshape(1, 48, 48, 3)

# Make predictions
predictions = model.predict(image)

Conclusion

This project demonstrates the application of deep learning techniques, specifically CNN, for brain tumor classification. By training a CNN model on a dataset of brain tumor images, we achieved high accuracy in classifying different types of brain tumors. This model can be further improved and used in real-world applications for automated brain tumor diagnosis and treatment planning.

Please refer to the source code and the saved model file for further details on implementation and usage.
