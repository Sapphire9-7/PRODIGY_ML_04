# PRODIGY_ML_04
Task 4 of the Prodigy InfoTech ML internship which involves building a hand gesture recognition model using CNN.

![Hand Gesture Recognition Demo](https://github.com/Sapphire9-7/PRODIGY_ML_04/blob/main/GIF/HandGes.gif)

# Hand Gesture Recognition using CNN

This project is focused on recognizing hand gestures using Convolutional Neural Networks (CNN). The project consists of two main components:

## File 1: HandGesRec.ipynb

### Description
This Jupyter Notebook file contains the code for building a CNN model for hand gesture recognition. It starts with data preprocessing, reading gesture images, processing them, and building the CNN model.

### Key Steps:
1. Reading and Preprocessing Data: Gesture images are read from the dataset, and preprocessing steps, including image resizing, are applied.
2. Label Encoding: Gesture labels are converted into numerical representations.
3. Data Split: The dataset is split into training and testing sets.
4. CNN Model Building: A CNN model is constructed with multiple layers for feature extraction and classification.
5. Model Training: The model is trained using the training data, with early stopping to prevent overfitting.
6. Model Evaluation: The model's accuracy is evaluated on the testing data.
7. Learning Curves: Learning curves are plotted to visualize model performance.
8. Model Saving: The trained model is saved for later use.

## File 2: Real-TimeRec.ipynb

### Description
This Jupyter Notebook file contains the code for real-time hand gesture recognition. It uses the pre-trained CNN model to classify gestures in real-time using the laptop's camera. The code also utilizes KNN background subtraction to segment the hand from the frame.

### Key Steps:
1. Pre-trained Model Loading: The pre-trained CNN model from `HandGesRec.ipynb` is loaded.
2. Camera Initialization: The laptop's camera is initialized for real-time gesture recognition.
3. Background Subtraction: KNN background subtraction is used to extract the hand from the frame.
4. Skin Color Detection: The code detects skin color within the frame.
5. Combining Masks: The skin mask is combined with the background subtraction mask.
6. Morphological Operations: Morphological operations are applied to reduce noise in the segmented hand.
7. Model Predictions: The segmented hand is resized, and the CNN model predicts the gesture in real-time.
8. Display: The recognized gesture is displayed on the frame.

## Requirements
- Python
- OpenCV
- NumPy
- PIL (Pillow)
- Matplotlib
- Keras
- TensorFlow

## Usage
1. Run `HandGesRec.ipynb` to train and save the CNN model.
2. Run `Real-TimeRec.ipynb` to perform real-time hand gesture recognition.

