# Deep Neural Network for Malaria Infected Cell Recognition

## AIM:

To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.

## Problem Statement:
The problem at hand is the automatic classification of red blood cell images into two categories: parasitized and uninfected. Malaria-infected red blood cells, known as parasitized cells, contain the Plasmodium parasite, while uninfected cells are healthy and free from the parasite. The goal is to build a convolutional neural network (CNN) model capable of accurately distinguishing between these two classes based on cell images.

Traditional methods of malaria diagnosis involve manual inspection of blood smears by trained professionals, which can be time-consuming and error-prone. Automating this process using deep learning can significantly speed up diagnosis, reduce the workload on healthcare professionals, and improve the accuracy of detection.

Our dataset comprises 27,558 cell images, evenly split between parasitized and uninfected cells. These images have been meticulously collected and annotated by medical experts, making them a reliable source for training and testing our deep neural network.
## Dataset:
![image](https://github.com/SOMEASVAR/malaria-cell-recognition/assets/93434149/388c0834-7f00-49f4-92eb-885baeea620a)


## Neural Network Model



## DESIGN STEPS

### Step 1: Import Libraries
We begin by importing the necessary Python libraries, including TensorFlow for deep learning, data preprocessing tools, and visualization libraries.

### Step 2: Allow GPU Processing
To leverage the power of GPU acceleration, we configure TensorFlow to allow GPU processing, which can significantly speed up model training.

### Step 3: Read Images and Check Dimensions
We load the dataset, consisting of cell images, and check their dimensions. Understanding the image dimensions is crucial for setting up the neural network architecture.

### Step 4: Image Generator
We create an image generator that performs data augmentation, including rotation, shifting, rescaling, and flipping. Data augmentation enhances the model's ability to generalize and recognize malaria-infected cells in various orientations and conditions.

### Step 5: Build and Compile the CNN Model
We design a convolutional neural network (CNN) architecture consisting of convolutional layers, max-pooling layers, and fully connected layers. The model is compiled with appropriate loss and optimization functions.

### Step 6: Train the Model
We split the dataset into training and testing sets, and then train the CNN model using the training data. The model learns to differentiate between parasitized and uninfected cells during this phase.

### Step 7: Plot the Training and Validation Loss
We visualize the training and validation loss to monitor the model's learning progress and detect potential overfitting or underfitting.

### Step 8: Evaluate the Model
We evaluate the trained model's performance using the testing data, generating a classification report and confusion matrix to assess accuracy and potential misclassifications.

### Step 9: Check for New Image
We demonstrate the model's practical use by randomly selecting and testing a new cell image for classification.

## PROGRAM:

Include your code here

## OUTPUT:

### Training Loss, Validation Loss Vs Iteration Plot:

![image](https://github.com/SOMEASVAR/malaria-cell-recognition/assets/93434149/513b341e-535a-4fd6-8218-95ef96445da7)


### Classification Report:
![image](https://github.com/SOMEASVAR/malaria-cell-recognition/assets/93434149/7ef0deec-c612-4007-b0b3-e13fed02b655)


### Confusion Matrix:

![image](https://github.com/SOMEASVAR/malaria-cell-recognition/assets/93434149/3d038622-b3e0-4770-9076-f80cde2a7137)

### New Sample Data Prediction:

![image](https://github.com/SOMEASVAR/malaria-cell-recognition/assets/93434149/1304d333-4c88-4556-8cf5-785a74820e49)


## RESULT:
Thus, a deep neural network for Malaria infected cell recognition is developed and the performance is analyzed.
