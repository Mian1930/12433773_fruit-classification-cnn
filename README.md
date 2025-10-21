# 12433773_fruit-classification-cnn
Applied Deep Learning
# Image Classification of Fruits Using Convolutional Neural Networks (CNNs)

**Course:** Applied Deep Learning (2025)  
**Student Name:** Mian Azan Farooq 
**Matriculation Number:** 12433773  




## 1. Introduction

The goal of this project is to build a simple and effective deep learning model that can recognize different types of fruits from images.  
I chose this topic because it’s straightforward, visual, and helps me understand the basics of image classification using Convolutional Neural Networks (CNNs).  
Fruit images are easy to collect and work with, and the dataset I’m using is already well-prepared for experiments.

This project will serve as my starting point for understanding how neural networks learn visual features such as color, texture, and shape.




## 2. Related Work (References)

1. Mureșan, H., & Oltean, M. (2018). *Fruit recognition from images using deep learning.* Acta Universitatis Sapientiae, Informatica, 10(1), 26–42. [https://arxiv.org/abs/1712.00580]

2. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet classification with deep convolutional neural networks.* Communications of the ACM, 60(6), 84–90. [https://doi.org/10.1145/3065386]

These two papers helped me understand how convolutional layers work for image recognition and inspired me to apply similar principles on a smaller dataset.



## 3. Dataset Description

**Dataset:** Fruits 360 Dataset  
**Source:**(https://www.kaggle.com/moltean/fruits)  
**Size:** Over 70,000 labeled fruit and vegetable images  
**Image format:** 100x100 RGB images  
**Number of classes:** More than 100 fruit categories  

The dataset is already divided into training and testing sets.  
Each fruit type is captured from multiple angles under controlled lighting conditions, making it suitable for image classification tasks.



## 4. Approach

1. **Data Preparation**  
   - Download and explore the Fruits 360 dataset.  
   - Resize and normalize images.  
   - Apply data augmentation (flipping, rotation, brightness adjustment).

2. **Model Design**  
   - Build a CNN with 3 convolutional layers followed by dense layers.  
   - Use ReLU activation, MaxPooling, and Dropout to prevent overfitting.

3. **Training and Fine-tuning**  
   - Train the model for around 20 epochs.  
   - Experiment with optimizers (Adam, RMSprop) and learning rates.  
   - Evaluate accuracy and confusion matrix.

4. **Application Development**  
   - Create a small Streamlit web app that allows users to upload a fruit image.  
   - Display the predicted fruit type and confidence level.





## 5. Tools and Libraries

- Python 3.10  
- TensorFlow / Keras  
- NumPy and Matplotlib  
- Streamlit (for simple demo app)  
- Google Colab (for training)



## 6. Expected Outcome

By the end of this project, I aim to achieve:
- A trained CNN model with around **90% accuracy** on the test data.  
- A small web app for fruit classification.  
- A better understanding of CNNs and data preprocessing.  

If time allows, I might compare the CNN results with a simple traditional image classifier (like SVM or k-NN) to see how much improvement deep learning provides.



## 7. Repository and Access

This repository contains my project files .  
It will be updated throughout the semester as the project progresses.

**Repository Link:(https://github.com/Mian1930/12433773_fruit-classification-cnn)




