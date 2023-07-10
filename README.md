# SVM Gender Classifier from Handwriting
This repository contains a program that recognizes and classifies the gender of the writer from handwritten text in Hebrew. 
It utilizes image processing techniques and a Support Vector Machine (SVM) model to analyze and classify the gender based on extracted Local Binary Patterns (LBP) features.
<br/>**This project was developed as part of an image processing course (Final Project).**

## Environment
- Operating System: Windows/macOS
- Development Environment: PyCharm
- Language: Python
- Required Packages: cv2, numpy, sklearn, skinmage

Before running the program, please ensure that you have the necessary packages installed in your Python environment.

## How to Run
To run the program, please follow these steps:

1. Format to run on the command line:
   ```shell
   python classifier.py <train_set_path> <validation_set_path> <test_set_path>
   ```
2. The program will utilize image processing techniques to extract LBP features from the handwritten texts in the training set.
3. The SVM model will be trained on the extracted features to classify the gender of the writer.
4. Once the model is trained, it will be evaluated on the validation set to assess its accuracy.
5. Finally, the model will be tested on the test set to predict and classify the gender of the writers.

Please ensure that the datasets are properly formatted and labeled to achieve accurate results.
### The used dataset:
    Rabaev I., Litvak M., Asulin S., Tabibi O.H. (2021) Automatic Gender Classification from
    Handwritten Images: A Case Study. In: Computer Analysis of Images and Patterns. CAIP 2021.
    Lecture Notes in Computer Science, vol 13053. Springer, Cham.
https://link.springer.com/chapter/10.1007/978-3-030-89131-2_30

## Authors
- [Gil Ben Hamo](https://github.com/gilbenhamo)
- Yovel Aloni
