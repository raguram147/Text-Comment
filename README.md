# Toxic Comment Classifier

The internet can be a mean and nasty place...but it doesn't need to be! This project demonstrates how to spot and detect toxic comments using deep learning and Python. The dataset leveraged is originally from Kaggle, but you can substitute it with your own data. Additionally, this project shows how to make predictions from raw strings, and it provides an interactive interface using Gradio.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Interactive Interface](#interactive-interface)
- [Results](#results)
- [Contributing](#contributing)

## Introduction

This project aims to detect toxic comments in textual data using deep learning techniques. Toxic comments can include various types of harmful content, such as insults, threats, and hate speech. By leveraging deep learning models, we can build a classifier to identify and filter out such comments to maintain a healthier online environment.

## Dataset

The dataset used in this project is originally from Kaggle. It contains comments labeled with different types of toxicity, such as toxic, severe toxic, obscene, threat, insult, and identity hate. You can download the dataset from [Kaggle's Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data).

## Installation

To run this project, you'll need to install the necessary dependencies. You can do this by running:

`bash
pip install -r requirements.txt`


## **Requirements**
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Gradio

**Model Architecture**
The model is built using TensorFlow and consists of the following layers:

- Embedding Layer
- Bidirectional LSTM Layer
- Dense Layer
- Dropout Layer
- Output Layer with Sigmoid Activation

This architecture helps capture the text data's sequential patterns and make accurate predictions.

**Training the Model**
The training script (train.py) reads the preprocessed data, defines the model architecture, and trains the model using the training data. The script saves the trained model to a specified directory for later use.

**Making Predictions**
The prediction script (predict.py) loads the trained model and makes predictions on new comments. You can provide a raw string input, and the script will output the probability of the comment being toxic.

Example:

`bash
python predict.py --text "This is an example comment"`

Interactive Interface
The interactive interface uses Gradio to provide a user-friendly web application for making predictions. Run the following command to start the Gradio interface:

`bash
python app.py`

This will open a web browser window where you can enter comments and see the model's predictions in real time.

##**Results**

After training the model, you can evaluate its performance using various metrics such as accuracy, precision, recall, and F1 score. The results will help you understand the effectiveness of the model in detecting toxic comments.

Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request.
