# neural_network_examples
This repository contains three neural network implementations about simple problems, as explained below.

Reference:
[_Deep Learning with Python._](https://www.amazon.com/Deep-Learning-Python-Francois-Chollet/dp/1617294438/)
Chollet, Francois. 2017

### house-prices-regression

Predicting house prices: a regression example with neural network.

Dataset: [Boston house prices](https://keras.io/api/datasets/boston_housing/)

The goal is to predict median house prices, given data points with features as crime rate, local property tax rate, and so on.
The dataset has 506 data points (404 training samples and 102 test samples), and each feature has a different scale.



### imdb-binary-classification

Binary classification of [IMDB](https://www.imdb.com) movie reviews with Neural Networks.

The model here applied is a two-class classification, one of the most known machine learning approaches.

In this example, I am using Keras/TensorFlow to classify movie reviews as positive or negative, based on the text content of the reviews.

The IMDB dataset is a set of 50k polarized reviews from the Internet Movie Database (IMDB), split into 25k reviews for training and 25k reviews for testing, and each set consisting of 50% negative and 50% positive reviews.



### reuters-newswire-multiclassification

Single label, multi-class classification of [Reuters Newswire Keras database](https://keras.io/api/datasets/reuters/), using a Deep Neural Network.

The model here applied is to solve a very usual problem in many businesses: a _single label, multi-class classification problem_.

In this example, I am using Keras/TensorFlow to classify Reuters newswires into 46 mutually exclusive topics, based on its content. Notice that each data point should be classified into only one category, otherwise it would be a _multi-label classification problem_.

Reuters dataset is a set of newswires and their topics (there are 46 topics in total), published by Reuters in 1986.
