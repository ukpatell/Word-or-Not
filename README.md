<img src="./templates/img.png" alt="studazon-green.png">

# Word-or-Not

<hr>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0-blue.svg?cacheSeconds=2592000" />
</p>

## Authors

<hr>
👤 <b>Umang Patel</b>

* Website: https://ukpatell.com
* Github: [@UKPatell](https://github.com/ukpatell)
* LinkedIn: [@UKPatel](https://linkedin.com/in/ukpatel)

## Overview

> The goal of this project is to create a program that can determine whether a given set of letters is a word or not. We
> will use a dataset containing many sets of random letters and their labels, which tell us if they are words or not. We
> will train and test different machine learning methods to find the one that can do this task the best. The goal is to
> create a program that can accurately and confidently tell us if a given set of letters is a word or not.

## Data Sources

> We obtained the dataset from Kaggle, which contains over 479k English words.

## Role of Artificial Intelligence

> This project is related to artificial intelligence because it involves training a computer program to learn how to
> distinguish between words and non-words. This is done using machine learning algorithms, which allow the program to
> automatically learn from a large dataset of examples without being explicitly programmed for every possible scenario.
> The program will analyze the patterns and characteristics of the input data to identify the features that are most
> important for distinguishing between words and non-words, and then use this knowledge to classify new data that it has
> never seen before. The end goal is to create a program that can accurately and automatically classify new strings of
> characters as either words or non-words with a high degree of accuracy.

## Libraries Used

- Pandas: library for data manipulation and analysis in Python.
- NumPy: library for numerical computing in Python.
- CountVectorizer: tool for converting text data to numerical vectors for input in machine learning algorithms.
- train_test_split: function for splitting datasets into training and testing sets.
- LogisticRegression: algorithm for classification of words and non-words.
- accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay: functions for testing the performance
  of the model.
- Seaborn: library for data visualization.
- Pickle: tool for saving machine learning models.

## Data Preprocessing

> We loaded the dataset and applied some pre-processing techniques to clean the data, such as removing any
> non-alphabetic characters and converting all characters to lowercase.

## Data Visualization

> We used Seaborn to create a heatmap visualization of the input data to help identify any patterns or correlations in
> the data.

## Train Test Split

> We used the train_test_split function to split the data into training and testing sets for use in our machine learning
> algorithm.

## Model Building

> We used the Logistic Regression algorithm to train our model to distinguish between words and non-words. We then
> evaluated the performance of our model using the accuracy_score, classification_report, and confusion_matrix functions.

## Save Model

> Finally, we saved our machine learning model using the Pickle tool to be used in other applications.

Overall, this project provides an example of how machine learning algorithms can be used to automatically distinguish
between words and non-words, which can have practical applications in areas such as spell checking, text processing, and
natural language understanding.