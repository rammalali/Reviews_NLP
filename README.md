# Project Name

## Amazon Fine Food Reviews Sentiment Analysis

### Introduction

This project aims to analyze Amazon Fine Food Reviews and predict whether they are positive or negative. The dataset can be obtained from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews). The project uses pre-trained GloVe vectors for word embedding, which can be obtained from the [GloVe website](https://nlp.stanford.edu/projects/glove/).

### Installation

To run this project, you need to have Python installed on your system. You also need the following libraries:

- NumPy
- Pandas
- Matplotlib
- Regular expression (re)
- Scikit-Learn

You can install these libraries using Anaconda. Once you have installed Anaconda, you can create a new environment and install the required libraries using the following command:
* conda create --name myenv python numpy pandas matplotlib scikit-learn . . .


### Usage

To use this project, you need to download the Amazon Fine Food Reviews dataset from Kaggle and place it in a `datasets/` folder in the root directory of the project.

You also need to download the pre-trained GloVe vectors from the GloVe website and unzip them in a `GloVe/` folder in the root directory of the project.

Once you have downloaded the dataset and the pre-trained GloVe vectors, you can run the following Jupyter notebooks in order:

- NLP1.ipynb
- NLP_GloVe.ipynb or NLP_vectorizations.ipynb

You can run these notebooks in Jupyter Lab or Jupyter Notebook. The NLP1.ipynb notebook performs data preprocessing and creates the train and test datasets. The NLP_GloVe.ipynb notebook trains the model using GloVe vectors, while the NLP_vectorizations.ipynb notebook trains the model using vectorizations(bow, TF-IDF).

### Conclusion

This project demonstrates the use of pre-trained word embeddings and machine learning algorithms to analyze sentiment in Amazon Fine Food Reviews. With further improvements, the project can be extended to other datasets and applications.


![Your Repository's Stats](https://github-readme-stats.vercel.app/api?username=rammalali&show_icons=true)
