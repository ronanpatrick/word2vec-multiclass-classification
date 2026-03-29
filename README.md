# E-Commerce Sentiment Analysis via Word2Vec Embeddings

> **Project Objective:** To develop a multiclass classification model capable of predicting customer sentiment from localized e-commerce reviews using semantic word embeddings.

## Overview
This repository contains an advanced Natural Language Processing (NLP) project focusing on semantic feature extraction. Unlike standard frequency-based vectorization (such as TF-IDF), this pipeline utilizes Gensim's Word2Vec to generate dense vector embeddings. This allows the machine learning models to understand the contextual meaning and relationships between Tagalog and English words within product reviews, mapping them to multiple sentiment classes.

**A detailed evaluation of the model performance, classification reports, and accuracy metrics is available in the included PDF Evaluation Report.**

## Dataset
* **Structure:** The dataset is pre-split into `train.csv`, `test.csv`, and `val.csv`.
* **Domain:** Tagalog and Taglish e-commerce product reviews.
* **Target Variable:** `sentiment` (Multiclass discrete categories)

## Repository Structure
* **`Word2Vec_Evaluation_Report.pdf`**: The formal documentation detailing the methodology, embedding strategy, and a comparative analysis of model performance.
* **`Word2Vec_Sentiment_Analysis.ipynb`**: The core Jupyter Notebook containing the data preprocessing, Gensim Word2Vec implementation, TSNE visualization, and multi-model training.
* **Data Files**: `train.csv`, `test.csv`, and `val.csv`.

## Tech Stack
* **Language:** Python
* **Natural Language Processing:** Gensim (`Word2Vec`), Regular Expressions (`re`)
* **Machine Learning Models:** Scikit-Learn (`LogisticRegression`, `SVC`, `RandomForestClassifier`)
* **Data Manipulation & Viz:** Pandas, NumPy, Matplotlib, Seaborn, TSNE

## How to Run the Notebook

### Run in Cloud (Recommended)
You can view and execute the code instantly in your browser via Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ronanpatrick/word2vec-multiclass-classificationc/blob/main/Word2Vec_Multiclass_Classification.ipynb)

*(Note: If running in Colab, please ensure you upload `train.csv`, `test.csv`, and `val.csv` to your session storage first).*

### Run Locally
1. Clone this repository: `git clone https://github.com/ronanpatrick/ecommerce-sentiment-word2vec.git`
2. Install the required libraries: `pip install pandas numpy scikit-learn gensim matplotlib seaborn`
3. Open the `.ipynb` file in Jupyter Notebook or VS Code and execute the cells.
