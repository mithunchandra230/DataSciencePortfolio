# Data Science Portfolio

Welcome to my Data Science Portfolio! I'm a passionate data scientist with expertise in machine learning, deep learning, and data visualization. This repository showcases eight end-to-end projects demonstrating skills in regression, classification, NLP, computer vision, time series forecasting, and deployment. Each project includes data preprocessing, model development, evaluation, visualizations, and (where applicable) interactive Streamlit apps.

## Projects

### 1. Food Delivery Time Prediction

- **Description**: Predicted food delivery times to optimize logistics and improve customer satisfaction.
- **Dataset**: Custom dataset with features like order time, restaurant location, and distance (assumed, e.g., Kaggle Food Delivery dataset).
- **Methods**:
  - Regression models: Linear Regression, Random Forest.
  - Feature engineering: Distance calculations, time-based features.
  - Visualizations: Delivery time distributions, feature importance plots.
- **Results**: Achieved RMSE of \~5 minutes with Random Forest.
- **Tech**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn.
- **Files**: Notebook, Feature Importance.

### 2. Heart Disease Prediction

- **Description**: Developed a classifier to predict heart disease risk, aiding medical decision-making.
- **Dataset**: UCI Heart Disease dataset (e.g., age, cholesterol, chest pain type).
- **Methods**:
  - Classification: Logistic Regression, Random Forest, SVM.
  - Preprocessing: Handling missing values, scaling features.
  - Visualizations: Correlation heatmap, ROC curve.
- **Results**: Achieved \~85% accuracy with Random Forest.
- **Tech**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn.
- **Files**: Notebook, ROC Curve.

### 3. Spam/Ham SMS Classifier

- **Description**: Built a model to classify SMS messages as spam or ham to enhance messaging security.
- **Dataset**: UCI SMS Spam Collection (text messages labeled as spam/ham).
- **Methods**:
  - NLP: TF-IDF vectorization, text preprocessing (tokenization, stop words).
  - Models: Naive Bayes, Logistic Regression.
  - Visualizations: Word clouds, class distribution.
- **Results**: Achieved \~98% accuracy with Naive Bayes.
- **Tech**: Python, Pandas, NLTK, Scikit-learn, Matplotlib, WordCloud.
- **Files**: Notebook, Spam Word Cloud.

### 4. Sentiment-Based Product Review Analysis

- **Description**: Analyzed customer reviews to classify sentiment and provide business insights.
- **Dataset**: Amazon product reviews (assumed, e.g., Kaggle dataset).
- **Methods**:
  - NLP: TF-IDF, sentiment analysis with Naive Bayes.
  - Preprocessing: SMOTE for class imbalance, text cleaning.
  - Visualizations: Sentiment distribution, word clouds.
- **Results**: Achieved \~90% accuracy in sentiment classification.
- **Tech**: Python, Pandas, NLTK, Scikit-learn, Matplotlib, Seaborn.
- **Files**: Notebook, Sentiment Plot.

### 5. Movie Recommendation System

- **Description**: Created a recommendation system to suggest movies based on user preferences.
- **Dataset**: MovieLens 100K (user ratings, movie genres).
- **Methods**:
  - Collaborative filtering: Cosine similarity for user-based recommendations.
  - Content-based filtering: TF-IDF on genres.
  - Visualizations: Rating distribution, genre popularity.
- **Results**: Precision@5 of \~0.75 for collaborative filtering.
- **Tech**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn, Streamlit.
- **Files**: Notebook, Rating Plot, Streamlit App.

### 6. House Price Prediction

- **Description**: Predicted house prices to support real estate decisions.
- **Dataset**: California Housing (Scikit-learn, features like median income, house age).
- **Methods**:
  - Regression: Random Forest, XGBoost.
  - Preprocessing: Feature scaling, outlier handling.
  - Visualizations: Price distribution, feature importance.
- **Results**: Achieved RMSE of \~0.52 ($52,000) with Random Forest.
- **Tech**: Python, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, Streamlit.
- **Files**: Notebook, Feature Importance, Streamlit App.
- **Demo**: Live App (replace with actual URL).

### 7. Image Classification with Convolutional Neural Networks (CNNs)

- **Description**: Classified images into 10 categories using deep learning.
- **Dataset**: CIFAR-10 (32x32 RGB images, 10 classes like airplane, dog).
- **Methods**:
  - Deep learning: Custom CNN with convolutional and pooling layers.
  - Preprocessing: Image normalization, data augmentation.
  - Visualizations: Accuracy/loss curves, confusion matrix.
- **Results**: Achieved \~70% test accuracy.
- **Tech**: Python, TensorFlow, Keras, Matplotlib, Seaborn, Streamlit.
- **Files**: Notebook, Confusion Matrix, Streamlit App.
- **Demo**: Live App (replace with actual URL).

### 8. Time Series Forecasting for Stock Prices

- **Description**: Forecasted stock prices to inform investment strategies.
- **Dataset**: Apple (AAPL) stock prices (yfinance, 2020–2025).
- **Methods**:
  - Deep learning: LSTM for time series prediction.
  - Preprocessing: MinMax scaling, sequence creation.
  - Visualizations: Historical vs. predicted prices, loss curves.
- **Results**: Achieved RMSE of \~$6.82 on test set.
- **Tech**: Python, Pandas, yfinance, TensorFlow, Matplotlib, Seaborn, Streamlit.
- **Files**: Notebook, Forecast Plot, Streamlit App.
- **Demo**: Live App (replace with actual URL).

## Skills

- **Programming**: Python, Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost, Naive Bayes, Logistic Regression, Random Forest
- **Deep Learning**: TensorFlow, Keras, CNNs, LSTM
- **NLP**: NLTK, TF-IDF, sentiment analysis
- **Data Visualization**: Matplotlib, Seaborn, WordCloud
- **Deployment**: Streamlit, GitHub Pages
- **Other**: yfinance, data augmentation, feature engineering, time series analysis

## Setup

To run these projects, install the required libraries:

```bash
pip install pandas numpy scikit-learn tensorflow yfinance nltk xgboost streamlit matplotlib seaborn wordcloud
```

- Clone this repository:

  ```bash
  git clone https://github.com/your-username/DataSciencePortfolio.git
  ```

- Run notebooks in Jupyter or Colab.

- For Streamlit apps, use:

  ```bash
  streamlit run app.py
  ```

## Contact

- **LinkedIn**: Your LinkedIn
- **Portfolio Website**: Your Website
- **Email**: your.email@example.com
- **GitHub**: Your GitHub

Feel free to explore the notebooks, try the Streamlit apps, and reach out with questions or feedback!