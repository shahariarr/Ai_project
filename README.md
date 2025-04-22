# Sleep Disorder Prediction Model

## Overview
This project aims to predict sleep disorders (None, Sleep Apnea, or Insomnia) using machine learning and deep learning algorithms. The models analyze health, lifestyle, and demographic features to classify individuals into the appropriate sleep disorder category.

## Dataset
The project uses the "Sleep Health and Lifestyle Dataset" that contains various parameters such as:
- Demographics (Age, Gender, Occupation)
- Health metrics (BMI, Blood Pressure, Heart Rate)
- Sleep patterns (Sleep Duration, Quality of Sleep)
- Lifestyle factors (Physical Activity Level, Stress Level, Daily Steps)

## Features
- **Data Preprocessing**: Cleaning, handling missing values, feature engineering
- **Feature Selection**: Using Chi-square test to select the most relevant features
- **Multiple Classification Models**: Implementation of 10+ machine learning algorithms
- **Oversampling Techniques**: SMOTE, ADASYN, RandomOverSampler, etc. to handle class imbalance
- **Deep Learning Models**: Neural networks with different optimizers (Adam, SGD)
- **Model Evaluation**: Accuracy, confusion matrix, classification reports
- **Visualizations**: Distribution plots, correlation matrices, model comparisons

## Implementation Steps
1. **Import Libraries**: Data manipulation, visualization, and ML/DL packages
2. **Load and Clean Data**: Handle missing values and engineer features
3. **Normalize Features**: Standardize numerical features
4. **Feature Selection**: Select most important features using Chi-squared test
5. **Split Data**: Create training, validation, and test sets
6. **Apply Oversampling**: Handle class imbalance with various techniques
7. **Train Models**: Implement various ML algorithms and ensembles
8. **Evaluate Models**: Compare model performance metrics
9. **Deep Learning**: Train neural networks with different configurations
10. **Visualize Results**: Create insightful visualizations of the data and results

## Models Implemented
### Machine Learning Models
- Logistic Regression
- Naive Bayes
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Gradient Boosting
- AdaBoost
- XGBoost
- Multi-Layer Perceptron

### Hybrid Models
- Voting Classifier (Logistic Regression + Random Forest + Gradient Boosting)
- Voting Classifier (SVM + XGBoost + MLP)

### Deep Learning Models
- Sequential Neural Network with Adam optimizer
- Sequential Neural Network with SGD optimizer

## Requirements
- Python 3.x
- pandas
- numpy
- scikit-learn
- tensorflow
- xgboost
- imbalanced-learn
- matplotlib
- seaborn

## Usage
1. Ensure all required libraries are installed
2. Update the dataset path in the code
3. Run the notebook cells sequentially
4. View the results, including model performance and visualizations

## Visualizations
The project includes several visualizations:
- Sleep disorder class distribution (pie chart)
- Feature distributions by class (box plots)
- Correlation heatmap of features
- Model performance comparison
- Confusion matrix for the best model

## Results
The notebook automatically identifies and reports the best-performing model based on test accuracy, along with detailed evaluation metrics for all models.
