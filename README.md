# K-Nearest-Neighbors-Multilabel

## Project Overview
This project uses the K-Nearest Neighbors (KNN) machine learning algorithm to classify Iris flowers into three species: Setosa, Versicolor, and Virginica. The model is trained using flower measurements and evaluated using different performance metrics.

## Objectives
- Understand and implement the KNN classification algorithm.
- Perform data preprocessing.
- Split the dataset into training and testing sets.
- Train a machine learning model.
- Evaluate model performance.
- Predict the species of a new Iris flower.
- Save and load the trained model using Pickle.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Pickle

## Dataset
The project uses the Iris dataset, which contains measurements of three Iris flower species.

The features used for prediction are:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable is the flower species:
- Iris Setosa
- Iris Versicolor
- Iris Virginica

## Project Workflow
1. Load the Iris dataset using Pandas.
2. Remove the unnecessary ID column.
3. Convert species names into numerical labels.
4. Separate the input features and target variable.
5. Split the dataset into training and testing sets.
6. Train the KNN classification model.
7. Evaluate the model using accuracy, confusion matrix, and classification report.
8. Predict the species for a new data point.
9. Save and reload the trained model using Pickle.

## Algorithm Used: K-Nearest Neighbors (KNN)
KNN is a supervised machine learning algorithm used for classification and regression. It predicts the class of a new data point by identifying its nearest neighbors in the training dataset.

In this project, KNN classifies Iris flowers based on their sepal and petal measurements.

## Model Evaluation
The model's performance is evaluated using:
- **Accuracy Score:** Measures the proportion of correct predictions.
- **Confusion Matrix:** Compares actual and predicted classifications.
- **Classification Report:** Provides precision, recall, F1-score, and support for each class.

## Model Saving
The trained model is saved using Python's Pickle library. It can be loaded later to make predictions without training the model again.

## How to Run the Project

### 1. Install the Required Libraries
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Prepare the Dataset
Place `iris.csv` in the same directory as the Python script.

### 3. Run the Python File
```bash
python KNN_Multilabel.py
```

The program displays the training and testing evaluation results and predicts the species for a sample input.

## Future Enhancements
- Improve performance through feature scaling.
- Find the optimal number of neighbors.
- Compare KNN with other machine learning algorithms.
- Develop a web application for interactive flower classification.

## Conclusion
This project demonstrates how the KNN algorithm can be applied to a classification problem using Python and Scikit-learn. It covers data preprocessing, model training, evaluation, prediction, and model persistence.
