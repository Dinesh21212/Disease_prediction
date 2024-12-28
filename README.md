# Disease_prediction

Project Overview:
This project implements a machine-learning model to predict diseases based on symptoms. By utilizing a dataset from Kaggle, the system leverages Support Vector Classifier (SVC), Naive Bayes Classifier, and Random Forest Classifier to create a robust and accurate ensemble prediction model.

Features:
Data Preprocessing: Handles cleaning, encoding, and splitting of data.
Model Training: Utilizes SVC, Gaussian Naive Bayes, and Random Forest models.
Ensemble Prediction: Combines predictions from multiple models for higher accuracy.
Disease Prediction API: Predicts diseases based on user-input symptoms in JSON format.

Dataset:
  Source: Kaggle
  Files:
    Training.csv: For training the models.
    Testing.csv: For testing and evaluation.
  Structure: 132 columns for symptoms, 1 column for prognosis.
  
Prerequisites:
  Python 3.x
    Libraries:
    numpy
    pandas
    scikit-learn
    matplotlib
    seaborn
    
Instructions
1. Setup:
  Clone the repository or download the project files.
  Place the Training.csv and Testing.csv files in a dataset folder.
  Install the required Python libraries using:

  pip install -r requirements.txt
  
2. Data Preprocessing:
  Load the dataset.
  Remove null columns.
  Encode target labels using LabelEncoder.

3. Training the Models:
  Use K-Fold Cross-Validation to evaluate:
  Support Vector Classifier
  Gaussian Naive Bayes Classifier
  Random Forest Classifier
  Train models on the full dataset.

4. Predictions:
Predict diseases based on symptoms by combining outputs from all three models.

5. Running the Function:
Use the predictDisease function to input symptoms (comma-separated) and get predictions.

  print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))
  
Outputs:
  Accuracy: High accuracy achieved for both individual and combined models (up to 100% on test data).
Prediction Format:

  {
      "rf_model_prediction": "Fungal infection",
      "naive_bayes_prediction": "Fungal infection",
      "svm_model_prediction": "Fungal infection",
      "final_prediction": "Fungal infection"
  }
  
Visualization
  Bar Plot: Shows dataset balance across diseases.
  Confusion Matrix: Evaluates model predictions visually.

Notes:
  Ensure input symptoms match the symptom names in the dataset.
  Run the code in Jupyter Notebook for a better step-by-step understanding.
  Enjoy exploring and enhancing this disease prediction system! 😊
