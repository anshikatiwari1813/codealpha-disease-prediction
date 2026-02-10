import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def load_datasets():
    """
    Loads Breast Cancer (from sklearn), Heart Disease (manual UCI path),
    and Diabetes (scikit-learn or UCI path).
    """
    datasets = {}

    # 1. Breast Cancer (Built-in)
    cancer = load_breast_cancer()
    datasets['breast_cancer'] = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    datasets['breast_cancer']['target'] = cancer.target
    print("Loaded Breast Cancer dataset.")

    # 2. Heart Disease (UCI repository - processed.cleveland.data)
    # URL: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
    heart_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    try:
        df_heart = pd.read_csv(heart_url, names=heart_cols, na_values="?")
        df_heart.dropna(inplace=True) # Simple handling
        # Target 0 = No disease, 1-4 = Disease. Convert to binary 0/1.
        df_heart['target'] = df_heart['target'].apply(lambda x: 1 if x > 0 else 0)
        datasets['heart_disease'] = df_heart
        print("Loaded Heart Disease dataset.")
    except Exception as e:
        print(f"Error loading Heart Disease dataset: {e}")

    # 3. Diabetes (Pima Indians from UCI via raw GitHub or equivalent)
    # URL: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
    diabetes_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    diabetes_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    try:
        df_diabetes = pd.read_csv(diabetes_url, names=diabetes_cols)
        datasets['diabetes'] = df_diabetes
        print("Loaded Diabetes dataset.")
    except Exception as e:
        print(f"Error loading Diabetes dataset: {e}")

    return datasets

if __name__ == "__main__":
    data = load_datasets()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
