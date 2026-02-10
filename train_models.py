import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_datasets

def train_and_evaluate():
    datasets = load_datasets()
    results = {}

    for name, df in datasets.items():
        print(f"\n{'='*20} Processing {name} {'='*20}")
        
        X = df.drop('target', axis=1) if 'target' in df.columns else df.drop('Outcome', axis=1)
        y = df['target'] if 'target' in df.columns else df['Outcome']

        # Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }

        dataset_results = {}
        for model_name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            dataset_results[model_name] = acc
            print(f"{model_name} Accuracy: {acc:.4f}")

        results[name] = dataset_results

    # Summary Table
    summary_df = pd.DataFrame(results).T
    print("\nSummary of Accuracies:")
    print(summary_df)
    
    # Simple Plotting
    summary_df.plot(kind='bar', figsize=(10, 6))
    plt.title("Model Accuracy Comparison across Datasets")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    print("\nAccuracy comparison plot saved as 'accuracy_comparison.png'")

if __name__ == "__main__":
    train_and_evaluate()
