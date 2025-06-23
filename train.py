import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import argparse

def load_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    # Encode categorical features
    for col in df.select_dtypes(include='object').columns:
        if col != 'Churn':
            df[col] = LabelEncoder().fit_transform(df[col])
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    return df

def main(data_path, n_estimators, max_depth):
    df = load_data(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True)
    }
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, preds)
            roc = roc_auc_score(y_test, probs)

            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("roc_auc", roc)
            mlflow.sklearn.log_model(model, "model")

            print(f"{model_name}: Accuracy={acc:.3f}, ROC AUC={roc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/telco_churn.csv")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    main(args.data_path, args.n_estimators, args.max_depth)
# This script trains multiple models on a dataset and logs the results using MLflow.
