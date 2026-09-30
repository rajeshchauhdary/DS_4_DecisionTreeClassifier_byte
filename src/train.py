import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def load_data():
    csv_path = os.path.join("data", "bank-additional", "bank-additional-full.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join("data", "bank-full.csv")
    df = pd.read_csv(csv_path, sep=";")
    return df

def preprocess_data(df):
    df_clean = df.copy()
    
    # Target encoding: 'yes' -> 1, 'no' -> 0
    df_clean['y'] = df_clean['y'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Separate features and target
    X = df_clean.drop('y', axis=1)
    y = df_clean['y']
    
    # Encode categorical variables using dummy encoding
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X_encoded, y

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("[1/5] Loading data...")
    df = load_data()
    print(f"Dataset shape: {df.shape}")
    
    print("[2/5] Preprocessing and encoding features...")
    X, y = preprocess_data(df)
    
    # Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("[3/5] Training Decision Tree Classifier...")
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    print("[4/5] Evaluating model performance...")
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save Confusion Matrix visual
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title("Confusion Matrix - Bank Subscription Prediction")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # Top 5 Feature Importance visual
    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top5 = importances.head(5)
    
    plt.figure(figsize=(8, 4))
    sns.barplot(x=top5.values, y=top5.index, palette='viridis')
    plt.title("Top 5 Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "feature_importance.png"), dpi=300)
    plt.close()
    
    print("[5/5] Saving model and test sets...")
    joblib.dump(clf, os.path.join("models", "decision_tree_model.joblib"))
    X_test.to_csv(os.path.join("data", "X_test_sample.csv"), index=False)
    y_test.to_csv(os.path.join("data", "y_test_sample.csv"), index=False)
    print("Training pipeline executed successfully. Artifacts saved in models/ and outputs/.")

if __name__ == "__main__":
    main()
