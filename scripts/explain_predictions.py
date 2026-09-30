import os
import joblib
import pandas as pd
import numpy as np
import shap

def explain_user(sample_index, model, X_test, feature_names):
    sample = X_test.iloc[[sample_index]]
    pred = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]

    # Compute SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Scikit-learn binary tree produces array of shape (1, n_features, 2) or list of 2 arrays
    if isinstance(shap_values, list):
        vals = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        vals = shap_values[0, :, 1]
    else:
        vals = shap_values[0]

    # Find top 2 positive drivers pushing toward subscription
    top_indices = np.argsort(vals)[::-1]
    reasons = []
    for idx in top_indices:
        feat_name = feature_names[idx]
        feat_val = sample.iloc[0, idx]
        if vals[idx] > 0:
            reasons.append(f"{feat_name} = {feat_val}")
        if len(reasons) == 2:
            break

    if not reasons:
        reasons = [f"{feature_names[top_indices[0]]} = {sample.iloc[0, top_indices[0]]}"]

    reason_str = " and ".join(reasons)
    status = "Will Buy" if pred == 1 else "Will Not Buy"
    explanation = f"User {sample_index} -> [{status}] (Confidence: {prob:.2%}) because {reason_str}."
    return explanation

def main():
    model_path = os.path.join("models", "decision_tree_model.joblib")
    test_path = os.path.join("data", "X_test_sample.csv")

    if not os.path.exists(model_path) or not os.path.exists(test_path):
        print("Model or test data missing. Run 'python src/train.py' first.")
        return

    model = joblib.load(model_path)
    X_test = pd.read_csv(test_path)
    feature_names = list(X_test.columns)

    print("\n--- SHAP 'Mind Reader' Inference Output ---")
    # Generate explanations for sample predictions
    for idx in [0, 5, 12, 25, 40]:
        if idx < len(X_test):
            sentence = explain_user(idx, model, X_test, feature_names)
            print(sentence)
    print("-------------------------------------------\n")

if __name__ == "__main__":
    main()
