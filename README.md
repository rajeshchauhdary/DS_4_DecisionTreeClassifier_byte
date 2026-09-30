# DS_4_DecisionTreeClassifier_byte

Decision Tree Classifier for bank marketing subscription prediction featuring Explainable AI (TreeSHAP) rationale generation, built for the **B.Y.T.E Arithmatrix Virtual Internship Program (AVIP 2026)**.

---

## 📌 Problem Overview
Financial institutions require transparent and interpretable models to determine whether a client will subscribe to a term deposit (`yes`/`no`) during direct marketing campaigns. This pipeline implements a decision tree classifier with local interpretability to produce explicit diagnostic rationale sentences for individual client decisions.

---

## 📊 Dataset Reference
- **Source**: [UCI Machine Learning Repository - Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Primary Data**: `bank-additional-full.csv` (41,188 instances, 20 input attributes)
- **Target**: `y` (binary: 1 = Subscribed, 0 = Not Subscribed)

---

## ⚙️ Project Structure
```text
DS_4_DecisionTreeClassifier_byte/
├── data/                       # Dataset archives and extracted CSVs
├── models/                     # Serialized model artifact (.joblib)
├── notebooks/                  # Interactive experimentation notebook
│   └── bank_marketing_classification.ipynb
├── outputs/                    # Exported visualizations
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── scripts/                    # Inference & explainability scripts
│   └── explain_predictions.py
├── src/                        # Model training and data preprocessing
│   └── train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 📈 Model Performance & Evaluation

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 84.55% |
| **Precision** | 41.52% |
| **Recall** | 91.06% |
| **F1-Score** | 57.04% |

The classifier is tuned with balanced class weights (`class_weight='balanced'`) to prioritize customer acquisition recall in an imbalanced scenario.

### Key Visuals
- **Confusion Matrix**: `outputs/confusion_matrix.png`
- **Feature Importance**: `outputs/feature_importance.png` (Top drivers: `duration`, `nr.employed`, `euribor3m`, `pdays`, `cons.conf.idx`)

---

## 🧠 'Mind Reader' SHAP Explainability Output
Rather than returning raw classification outputs, local TreeSHAP values dynamically determine positive contributing attributes per client:

```text
User 0  -> [Will Not Buy] (Confidence: 1.21%)  because cons.price.idx = 93.918 and previous = 0.
User 5  -> [Will Buy]     (Confidence: 70.15%) because nr.employed = 5076.2 and euribor3m = 0.869.
User 12 -> [Will Not Buy] (Confidence: 2.77%)  because cons.conf.idx = -47.1 and euribor3m = 1.41.
User 25 -> [Will Not Buy] (Confidence: 1.21%)  because previous = 0.
User 40 -> [Will Not Buy] (Confidence: 1.21%)  because cons.price.idx = 93.918 and previous = 0.
```

---

## 🚀 Execution Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Model & Generate Artifacts**:
   ```bash
   python src/train.py
   ```

3. **Run SHAP 'Mind Reader' Explanations**:
   ```bash
   python scripts/explain_predictions.py
   ```