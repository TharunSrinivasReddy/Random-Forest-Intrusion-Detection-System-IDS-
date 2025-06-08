# Random-Forest-Intrusion-Detection-System-IDS-
A Streamlit-based Intrusion Detection System using a Random Forest Classifier with real-time metrics, charts, and downloadable results.
# Random Forest Intrusion Detection System

A web-based Intrusion Detection System (IDS) built using **Streamlit** and **Random Forest Classifier**. Upload your network dataset (CSV), get live evaluation metrics, visualizations, and download results like predictions, feature importances, and model parameters.

## 🔧 Features

- Upload CSV dataset with `label` column (binary or string: attack/normal)
- Preprocessing with StandardScaler
- Random Forest training with custom tree count
- Evaluation: Accuracy, Precision, Recall, F1, ROC AUC
- Plots: Confusion Matrix, ROC Curve, PR Curve, Feature Importances
- Export CSVs:
  - Predictions
  - Feature importances
  - Model parameters

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

