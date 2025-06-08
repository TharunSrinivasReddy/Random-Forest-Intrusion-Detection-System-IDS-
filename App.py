import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import io
import pickle

# Load and preprocess the dataset
def load_and_preprocess_data(data):
    data.columns = data.columns.str.strip()

    if 'label' not in data.columns:
        st.error("Column 'label' not found in dataset.")
        st.stop()

    # Ensure binary format for labels (0 for Normal, 1 for Attack)
    if set(data['label'].unique()) != {0, 1}:
        data['label'] = data['label'].apply(lambda x: 1 if str(x).lower() == 'attack' else 0)

    X = data.drop('label', axis=1).values
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.drop('label', axis=1).columns, scaler

# Train and evaluate the Random Forest model with fix for single-class case
def random_forest_model(X_train, X_test, y_train, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    if len(rf_model.classes_) == 2:
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    else:
        # Only one class trained, so probabilities are trivial
        single_class = rf_model.classes_[0]
        if single_class == 1:
            y_pred_proba = np.ones_like(y_pred, dtype=float)
        else:
            y_pred_proba = np.zeros_like(y_pred, dtype=float)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = 0.0

    return rf_model, y_test, y_pred, y_pred_proba, accuracy, precision, recall, f1, roc_auc

# Plot confusion matrix with color mapping
def plot_confusion_matrix(y_test, y_pred, accuracy):
    if accuracy < 0.5:
        cmap = 'Reds'
    elif accuracy < 0.75:
        cmap = 'YlOrBr'
    else:
        cmap = 'Greens'

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
    return fig

# Plot ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label='ROC Curve', color='blue')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    return fig

# Plot Precision-Recall Curve
def plot_precision_recall_curve(y_test, y_pred_proba):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(recall_vals, precision_vals, color='purple')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    return fig

# Plot Feature Importances
def plot_feature_importances(rf_model, feature_names):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(importances)), importances[indices], align='center')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_names)[indices], rotation=90)
    ax.set_title("Feature Importances")
    fig.tight_layout()
    return fig

# Streamlit interface
def main():
    st.title("🔍 Random Forest Classifier - Intrusion Detection System")

    st.write("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data(data)
            rf_model, y_test, y_pred, y_pred_proba, accuracy, precision, recall, f1, roc_auc = random_forest_model(X_train, X_test, y_train, y_test)

            st.write("### 📊 Evaluation Metrics")
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("ROC AUC", f"{roc_auc:.4f}")

            # Confusion Matrix
            st.write("### 📌 Confusion Matrix")
            fig_cm = plot_confusion_matrix(y_test, y_pred, accuracy)
            st.pyplot(fig_cm)

            # ROC Curve
            st.write("### 📈 ROC Curve")
            fig_roc = plot_roc_curve(y_test, y_pred_proba)
            st.pyplot(fig_roc)

            # Precision-Recall Curve
            st.write("### 🧪 Precision-Recall Curve")
            fig_pr = plot_precision_recall_curve(y_test, y_pred_proba)
            st.pyplot(fig_pr)

            # Feature Importance
            st.write("### 🧬 Feature Importances")
            fig_fi = plot_feature_importances(rf_model, feature_names)
            st.pyplot(fig_fi)

            # Summary
            st.write("### 💡 Performance Summary")
            if accuracy < 0.4 or precision == 0.0 or recall == 0.0:
                st.error("🔴 Critical Danger: Very poor model performance.")
            elif accuracy < 0.5 or f1 < 0.3:
                st.warning("🟠 Danger: Model performance is low.")
            elif accuracy < 0.6 or roc_auc < 0.6:
                st.info("🟡 Needs Improvement: Predictions exist but are weak.")
            elif accuracy >= 0.6 and f1 >= 0.5:
                st.success("🟢 Acceptable Performance.")
            else:
                st.success("✅ Good Performance!")

            # Model download option (pickle file)
            buffer = io.BytesIO()
            pickle.dump({
                'model': rf_model,
                'scaler': scaler,
                'feature_names': feature_names
            }, buffer)
            buffer.seek(0)
            st.download_button(
                label="📥 Download Trained Model (pickle)",
                data=buffer,
                file_name="random_forest_model.pkl",
                mime="application/octet-stream"
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.text(traceback.format_exc())

if __name__ == '__main__':
    main()
