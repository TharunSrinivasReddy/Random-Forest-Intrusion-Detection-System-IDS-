import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
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

# Load and preprocess the dataset
def load_and_preprocess_data(data):
    data.columns = data.columns.str.strip()

    if 'label' not in data.columns:
        st.error("Column 'label' not found in dataset.")
        st.stop()

    # Convert label to binary (0 for normal, 1 for attack)
    data['label'] = data['label'].apply(lambda x: 1 if str(x).strip().lower() == 'attack' or str(x) == '1' else 0)

    # Drop non-numeric columns except label
    label_col = data['label']
    data = data.drop('label', axis=1)
    data = data.select_dtypes(include=[np.number])
    data['label'] = label_col

    # Drop NaNs
    data = data.dropna()

    X = data.drop('label', axis=1).values
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, data.drop('label', axis=1).columns

# Train and evaluate the Random Forest model
def random_forest_model(X_train, X_test, y_train, y_test, n_estimators):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        roc_auc = 0.0

    return rf_model, y_test, y_pred, y_pred_proba, accuracy, precision, recall, f1, roc_auc

# Plotting functions
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

def plot_precision_recall_curve(y_test, y_pred_proba):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(recall_vals, precision_vals, color='purple')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    return fig

def plot_feature_importances(rf_model, feature_names):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots()
    ax.bar(range(len(importances)), importances[indices], align='center')
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(np.array(feature_names)[indices], rotation=90)
    ax.set_title("Feature Importances")
    fig.tight_layout()
    return fig

@st.cache_data
def generate_sample_csv():
    return pd.DataFrame({
        'duration': [0, 1, 2],
        'src_bytes': [491, 146, 232],
        'dst_bytes': [0, 0, 123],
        'label': ['normal', 'attack', 'normal']
    })

# Streamlit Interface
def main():
    st.set_page_config(page_title="Random Forest IDS", layout="wide")
    st.title("🔍 Random Forest Classifier - Intrusion Detection System")

    # Sidebar Parameters
    st.sidebar.header("🔧 Model Parameters")
    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100, step=10)

    st.sidebar.write("📄 Download sample CSV")
    sample_df = generate_sample_csv()
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("📥 Download Sample", csv, "sample_data.csv", "text/csv")

    st.write("### 📂 Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data(data)
            rf_model, y_test, y_pred, y_pred_proba, accuracy, precision, recall, f1, roc_auc = random_forest_model(X_train, X_test, y_train, y_test, n_estimators)

            st.write("### 📊 Evaluation Metrics")
            st.metric("Accuracy", f"{accuracy:.4f}")
            st.metric("Precision", f"{precision:.4f}")
            st.metric("Recall", f"{recall:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("ROC AUC", f"{roc_auc:.4f}")

            st.write("### 📌 Confusion Matrix")
            st.pyplot(plot_confusion_matrix(y_test, y_pred, accuracy))

            st.write("### 📈 ROC Curve")
            st.pyplot(plot_roc_curve(y_test, y_pred_proba))

            st.write("### 🧪 Precision-Recall Curve")
            st.pyplot(plot_precision_recall_curve(y_test, y_pred_proba))

            st.write("### 🧬 Feature Importances")
            st.pyplot(plot_feature_importances(rf_model, feature_names))

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

            # Download Predictions as CSV
            predictions_df = pd.DataFrame({
                "Actual": y_test,
                "Predicted": y_pred,
                "Predicted Probability": y_pred_proba
            })
            st.download_button("📥 Download Predictions (CSV)",
                               predictions_df.to_csv(index=False).encode("utf-8"),
                               file_name="rf_predictions.csv",
                               mime="text/csv")

            # Download Feature Importances as CSV
            feat_imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": rf_model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            st.download_button("📥 Download Feature Importances (CSV)",
                               feat_imp_df.to_csv(index=False).encode("utf-8"),
                               file_name="rf_feature_importance.csv",
                               mime="text/csv")

            # Download Model Parameters
            params_df = pd.DataFrame.from_dict(rf_model.get_params(), orient='index', columns=['Value'])
            st.download_button("📥 Download Model Parameters (CSV)",
                               params_df.to_csv().encode('utf-8'),
                               file_name="rf_model_params.csv",
                               mime="text/csv")

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.text(traceback.format_exc())

if __name__ == '__main__':
    main()
