import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.ensemble import AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from docx import Document

# ========== Load Data ==========
df = pd.read_csv('Filtered_Dataset.csv')
X = df.drop('CS', axis=1)
y = df['CS']

# === Save Styled Dataset Description as JPEG ===
desc = df.describe().T.round(2)

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

table = ax.table(
    cellText=desc.values,
    colLabels=desc.columns,
    rowLabels=desc.index,
    loc='center',
    cellLoc='center',
    colColours=['#f0f0f0'] * len(desc.columns)
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

# Make header row and index bold
for (row, col), cell in table.get_celld().items():
    if row == 0 or col == -1:
        cell.set_text_props(weight='bold')

plt.tight_layout()
plt.savefig("Dataset_Description_Styled.jpeg", dpi=300)
plt.close()

# === Save Correlation Heatmap as JPEG ===
plt.figure(figsize=(12, 10))
corr = df.corr(numeric_only=True)

# Replace NaN with 0 for full visibility
corr_filled = corr.fillna(0)

sns.heatmap(
    corr_filled,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 9, "color": "black"}
)
plt.title("Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.savefig("Correlation_Heatmap_Improved.jpeg", dpi=300)
plt.close()

# === Save Distribution Plots for Each Feature ===
for col in df.columns:
    if col != 'CS':  # Skip target
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=20, color='steelblue')
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(f"Distribution_{col}.jpeg", dpi=300)
        plt.close()

# ========== Categorical & Numerical ==========
categorical_features = ['FT', 'CT', 'SPB', 'ST']
numerical_features = [col for col in X.columns if col not in categorical_features]

# ========== Outlier Removal ==========
Q1 = X[numerical_features].quantile(0.25)
Q3 = X[numerical_features].quantile(0.75)
IQR = Q3 - Q1
X = X[~((X[numerical_features] < (Q1 - 1.5 * IQR)) | (X[numerical_features] > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
y = y.loc[X.index]

# ========== Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Preprocessing ==========
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)

# Save the preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')

# ========== Models ==========
models = {
    'XGBoost': XGBRegressor(random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(random_state=42),
    'CatBoost': CatBoostRegressor(verbose=0, random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42)
}

# ========== Evaluation Function ==========
def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    return {
        'Train R2': r2_score(y_train, train_pred),
        'Test R2': r2_score(y_test, test_pred),
        'Train RMSE': np.sqrt(np.mean((y_train - train_pred) ** 2)),
        'Test RMSE': np.sqrt(np.mean((y_test - test_pred) ** 2)),
        'Train MAE': mean_absolute_error(y_train, train_pred),
        'Test MAE': mean_absolute_error(y_test, test_pred),
        'Train MAPE': mean_absolute_percentage_error(y_train, train_pred) * 100,
        'Test MAPE': mean_absolute_percentage_error(y_test, test_pred) * 100
    }

# ========== Run and Save Models ==========
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    metrics = evaluate(model, X_train_trans, X_test_trans, y_train, y_test)
    results[name] = metrics
    joblib.dump(model, f'{name.lower()}_model.joblib')

# ========== Save Results to Word ==========
doc = Document()
doc.add_heading("Model Performance Summary", level=0)

table = doc.add_table(rows=1, cols=9)
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Model'
hdr_cells[1].text = 'Train R2'
hdr_cells[2].text = 'Test R2'
hdr_cells[3].text = 'Train RMSE'
hdr_cells[4].text = 'Test RMSE'
hdr_cells[5].text = 'Train MAE'
hdr_cells[6].text = 'Test MAE'
hdr_cells[7].text = 'Train MAPE'
hdr_cells[8].text = 'Test MAPE'

for model_name, metrics in results.items():
    row_cells = table.add_row().cells
    row_cells[0].text = model_name
    row_cells[1].text = f"{metrics['Train R2']:.3f}"
    row_cells[2].text = f"{metrics['Test R2']:.3f}"
    row_cells[3].text = f"{metrics['Train RMSE']:.2f}"
    row_cells[4].text = f"{metrics['Test RMSE']:.2f}"
    row_cells[5].text = f"{metrics['Train MAE']:.2f}"
    row_cells[6].text = f"{metrics['Test MAE']:.2f}"
    row_cells[7].text = f"{metrics['Train MAPE']:.2f}%"
    row_cells[8].text = f"{metrics['Test MAPE']:.2f}%"

doc.save("Model_Results_Summary.docx")
print("📄 Results saved to Model_Results_Summary.docx")

import shap
import os
import matplotlib.pyplot as plt

# Ensure SHAP output folder exists
os.makedirs("shap_plots", exist_ok=True)

# Load preprocessor and transform full training data
preprocessor = joblib.load("preprocessor.joblib")
X_train_trans = preprocessor.transform(X_train)
X_trans_df = pd.DataFrame(X_train_trans.toarray() if hasattr(X_train_trans, 'toarray') else X_train_trans)

# Recover feature names
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([cat_feature_names, numerical_features])

X_trans_df.columns = all_feature_names

# Load models
models = {
    'XGBoost': joblib.load('xgboost_model.joblib'),
    'LightGBM': joblib.load('lightgbm_model.joblib'),
    'CatBoost': joblib.load('catboost_model.joblib'),
    'AdaBoost': joblib.load('adaboost_model.joblib')  # will use KernelExplainer
}

# SHAP analysis
for model_name, model in models.items():
    print(f"🔍 Generating SHAP for {model_name}...")

    try:
        if model_name == 'AdaBoost':
            # Use KernelExplainer for AdaBoost
            background = shap.sample(X_trans_df, 100, random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_trans_df[:200], nsamples="auto")
        else:
            # Use TreeExplainer for tree-based models
            explainer = shap.Explainer(model, X_trans_df)
            shap_values = explainer(X_trans_df)

        # Save SHAP beeswarm plot
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"{model_name} SHAP Summary", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"shap_plots/{model_name}_shap_summary.jpeg", dpi=300)
        plt.close()
        print(f"✅ SHAP saved: shap_plots/{model_name}_shap_summary.jpeg")

    except Exception as e:
        print(f"❌ SHAP failed for {model_name}: {e}")
