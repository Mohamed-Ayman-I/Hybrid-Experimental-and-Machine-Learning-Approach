# ================================ Imports ================================
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# Load your dataset
df = pd.read_csv('Filtered_Dataset.csv')

# Define features and target
X = df.drop('CS', axis=1)
y = df['CS']

# Identify Categorical and Numerical Features
categorical_features = ['FT', 'CT', 'SPB', 'ST']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Outlier Removal
Q1 = X[numerical_features].quantile(0.25)
Q3 = X[numerical_features].quantile(0.75)
IQR = Q3 - Q1
X = X[~((X[numerical_features] < (Q1 - 1.5 * IQR)) | (X[numerical_features] > (Q3 + 1.5 * IQR))).any(axis=1)].copy()
y = y.loc[X.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling numerical features and one-hot encoding categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

preprocessor.fit(X_train)

X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Save preprocessor
joblib.dump(preprocessor, 'preprocessor.joblib')

# Evaluation Function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'Train MAPE': mean_absolute_percentage_error(y_train, y_train_pred),
        'Test MAPE': mean_absolute_percentage_error(y_test, y_test_pred),
        'Train MSE': mean_squared_error(y_train, y_train_pred),
        'Test MSE': mean_squared_error(y_test, y_test_pred),
        'Train RMSE': mean_squared_error(y_train, y_train_pred) ** 0.5,
        'Test RMSE': mean_squared_error(y_test, y_test_pred) ** 0.5,
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Test MAE': mean_absolute_error(y_test, y_test_pred),
        'Train R2': r2_score(y_train, y_train_pred),
        'Test R2': r2_score(y_test, y_test_pred)
    }
    return metrics

# Define models with default hyperparameters
models = {
    'decision_tree': DecisionTreeRegressor(),
    'random_forest': RandomForestRegressor(),
    'extra_trees': ExtraTreesRegressor()
}

# Evaluate and save all models
results = {}
for name, model in models.items():
    print(f'Training and evaluating {name}...')
    metrics = evaluate_model(model, X_train_transformed, X_test_transformed, y_train, y_test)
    results[name] = metrics
    joblib.dump(model, f'{name}_model.joblib')

# Create directory for SHAP plots
os.makedirs("shap_plots", exist_ok=True)

# Inverse-transform feature names for SHAP
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([cat_feature_names, numerical_features])
X_train_array = X_train_transformed if isinstance(X_train_transformed, np.ndarray) else X_train_transformed.toarray()

# SHAP analysis per model
for name, model in models.items():
    try:
        print(f"\nGenerating SHAP plot for {name}...")

        explainer = shap.Explainer(model, X_train_array, feature_names=all_feature_names)
        shap_values = explainer(X_train_array)

        # Plot SHAP summary
        plt.figure()
        shap.plots.beeswarm(shap_values, show=False)
        plt.title(f"SHAP Summary: {name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"shap_plots/{name}_shap_summary.jpeg", dpi=300, format='jpeg')
        plt.close()
        print(f"✅ SHAP plot saved for {name}.")

    except Exception as e:
        print(f"[❌ ERROR] Could not generate SHAP for {name}: {e}")

# Display results
for model_name, metrics in results.items():
    print(f'\nModel: {model_name}')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')
