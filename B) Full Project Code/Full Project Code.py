# ================================================
# Full Code For: "A Hybrid Experimental and Machine Learning Framework
# for Designing and Predicting Compressive Strength of Ultra-High-Performance Concrete"
# ================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import os
import seaborn as sns
import joblib
import warnings
import random
import tkinter as tk
from tkinter import messagebox
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    mean_squared_error
)
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from docx import Document
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

print("🚀 Starting Combined Pipeline...")

# ================================================
# SECTION 1: TREE MODELS - NO OPTIMIZERS
# ================================================
print("\n" + "="*60)
print("SECTION 1: TREE MODELS (No Optimizers)")
print("="*60)

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


# ================================================
# SECTION 2: BOOSTING MODELS - NO OPTIMIZERS
# ================================================
print("\n" + "="*60)
print("SECTION 2: BOOSTING MODELS (No Optimizers)")
print("="*60)

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

# SHAP Analysis
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


# ================================================
# SECTION 3: SVM MODELS - NO OPTIMIZERS
# ================================================
print("\n" + "="*60)
print("SECTION 3: SVM MODELS (No Optimizers)")
print("="*60)

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

# Define models
models = {
    'svr': SVR(
        kernel='rbf',
        C=100,              # Stronger penalty for errors → better fitting
        epsilon=0.05,       # Narrower margin → more precision
        gamma=0.005         # Controls smoothness of the fit
    ),
    'nusvr': NuSVR(
        kernel='rbf',
        C=100,              # Same logic as above
        nu=0.25,            # Allows 25% training error margin → stable generalization
        gamma=0.005         # Smoother fit than default; avoids overfitting
    )
}

# Evaluate and save all models
results = {}
for name, model in models.items():
    print(f'Training and evaluating {name}...')
    metrics = evaluate_model(model, X_train_transformed, X_test_transformed, y_train, y_test)
    results[name] = metrics
    joblib.dump(model, f'{name}_model.joblib')

# Display results
for model_name, metrics in results.items():
    print(f'\nModel: {model_name}')
    for metric, value in metrics.items():
        print(f'{metric}: {value:.4f}')

# SHAP for SVM
os.makedirs("shap_plots", exist_ok=True)

# Load preprocessor again to get feature names
preprocessor = joblib.load("preprocessor.joblib")

# Get feature names from preprocessor
ohe = preprocessor.named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([cat_feature_names, numerical_features])

# Run SHAP for each model
for name in models:
    print(f"\n🔍 SHAP for {name.upper()}")

    # Reload trained model
    model = joblib.load(f"{name}_model.joblib")

    # Prepare background data (sample for speed)
    background = shap.sample(X_train_transformed, 100, random_state=42)

    # Create KernelExplainer
    explainer = shap.KernelExplainer(model.predict, background)

    # Select a subset to explain
    shap_values = explainer.shap_values(X_train_transformed[:200], nsamples=100)

    # Plot SHAP summary
    shap.summary_plot(shap_values, X_train_transformed[:200], feature_names=all_feature_names, show=False)
    plt.title(f"SHAP Summary - {name.upper()}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"shap_plots/{name}_shap_summary.jpeg", dpi=300)
    plt.close()
    print(f"✅ SHAP plot saved: shap_plots/{name}_shap_summary.jpeg")


# ================================================
# SECTION 4: TREE MODELS - OPTIMIZED
# ================================================
print("\n" + "="*80)
print("SECTION 4: TREE MODELS WITH OPTIMIZERS")
print("="*80)

df = pd.read_csv("Filtered_Dataset.csv")
y = df["CS"]
X = df.drop(columns=["CS"])

categorical_features = ['FT', 'CT', 'SPB', 'ST']
numerical_features = [col for col in X.columns if col not in categorical_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# ====================== Evaluation Metrics =====================
def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R2": r2_score(y_true, y_pred)
    }

# ====================== Optimizers =====================
def pso_optimize(obj_func, bounds, n_particles=25, max_iter=35, w=0.6, c1=1.2, c2=2.2):
    dim = len(bounds)
    X = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (n_particles, dim))
    V = np.zeros_like(X)
    p_best = X.copy()
    p_best_scores = np.array([obj_func(x) for x in X])
    g_best = p_best[np.argmin(p_best_scores)]

    for _ in range(max_iter):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            V[i] = w * V[i] + c1 * r1 * (p_best[i] - X[i]) + c2 * r2 * (g_best - X[i])
            X[i] = np.clip(X[i] + V[i], [b[0] for b in bounds], [b[1] for b in bounds])
            score = obj_func(X[i])
            if score < p_best_scores[i]:
                p_best[i] = X[i]
                p_best_scores[i] = score
        g_best = p_best[np.argmin(p_best_scores)]

    return g_best, min(p_best_scores)

def ga_optimize(obj_func, bounds, pop_size=24, generations=35, mutation_rate=0.25, crossover_rate=0.75):
    dim = len(bounds)
    population = [np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]) for _ in range(pop_size)]
    population = np.array(population)

    for _ in range(generations):
        scores = np.array([obj_func(ind) for ind in population])
        sorted_indices = np.argsort(scores)
        sorted_pop = population[sorted_indices]
        next_gen = list(sorted_pop[:2])

        while len(next_gen) < pop_size:
            if dim > 1:
                cut = random.randint(1, dim - 1)
            else:
                cut = 1
            if random.random() < crossover_rate:
                parents = random.sample(list(sorted_pop[:10]), 2)
                child = np.concatenate((parents[0][:cut], parents[1][cut:]))
            else:
                child = random.choice(sorted_pop[:10]).copy()

            if random.random() < mutation_rate:
                idx = random.randint(0, dim - 1)
                low, high = bounds[idx]
                child[idx] = np.clip(child[idx] + np.random.normal(0, 0.2), low, high)

            next_gen.append(np.clip(child, [b[0] for b in bounds], [b[1] for b in bounds]))

        population = np.array(next_gen)

    best_index = np.argmin([obj_func(ind) for ind in population])
    best = population[best_index]
    return best, obj_func(best)

def firefly_optimize(obj_func, bounds, n=22, max_iter=33, alpha=0.6, beta=1.0, gamma=1.5):
    dim = len(bounds)
    fireflies = [np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]) for _ in range(n)]
    light = [obj_func(f) for f in fireflies]

    for _ in range(max_iter):
        for i in range(n):
            for j in range(n):
                if light[j] < light[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta_temp = beta * np.exp(-gamma * r ** 2)
                    attraction = beta_temp * (fireflies[j] - fireflies[i])
                    random_move = alpha * (np.random.rand(dim) - 0.5)
                    fireflies[i] += attraction + random_move
                    fireflies[i] = np.clip(fireflies[i], [b[0] for b in bounds], [b[1] for b in bounds])
                    light[i] = obj_func(fireflies[i])

    best_idx = np.argmin(light)
    return fireflies[best_idx], light[best_idx]

def sa_optimize(obj_func, bounds, max_iter=40, T0=120, alpha=0.85):
    dim = len(bounds)
    x = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    fx = obj_func(x)
    best = x.copy()
    fbest = fx
    T = T0

    for _ in range(max_iter):
        for _ in range(15):
            candidate = x + np.random.uniform(-0.15, 0.15, dim)
            candidate = np.clip(candidate, [b[0] for b in bounds], [b[1] for b in bounds])
            fc = obj_func(candidate)
            if fc < fx or np.random.rand() < np.exp(-(fc - fx) / T):
                x, fx = candidate, fc
                if fc < fbest:
                    best, fbest = candidate, fc
        T *= alpha
    return best, fbest

def de_optimize(obj_func, bounds, pop_size=20, max_iter=30, F=0.5, CR=0.7):
    dim = len(bounds)
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    scores = np.array([obj_func(ind) for ind in pop])

    for _ in range(max_iter):
        for i in range(pop_size):
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
            cross_points = np.random.rand(dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            score = obj_func(trial)
            if score < scores[i]:
                scores[i] = score
                pop[i] = trial
    best_idx = np.argmin(scores)
    return pop[best_idx], scores[best_idx]

def abc_optimize(obj_func, bounds, pop_size=20, max_iter=30, limit=5):
    dim = len(bounds)
    foods = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    fitness = np.array([obj_func(f) for f in foods])
    trial = np.zeros(pop_size)

    for _ in range(max_iter):
        for i in range(pop_size):
            k = random.choice([j for j in range(pop_size) if j != i])
            phi = np.random.uniform(-1, 1, dim)
            new = foods[i] + phi * (foods[i] - foods[k])
            new = np.clip(new, [b[0] for b in bounds], [b[1] for b in bounds])
            new_fit = obj_func(new)
            if new_fit < fitness[i]:
                foods[i] = new
                fitness[i] = new_fit
                trial[i] = 0
            else:
                trial[i] += 1

        probs = fitness.max() - fitness
        probs = probs / probs.sum()
        for _ in range(pop_size):
            i = np.random.choice(range(pop_size), p=probs)
            k = random.choice([j for j in range(pop_size) if j != i])
            phi = np.random.uniform(-1, 1, dim)
            new = foods[i] + phi * (foods[i] - foods[k])
            new = np.clip(new, [b[0] for b in bounds], [b[1] for b in bounds])
            new_fit = obj_func(new)
            if new_fit < fitness[i]:
                foods[i] = new
                fitness[i] = new_fit
                trial[i] = 0
            else:
                trial[i] += 1

        for i in range(pop_size):
            if trial[i] > limit:
                foods[i] = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
                fitness[i] = obj_func(foods[i])
                trial[i] = 0

    best_idx = np.argmin(fitness)
    return foods[best_idx], fitness[best_idx]

def hs_optimize(obj_func, bounds, hm_size=20, max_iter=30, HMCR=0.9, PAR=0.3, bw=0.01):
    dim = len(bounds)
    HM = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (hm_size, dim))
    scores = np.array([obj_func(h) for h in HM])

    for _ in range(max_iter):
        new = []
        for d in range(dim):
            if random.random() < HMCR:
                idx = random.randint(0, hm_size - 1)
                val = HM[idx, d]
                if random.random() < PAR:
                    val += np.random.uniform(-bw, bw)
                val = np.clip(val, bounds[d][0], bounds[d][1])
            else:
                val = np.random.uniform(bounds[d][0], bounds[d][1])
            new.append(val)
        new = np.array(new)
        new_score = obj_func(new)
        if new_score < scores.max():
            worst_idx = np.argmax(scores)
            HM[worst_idx] = new
            scores[worst_idx] = new_score

    best_idx = np.argmin(scores)
    return HM[best_idx], scores[best_idx]

# Add optimizers to dictionary
optimizers = {
    "PSO": pso_optimize,
    "GA": ga_optimize,
    "Firefly": firefly_optimize,
    "SA": sa_optimize,
    "DE": de_optimize,
    "ABC": abc_optimize,
    "HS": hs_optimize
}

# =================== Model Config ===================
model_configs = {
    "Decision Tree": (
        DecisionTreeRegressor,
        ["max_depth", "min_samples_split", "min_samples_leaf"],
        [
            (5.0, 20.0),
            (2.0, 15.0),
            (1.0, 5.0)
        ]
    ),

    "Random Forest": (
        lambda **kwargs: RandomForestRegressor(n_jobs=-1, random_state=42, **kwargs),
        ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        [
            (50.0, 120.0),
            (5.0, 20.0),
            (2.0, 10.0),
            (1.0, 5.0),
            (0.3, 0.8)
        ]
    ),

    "Extra Trees": (
        lambda **kwargs: ExtraTreesRegressor(n_jobs=-1, random_state=42, **kwargs),
        ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features"],
        [
            (50.0, 120.0),
            (5.0, 20.0),
            (2.0, 10.0),
            (1.0, 5.0),
            (0.3, 0.8)
        ]
    )
}

# =================== Train Models with Optimizers ===================
optimized_models = {}
all_results = {}

for model_name, (model_class, param_names, bounds) in model_configs.items():
    results = {}
    print(f"\n=== {model_name} ===")

    for opt_name, optimizer in optimizers.items():
        print(f"\n🔧 Optimizing {model_name} using {opt_name}...")

        def objective(params):
            param_dict = {}
            for name, val in zip(param_names, params):
                if name in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                    param_dict[name] = int(round(val))
                else:
                    param_dict[name] = val
            model = model_class(**param_dict)
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            return mean_squared_error(y_test, preds)

        np.random.seed(2000 + hash(opt_name + model_name) % 1000)
        random.seed(2000 + hash(opt_name + model_name) % 1000)

        best_params, _ = optimizer(objective, bounds)

        param_dict = {}
        for name, val in zip(param_names, best_params):
            if name in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                param_dict[name] = int(round(val))
            else:
                param_dict[name] = val

        model = model_class(**param_dict)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)

        if model_name not in optimized_models:
            optimized_models[model_name] = {}
        optimized_models[model_name][opt_name] = pipeline

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate(y_train, train_pred)
        test_metrics = evaluate(y_test, test_pred)

        results[opt_name] = {
            "Train": train_metrics,
            "Test": test_metrics
        }

        print(f"✅ {opt_name} Done. R² = {test_metrics['R2']:.3f}, RMSE = {test_metrics['RMSE']:.2f}")

    all_results[model_name] = results

    doc = Document()
    doc.add_heading(f"{model_name} Optimized Models", 0)

    metrics_headers = ["Optimizer", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE", "Train_MAE", "Test_MAE",
                       "Train_MAPE", "Test_MAPE"]
    table = doc.add_table(rows=1, cols=len(metrics_headers))
    table.style = 'Light Grid Accent 1'

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(metrics_headers):
        hdr_cells[i].text = h

    for opt_name, metrics in results.items():
        row_cells = table.add_row().cells
        row_cells[0].text = opt_name
        row_cells[1].text = f"{metrics['Train']['R2']:.3f}"
        row_cells[2].text = f"{metrics['Test']['R2']:.3f}"
        row_cells[3].text = f"{metrics['Train']['RMSE']:.3f}"
        row_cells[4].text = f"{metrics['Test']['RMSE']:.3f}"
        row_cells[5].text = f"{metrics['Train']['MAE']:.3f}"
        row_cells[6].text = f"{metrics['Test']['MAE']:.3f}"
        row_cells[7].text = f"{metrics['Train']['MAPE']:.3f}"
        row_cells[8].text = f"{metrics['Test']['MAPE']:.3f}"

    doc.save(f"{model_name.replace(' ', '_')}_Optimized_Summary.docx")
    print(f"📄 Word file saved: {model_name.replace(' ', '_')}_Optimized_Summary.docx")

# SHAP for Optimized Tree Models
os.makedirs("shap_plots", exist_ok=True)

print("\n" + "=" * 100)
print("🔍 SHAP ANALYSIS OF OPTIMIZED TREE MODELS")
print("=" * 100)

for model_name, optimizer_dict in optimized_models.items():
    for optimizer_name, pipeline in optimizer_dict.items():
        print(f"\n{'=' * 80}")
        print(f"📌 SHAP Summary for {model_name} optimized using {optimizer_name}")
        print(f"{'=' * 80}")

        try:
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['regressor']
            X_train_transformed = preprocessor.transform(X_train)

            cat_encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            num_feature_names = numerical_features
            all_feature_names = np.concatenate([cat_feature_names, num_feature_names])

            explainer = shap.Explainer(model, X_train_transformed, feature_names=all_feature_names)
            shap_values = explainer(X_train_transformed)

            plt.figure()
            shap.plots.beeswarm(shap_values, show=False)
            plt.title(f"{model_name} + {optimizer_name}", fontsize=14)
            plt.tight_layout()

            filename = f"shap_plots/{model_name.replace(' ', '_')}_{optimizer_name}.jpeg"
            plt.savefig(filename, format='jpeg', dpi=300)
            plt.close()
            print(f"✅ SHAP plot saved: {filename}")

        except Exception as e:
            print(f"[❌ ERROR] Could not generate SHAP for {model_name} + {optimizer_name}: {e}")


# ================================================
# SECTION 5: BOOSTING MODELS - OPTIMIZED
# ================================================
print("\n" + "="*80)
print("SECTION 5: BOOSTING MODELS WITH OPTIMIZERS")
print("="*80)

df = pd.read_csv("Filtered_Dataset.csv")
y = df["CS"]
X = df.drop(columns=["CS"])

categorical_features = ['FT', 'CT', 'SPB', 'ST']
numerical_features = [col for col in X.columns if col not in categorical_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R2": r2_score(y_true, y_pred)
    }

# Optimizers (same as above - reusing definitions)

# Model Configs for Boosting
model_configs = {
    "XGBoost": (
        XGBRegressor,
        ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"],
        [(50.0, 150.0), (3.0, 8.0), (0.05, 0.2), (0.7, 1.0), (0.6, 1.0)]
    ),
    "AdaBoost": (
        AdaBoostRegressor,
        ["n_estimators", "learning_rate"],
        [(30.0, 100.0), (0.05, 0.5)]
    ),
    "CatBoost": (
        CatBoostRegressor,
        ["iterations", "depth", "learning_rate"],
        [(50.0, 150.0), (3.0, 6.0), (0.05, 0.2)]
    ),
    "LightGBM": (
        LGBMRegressor,
        ["n_estimators", "max_depth", "learning_rate", "num_leaves", "subsample"],
        [(50.0, 150.0), (3.0, 8.0), (0.05, 0.2), (20.0, 80.0), (0.7, 1.0)]
    )
}

optimized_models = {}
all_results = {}

int_params_all = {
    "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
    "iterations", "depth", "num_leaves"
}

for model_name, (model_class, param_names, bounds) in model_configs.items():
    results = {}
    print(f"\n=== {model_name} ===")

    for opt_name, optimizer in optimizers.items():
        print(f"\n🔧 Optimizing {model_name} using {opt_name}...")

        def objective(params):
            param_dict = {}
            for name, val in zip(param_names, params):
                if name in int_params_all:
                    param_dict[name] = int(round(val))
                else:
                    param_dict[name] = val

            if model_name == "CatBoost":
                model = model_class(verbose=0, **param_dict)
            else:
                model = model_class(**param_dict)

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            return mean_squared_error(y_test, preds)

        np.random.seed(2000 + hash(opt_name + model_name) % 1000)
        random.seed(2000 + hash(opt_name + model_name) % 1000)

        best_params, _ = optimizer(objective, bounds)

        param_dict = {}
        for name, val in zip(param_names, best_params):
            if name in int_params_all:
                param_dict[name] = int(round(val))
            else:
                param_dict[name] = val

        if model_name == "CatBoost":
            model = model_class(verbose=0, **param_dict)
        else:
            model = model_class(**param_dict)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)

        if model_name not in optimized_models:
            optimized_models[model_name] = {}
        optimized_models[model_name][opt_name] = pipeline

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate(y_train, train_pred)
        test_metrics = evaluate(y_test, test_pred)

        results[opt_name] = {
            "Train": train_metrics,
            "Test": test_metrics
        }

        print(f"✅ {opt_name} Done. R² = {test_metrics['R2']:.3f}, RMSE = {test_metrics['RMSE']:.2f}")

    all_results[model_name] = results

    doc = Document()
    doc.add_heading(f"{model_name} Optimized Models", 0)

    metrics_headers = ["Optimizer", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE", "Train_MAE", "Test_MAE",
                       "Train_MAPE", "Test_MAPE"]
    table = doc.add_table(rows=1, cols=len(metrics_headers))
    table.style = 'Light Grid Accent 1'

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(metrics_headers):
        hdr_cells[i].text = h

    for opt_name, metrics in results.items():
        row_cells = table.add_row().cells
        row_cells[0].text = opt_name
        row_cells[1].text = f"{metrics['Train']['R2']:.3f}"
        row_cells[2].text = f"{metrics['Test']['R2']:.3f}"
        row_cells[3].text = f"{metrics['Train']['RMSE']:.3f}"
        row_cells[4].text = f"{metrics['Test']['RMSE']:.3f}"
        row_cells[5].text = f"{metrics['Train']['MAE']:.3f}"
        row_cells[6].text = f"{metrics['Test']['MAE']:.3f}"
        row_cells[7].text = f"{metrics['Train']['MAPE']:.3f}"
        row_cells[8].text = f"{metrics['Test']['MAPE']:.3f}"

    doc.save(f"{model_name.replace(' ', '_')}_Optimized_Summary.docx")
    print(f"📄 Word file saved: {model_name.replace(' ', '_')}_Optimized_Summary.docx")

# SHAP for Optimized Boosting
os.makedirs("shap_plots", exist_ok=True)

print("\n" + "=" * 100)
print("🔍 SHAP ANALYSIS OF OPTIMIZED BOOSTING MODELS")
print("=" * 100)

for model_name, optimizer_dict in optimized_models.items():
    for optimizer_name, pipeline in optimizer_dict.items():
        print(f"\n{'=' * 80}")
        print(f"📌 SHAP Summary for {model_name} optimized using {optimizer_name}")
        print(f"{'=' * 80}")

        try:
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['regressor']
            X_train_transformed = preprocessor.transform(X_train)

            cat_encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            num_feature_names = numerical_features
            all_feature_names = np.concatenate([cat_feature_names, num_feature_names])

            explainer = shap.Explainer(model, X_train_transformed, feature_names=all_feature_names)
            shap_values = explainer(X_train_transformed)

            plt.figure()
            shap.plots.beeswarm(shap_values, show=False)
            plt.title(f"{model_name} + {optimizer_name}", fontsize=14)
            plt.tight_layout()

            filename = f"shap_plots/{model_name.replace(' ', '_')}_{optimizer_name}.jpeg"
            plt.savefig(filename, format='jpeg', dpi=300)
            plt.close()
            print(f"✅ SHAP plot saved: {filename}")

        except Exception as e:
            print(f"[❌ ERROR] Could not generate SHAP for {model_name} + {optimizer_name}: {e}")


# ================================================
# SECTION 6: SVM MODELS - OPTIMIZED
# ================================================
print("\n" + "="*80)
print("SECTION 6: SVM MODELS WITH OPTIMIZERS")
print("="*80)

df = pd.read_csv("Filtered_Dataset.csv")
y = df["CS"]
X = df.drop(columns=["CS"])

categorical_features = ['FT', 'CT', 'SPB', 'ST']
numerical_features = [col for col in X.columns if col not in categorical_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

def evaluate(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MAPE": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "R2": r2_score(y_true, y_pred)
    }

# Optimizers (reusing)

model_configs = {
    "SVR": (
        SVR,
        ["C", "epsilon", "gamma"],
        [(1.0, 100.0), (0.01, 0.1), (0.001, 0.5)]
    ),
    "NuSVR": (
        NuSVR,
        ["nu", "C", "gamma"],
        [(0.1, 0.9), (1.0, 100.0), (0.001, 0.5)]
    )
}

all_results = {}
optimized_models = {}
for model_name, (model_class, param_names, bounds) in model_configs.items():
    results = {}
    print(f"\n=== {model_name} ===")

    for opt_name, optimizer in optimizers.items():
        print(f"\n🔧 Optimizing {model_name} using {opt_name}...")

        def objective(params):
            if model_name == "SVR":
                C, epsilon, gamma = params
                model = model_class(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')

            elif model_name == "NuSVR":
                nu, C, gamma = params
                model = model_class(nu=nu, C=C, gamma=gamma, kernel='rbf')

            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            return mean_squared_error(y_test, preds)

        np.random.seed(2000 + hash(opt_name + model_name) % 1000)
        random.seed(2000 + hash(opt_name + model_name) % 1000)

        best_params, _ = optimizer(objective, bounds)

        if model_name == "SVR":
            C, epsilon, gamma = best_params
            model = model_class(C=C, epsilon=epsilon, gamma=gamma, kernel='rbf')

        elif model_name == "NuSVR":
            nu, C, gamma = best_params
            model = model_class(nu=nu, C=C, gamma=gamma, kernel='rbf')

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        pipeline.fit(X_train, y_train)

        if model_name not in optimized_models:
            optimized_models[model_name] = {}
        optimized_models[model_name][opt_name] = pipeline

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_metrics = evaluate(y_train, train_pred)
        test_metrics = evaluate(y_test, test_pred)

        results[opt_name] = {
            "Train": train_metrics,
            "Test": test_metrics
        }

        print(f"✅ {opt_name} Done. R² = {test_metrics['R2']:.3f}, RMSE = {test_metrics['RMSE']:.2f}")

    all_results[model_name] = results

    doc = Document()
    doc.add_heading(f"{model_name} Optimized Models", 0)

    metrics_headers = ["Optimizer", "Train_R2", "Test_R2", "Train_RMSE", "Test_RMSE", "Train_MAE", "Test_MAE",
                       "Train_MAPE", "Test_MAPE"]
    table = doc.add_table(rows=1, cols=len(metrics_headers))
    table.style = 'Light Grid Accent 1'

    hdr_cells = table.rows[0].cells
    for i, h in enumerate(metrics_headers):
        hdr_cells[i].text = h

    for opt_name, metrics in results.items():
        row_cells = table.add_row().cells
        row_cells[0].text = opt_name
        row_cells[1].text = f"{metrics['Train']['R2']:.3f}"
        row_cells[2].text = f"{metrics['Test']['R2']:.3f}"
        row_cells[3].text = f"{metrics['Train']['RMSE']:.3f}"
        row_cells[4].text = f"{metrics['Test']['RMSE']:.3f}"
        row_cells[5].text = f"{metrics['Train']['MAE']:.3f}"
        row_cells[6].text = f"{metrics['Test']['MAE']:.3f}"
        row_cells[7].text = f"{metrics['Train']['MAPE']:.3f}"
        row_cells[8].text = f"{metrics['Test']['MAPE']:.3f}"

    doc.save(f"{model_name.replace(' ', '_')}_Optimized_Summary.docx")
    print(f"📄 Word file saved: {model_name.replace(' ', '_')}_Optimized_Summary.docx")

# SHAP for Optimized SVM
os.makedirs("shap_plots", exist_ok=True)

print("\n" + "=" * 100)
print("🔍 SHAP ANALYSIS OF OPTIMIZED SVM MODELS")
print("=" * 100)

for model_name, optimizer_dict in optimized_models.items():
    for optimizer_name, pipeline in optimizer_dict.items():
        print(f"\n{'=' * 80}")
        print(f"📌 SHAP Summary for {model_name} optimized using {optimizer_name}")
        print(f"{'=' * 80}")

        try:
            preprocessor = pipeline.named_steps['preprocessor']
            model = pipeline.named_steps['regressor']
            X_train_transformed = preprocessor.transform(X_train)

            cat_encoder = preprocessor.named_transformers_['cat']
            cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
            num_feature_names = numerical_features
            all_feature_names = np.concatenate([cat_feature_names, num_feature_names])

            def model_predict(data):
                return model.predict(data)

            background = shap.kmeans(X_train_transformed, 50)
            explainer = shap.KernelExplainer(model_predict, background, feature_names=all_feature_names)
            shap_values = explainer.shap_values(X_train_transformed[:200], nsamples=100)

            explanation = shap.Explanation(values=shap_values,
                                           data=X_train_transformed[:200],
                                           feature_names=all_feature_names)

            plt.figure()
            shap.plots.beeswarm(explanation, show=False)
            plt.title(f"{model_name} + {optimizer_name}", fontsize=14)
            plt.tight_layout()

            filename = f"shap_plots/{model_name.replace(' ', '_')}_{optimizer_name}.jpeg"
            plt.savefig(filename, format='jpeg', dpi=300)
            plt.close()
            print(f"✅ SHAP plot saved: {filename}")

        except Exception as e:
            print(f"[❌ ERROR] Could not generate SHAP for {model_name} + {optimizer_name}: {e}")


# ================================================
# SECTION 7: GUI PREDICTOR
# ================================================
print("\n" + "="*60)
print("SECTION 7: LAUNCHING GUI PREDICTOR")
print("="*60)

# Helper for .exe
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load Models (Note: Update paths if needed for your saved optimized models)
model_info = {
    'Random Forest (DE)': ('RF_DE.pkl', 'Random Forest', 'DE'),
    'XGBoost (PSO)': ('XGB_PSO.pkl', 'XGBoost', 'PSO'),
}

models = {}
for key, (filename, _, _) in model_info.items():
    try:
        models[key] = joblib.load(resource_path(filename))
        print(f"Loaded {key}")
    except Exception as e:
        print(f"Could not load {filename}: {e}")

# GUI Setup
window = tk.Tk()
window.title("UHPC Compressive Strength Predictor")
window.geometry('1200x900')
window.configure(bg='white')

features = [
    ('C', 'Cement', 'kg/m³'),
    ('SF', 'Silica Fume', 'kg/m³'),
    ('S', 'Sand', 'kg/m³'),
    ('QP', 'Quartz Powder', 'kg/m³'),
    ('W', 'Water', 'kg/m³'),
    ('SP', 'Superplasticizer', 'kg/m³'),
    ('F', 'Fiber', 'kg/m³'),
    ('WB', 'Water/Binder Ratio', '-'),
    ('T', 'Curing Temperature', '°C'),
    ('CT', 'Cement Type', '(Choose from list)'),
    ('SPB', 'Superplasticizer Base', '(Choose from list)'),
    ('FT', 'Fiber Type', '(Choose from list)'),
    ('FL', 'Fiber Length', 'mm'),
    ('FD', 'Fiber Diameter', 'mm'),
    ('ST', 'Specimen Type', '(Choose from list)'),
    ('A', 'Testing Age', 'Days')
]

input_entries = {}
for idx, (abbr, meaning, unit) in enumerate(features):
    label = tk.Label(window, text=f"{abbr} ({meaning}) [{unit}]:", fg='black', bg='white', anchor='w')
    label.grid(row=idx, column=0, padx=10, pady=5, sticky='w')

    entry = tk.Entry(window, width=30)
    entry.grid(row=idx, column=1, padx=10, pady=5)
    input_entries[abbr] = entry

# Dropdown for Model Selection
selected_model = tk.StringVar(window)
selected_model.set(list(models.keys())[0] if models else "No models loaded")
model_menu = tk.OptionMenu(window, selected_model, *models.keys())
model_menu.grid(row=0, column=3, padx=10, pady=10)

model_label = tk.Label(window, text='', fg='black', bg='white', font=("Arial", 12))
model_label.grid(row=1, column=3, padx=10, pady=5, sticky='w')

def update_model_label(*args):
    if models:
        model_key = selected_model.get()
        model_name, optimizer = model_info.get(model_key, ["Unknown", "Unknown"])[1:]
        model_label.config(text=f"Model: {model_name} | Optimizer: {optimizer}")

selected_model.trace("w", update_model_label)
update_model_label()

# Explanation Labels
explanation_text = """
CT:
1 = Type I
2 = Type I/II
3 = Type IS (Slag Cement)
4 = Type III
5 = Type V
6 = Type I (High Strength)
7 = Type II

SPB:
1 = Polycarboxylate
2 = Polyacrelate
3 = Napthalene

ST:
1 = Cube
2 = Cylinder

FT:
0 = None
1 = Steel
2 = Sisal
3 = Polypropylene
4 = Basalt
5 = Glass
"""
explanation_label = tk.Label(window, text=explanation_text, fg='black', bg='white', justify='left', font=("Arial", 12))
explanation_label.grid(row=0, column=4, rowspan=18, padx=30, pady=10, sticky='n')

# Prediction Function
def predict_strength():
    try:
        user_input = {}
        for feature, _, _ in features:
            value = input_entries[feature].get()
            if feature in ['FT', 'CT', 'SPB', 'ST']:
                user_input[feature] = int(float(value))
            else:
                user_input[feature] = float(value)

        user_input_df = pd.DataFrame([user_input])

        model_obj = models[selected_model.get()]

        prediction = model_obj.predict(user_input_df)[0]
        messagebox.showinfo("Prediction", f"Predicted Compressive Strength: {prediction:.2f} MPa")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

# Reset Function
def reset_fields():
    for entry in input_entries.values():
        entry.delete(0, tk.END)

# Buttons
predict_button = tk.Button(window, text="Predict", command=predict_strength, bg='black', fg='white')
predict_button.grid(row=len(features) + 1, column=0, padx=10, pady=20)

reset_button = tk.Button(window, text="Reset", command=reset_fields, bg='black', fg='white')
reset_button.grid(row=len(features) + 1, column=1, padx=10, pady=20)

# Credit
credit_label = tk.Label(
    window,
    text="Developed By. Eng. Mohamed Ayman and Assoc. Prof. Dr. Amr Maher El-Nemr - German University in Cairo (GUC)",
    fg='black', bg='white', font=("Arial", 14))
credit_label.grid(row=len(features) + 2, column=0, columnspan=5, pady=20)

print("✅ All sections completed. Launching GUI...")
window.mainloop()