import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df_before = pd.read_csv("Dataset_Clean.csv")
df_after = pd.read_csv("Filtered_Dataset.csv")

# ----------------------------------------------------------
# FIXED SAVE DIRECTORY
# ----------------------------------------------------------
SAVE_DIR = r"D:\Masters\Theoretical Masters Approach\Working\Outliers"
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------------------------------------
# FORCE NUMERIC CONVERSION
# ----------------------------------------------------------
def force_numeric(df):
    df_numeric = df.copy()
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    return df_numeric


# ----------------------------------------------------------
# OUTLIER COMPUTATION
# ----------------------------------------------------------
def compute_iqr_outliers(df_before: pd.DataFrame, df_after: pd.DataFrame):
    numeric_cols = df_before.select_dtypes(include=[np.number]).columns
    print("Numeric columns detected:", list(numeric_cols))

    if len(numeric_cols) == 0:
        print("ERROR: No numeric columns found. Boxplots will not be generated.")
        return pd.DataFrame()

    results = []

    for col in numeric_cols:
        Q1 = df_before[col].quantile(0.25)
        Q3 = df_before[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_before = df_before[(df_before[col] < lower_bound) | (df_before[col] > upper_bound)]
        outliers_after = df_after[(df_after[col] < lower_bound) | (df_after[col] > upper_bound)]

        results.append({
            "Feature": col,
            "Q1": Q1,
            "Q3": Q3,
            "IQR": IQR,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers Before Removal": len(outliers_before),
            "Outliers After Removal": len(outliers_after),
            "Removed Outliers": len(outliers_before) - len(outliers_after)
        })

    return pd.DataFrame(results)


# ----------------------------------------------------------
# COLORFUL BOXPLOT FUNCTION
# ----------------------------------------------------------
def save_color_boxplot(df, title, filename):
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) == 0:
        print("No numeric columns — skipping boxplot.")
        return

    # Prepare data
    data = [df[col].dropna().values for col in numeric_cols]

    # Build figure
    plt.figure(figsize=(16, 8))

    # Generate colorful boxplot
    boxprops = dict(linewidth=1.5)
    medianprops = dict(linewidth=2)

    box = plt.boxplot(
        data,
        patch_artist=True,
        labels=numeric_cols,
        boxprops=boxprops,
        medianprops=medianprops
    )

    # Apply colormap for colored boxes
    cmap = plt.cm.tab20
    for patch, color_index in zip(box['boxes'], range(len(box['boxes']))):
        patch.set_facecolor(cmap(color_index % 20))

    plt.title(title, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save
    full_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(full_path, dpi=300, format='jpg')
    plt.close()

    print(f"Saved: {full_path}")



# ----------------------------------------------------------
# PREPARE NUMERIC DATA
# ----------------------------------------------------------
df_before = force_numeric(df_before)
df_after  = force_numeric(df_after)

# ----------------------------------------------------------
# RUN OUTLIER ANALYSIS
# ----------------------------------------------------------
report = compute_iqr_outliers(df_before, df_after)
print("\n--- OUTLIER REPORT ---")
print(report)

# ----------------------------------------------------------
# SAVE COLORFUL BEFORE & AFTER BOXPLOTS
# ----------------------------------------------------------
save_color_boxplot(df_before, "Boxplot Before Removing Outliers", "before_outliers_boxplot.jpg")
save_color_boxplot(df_after, "Boxplot After Removing Outliers", "after_outliers_boxplot.jpg")
import matplotlib.pyplot as plt

# ----------------- SAMPLE CALCULATION FUNCTION -----------------
def sample_feature_calculation_plot_C(df_before, df_after, save_dir):
    feature_name = "C"
    if feature_name not in df_before.columns:
        print(f"Feature '{feature_name}' not found in dataset.")
        return

    before_vals = pd.to_numeric(df_before[feature_name], errors='coerce').dropna()
    after_vals  = pd.to_numeric(df_after[feature_name], errors='coerce').dropna()

    Q1 = before_vals.quantile(0.25)
    Q3 = before_vals.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_before = before_vals[(before_vals < lower_bound) | (before_vals > upper_bound)]
    outliers_after  = after_vals[(after_vals < lower_bound) | (after_vals > upper_bound)]
    removed_count = len(outliers_before) - len(outliers_after)

    text = (
        f"Feature: {feature_name}\n"
        f"Q1 = {Q1}\n"
        f"Q3 = {Q3}\n"
        f"IQR = {IQR}\n"
        f"Lower Bound = {lower_bound}\n"
        f"Upper Bound = {upper_bound}\n"
        f"Outliers Before Removal = {len(outliers_before)}\n"
        f"Outliers After Removal = {len(outliers_after)}\n"
        f"Removed Outliers = {removed_count}\n"
        f"Values Before Removal: {outliers_before.values}"
    )

    plt.figure(figsize=(8, 6))
    plt.axis('off')
    plt.text(0.01, 0.99, text, fontsize=12, verticalalignment='top', fontfamily='monospace')
    plt.tight_layout()

    filename = "sample_calculation_C.jpg"
    full_path = os.path.join(save_dir, filename)
    plt.savefig(full_path, dpi=300, format='jpg')
    plt.close()

    print(f"Sample calculation for '{feature_name}' saved as: {full_path}")

# Save sample calculation for feature "C"
sample_feature_calculation_plot_C(df_before, df_after, SAVE_DIR)
