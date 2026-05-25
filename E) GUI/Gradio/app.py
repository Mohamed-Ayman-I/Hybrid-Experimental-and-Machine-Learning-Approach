# ================================ Imports ================================
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
import joblib
import sys
import os

# ================================ Helper to Read Files in .exe ================================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ================================ Load Models ================================
model_info = {
    'Random Forest (DE)': ('RF_DE.pkl', 'Random Forest', 'DE'),
    'XGBoost (PSO)': ('XGB_PSO.pkl', 'XGBoost', 'PSO'),

}

models = {}
for key, (filename, _, _) in model_info.items():
    models[key] = joblib.load(resource_path(filename))

# ================================ GUI Setup ================================
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

# ================================ Dropdown for Model Selection ================================
selected_model = tk.StringVar(window)
selected_model.set(list(models.keys())[0])
model_menu = tk.OptionMenu(window, selected_model, *models.keys())
model_menu.grid(row=0, column=3, padx=10, pady=10)

model_label = tk.Label(window, text='', fg='black', bg='white', font=("Arial", 12))
model_label.grid(row=1, column=3, padx=10, pady=5, sticky='w')

def update_model_label(*args):
    model_key = selected_model.get()
    model_name, optimizer = model_info[model_key][1:]
    model_label.config(text=f"Model: {model_name} | Optimizer: {optimizer}")

selected_model.trace("w", update_model_label)
update_model_label()

# ================================ Explanation Labels ================================
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


# ================================ Prediction Function ================================
def predict_strength():
    try:
        user_input = {}
        for feature, _, _ in features:
            value = input_entries[feature].get()
            if feature in ['FT', 'CT', 'SPB', 'ST']:
                user_input[feature] = int(float(value))  # Convert safely to int
            else:
                user_input[feature] = float(value)

        user_input_df = pd.DataFrame([user_input])

        model_obj = models[selected_model.get()]  # full pipeline: preprocess + predict

        # DEBUG: Show input before and after preprocessing
        print("\n📥 Raw Input DataFrame:")
        print(user_input_df)

        try:
            # Try to extract preprocessing step and transform
            preprocessor = model_obj.named_steps['preprocessor']
            X_transformed = preprocessor.transform(user_input_df)
            print("🔍 Transformed input for NuSVR:")
            print(X_transformed)
        except Exception as e:
            print("⚠️ Could not transform input:", str(e))


        prediction = model_obj.predict(user_input_df)[0]
        messagebox.showinfo("Prediction", f"Predicted Compressive Strength: {prediction:.2f} MPa")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{str(e)}")


# ================================ Reset Function ================================
def reset_fields():
    for entry in input_entries.values():
        entry.delete(0, tk.END)

# ================================ Predict Button ================================
predict_button = tk.Button(window, text="Predict", command=predict_strength, bg='black', fg='white')
predict_button.grid(row=len(features) + 1, column=0, padx=10, pady=20)

# ================================ Reset Button ================================
reset_button = tk.Button(window, text="Reset", command=reset_fields, bg='black', fg='white')
reset_button.grid(row=len(features) + 1, column=1, padx=10, pady=20)

# ================================ Built By Label ================================
credit_label = tk.Label(
    window,
    text="Developed By. Eng. Mohamed Ayman and Assoc. Prof. Dr. Amr Maher El-Nemr - German University in Cairo (GUC)",
    fg='black', bg='white', font=("Arial", 14))
credit_label.grid(row=len(features) + 2, column=0, columnspan=5, pady=20)

# ================================ Run GUI ================================
window.mainloop()
