# Hybrid-Experimental-and-Machine-Learning-Approach
This repository includes all the resources (datasets, and codes) developed for the study title "A Hybrid Experimental and Machine Learning Framework for Designing and Predicting Compressive Strength of Ultra-High-Performance Concrete" authored by Mohamed Ayman, and Prof. Amr ElNemr

This "read me" describes the resources and provides a brief abstract for the work performed.
**All Codes have been developed using "PyCharm 2025.1.2", All Files uploaded as ".py"**

# Resources Describtion
# 1 - Data
Three data files are available:
A) Raw_Dataset.csv: This contains the raw compiled dataset prior to any filtration
B) Dataset_Clean.csv: This contains manually filtered data as described in 3.1 Dataset Development
C) Filtered_Dataset.csv: This contains the final dataset prior to all filteration and outliers removal with 550 comrpessive strength results.
# 2- Codes
Full Project running as one code is available in the **Full Project Code** Folder

Separate Codes for Each Project Phase are Available in **Separate Codes Folder**
A) Outliers Removal: This folder contains the code used to remove outliers in the preprocessing phase. **Dataset (B) Dataset_Clean.csv Was Used**.
**The Following Codes Used Dataset (C) Filtered_Dataset.csv**
B) Tree Models: This folder contains two codes; one for non-optimized models, and the other is for optimized models.
C) Boosting Models: This folder contains two codes; one for non-optimized models, and the other is for optimized models.
D) Support Vector Machines Models: This folder contains two codes; one for non-optimized models, and the other is for optimized models.
E) GUI: This folder contains the code developed for the Graphical User Interface (GUI). "app.py" is for GUI developed by Gradio, and "GUI_Predictor" is for GUI developed by Tkinter.

# Abstract
Ultra-high-performance concrete (UHPC) offers exceptional mechanical and durability properties but often relies on quartz powder, raising sustainability and occupational health concerns. This study introduces an integrated experimental-computational framework for predicting the compressive strength of UHPC and developing quartz-free mixtures. Experimentally, the effects of mixing sequence, sand characteristics, superplasticizer chemistry, and curing regime were investigated, leading to a quartz-free UHPC achieving 136 MPa at 28 days under heat-curing. A dataset of 550 UHPC compressive strength records was compiled, incorporating quantitative mix proportions and categorical variables (cement type, superplasticizer base, fiber type, and specimen geometry). Sixty-three machine learning models from tree-based, boosting, and support vector machine families were optimized using seven meta-heuristic algorithms. The Particle Swarm Optimization-tuned XGBoost model achieved the highest prediction accuracy (R² = 0.897, RMSE = 7.63 MPa), followed by the Differential Evolution-optimized Random Forest (R² = 0.867, RMSE = 8.70 MPa). SHapley Additive exPlanations (SHAP) analysis identified curing age as the most influential predictor after optimization. The proposed framework enables accurate and interpretable UHPC strength prediction and supports the design of safer and more sustainable quartz-free UHPC with reduced experimental effort.
