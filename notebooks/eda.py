# Credit Risk Model - Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("=" * 60)
print("CREDIT RISK MODEL - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Load data
data_path = "data/raw/"
if os.path.exists(data_path):
    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    if csv_files:
        df = pd.read_csv(os.path.join(data_path, csv_files[0]))
        print(f"Loaded: {csv_files[0]}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    else:
        print("No CSV files found in data/raw/")
else:
    print("data/raw/ directory does not exist")
    print("Please download the Xente dataset from Kaggle")

print("\nEDA Complete!")
