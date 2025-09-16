# script.py
# ------------------------------------
# Analyzing Data with Pandas and Visualizing Results with Matplotlib
# Using the Iris Dataset
# ------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Use seaborn style for better visuals
plt.style.use("seaborn-v0_8")

# -------------------------------
# Task 1: Load and Explore Dataset
# -------------------------------

# Load dataset from sklearn
iris_data = load_iris(as_frame=True)
df = iris_data.frame

print("First 5 rows of dataset:")
print(df.head(), "\n")

print("Dataset Info:")
print(df.info(), "\n")

print("Missing values:")
print(df.isnull().sum(), "\n")

# Add species column for readability
df["species"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# -------------------------------
# Task 2: Basic Data Analysis
# -------------------------------

print("Basic Statistics:")
print(df.describe(), "\n")

# Group by species and compute mean sepal length
grouped = df.groupby("species")["sepal length (cm)"].mean()
print("Average Sepal Length by Species:")
print(grouped, "\n")

print("Observations:")
print("- Setosa flowers have the smallest sepal length on average.")
print("- Virginica flowers tend to have the largest petals overall.\n")

# -------------------------------
# Task 3: Data Visualization
# -------------------------------

# 1. Line Chart: Sepal length across samples
plt.figure(figsize=(10,6))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
plt.title("Line Chart: Sepal Length across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart: Average petal length per species
plt.figure(figsize=(8,6))
df.groupby("species")["petal length (cm)"].mean().plot(
    kind="bar", color=["skyblue","orange","green"]
)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Sepal width distribution
plt.figure(figsize=(8,6))
plt.hist(df["sepal width (cm)"], bins=20, color="purple", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal length vs Petal length
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="Set1"
)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
