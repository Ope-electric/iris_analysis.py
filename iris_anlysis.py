# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({i: species for i, species in enumerate(iris.target_names)})

    print("✅ Dataset loaded successfully!")
    print(df.head())

    # Data types and missing values
    print("\n🔍 Data Types and Missing Values:")
    print(df.info())
    print(df.isnull().sum())

    # Clean missing values if any (none in Iris, but here for practice)
    df.dropna(inplace=True)

except Exception as e:
    print("❌ Error loading the dataset:", e)

# Task 2: Basic Data Analysis
print("\n📊 Basic Statistics:")
print(df.describe())

# Group by species and compute mean
print("\n📊 Mean of numerical columns grouped by species:")
print(df.groupby("species").mean())

# Insights (example): Sepal length comparison
print("\n🔎 Insight: Species with highest average sepal length:")
print(df.groupby("species")["sepal length (cm)"].mean().idxmax())

# Task 3: Data Visualization
sns.set(style="whitegrid")

# 1. Line Chart - Not typically time-based, but we simulate using index
plt.figure(figsize=(8, 4))
plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color='green')
plt.title("Trend of Sepal Length across dataset entries")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x="species", y="petal length (cm)", data=df)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram - Distribution of sepal width
plt.figure(figsize=(6, 4))
sns.histplot(df["sepal width (cm)"], bins=10, kde=True, color='blue')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot - Sepal vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title("Sepal Length vs Petal Length by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()
