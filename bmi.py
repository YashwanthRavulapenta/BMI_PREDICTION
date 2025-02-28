# --- Section 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set visual styles for graphs
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)

# --- Section 2: Load and Explore Dataset ---
# Load the dataset
file_path = 'bmi.csv'
data = pd.read_csv(file_path)

# Preview dataset
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

# Encode categorical variables for Gender and BMI Class
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])  # Encodes 'Male' as 1, 'Female' as 0
data['BMI Class'] = label_encoder.fit_transform(data['BMI Class'])

# --- Section 3: Prepare Data for Visualization ---
# Define numeric and categorical features
categorical_features = ['Gender']
numerical_features = ['Height', 'Weight', 'BMI']
target = 'BMI Class'

# Split data for model evaluation (used later)
X = data[['Gender', 'Height', 'Weight']]
y = data['BMI Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Section 4: Create Graphs ---
# 1. Gender Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='Gender', palette='coolwarm', hue='Gender', dodge=False)
plt.title("Gender Distribution", fontsize=16)
plt.xlabel("Gender (0: Female, 1: Male)", fontsize=14)
plt.ylabel("Count", fontsize=14)
for i, count in enumerate(data['Gender'].value_counts()):
    plt.text(i, count + 10, str(count), ha='center', fontsize=12, color="black")
plt.show()

# 2. BMI Distribution
plt.figure(figsize=(8, 6))
sns.histplot(data['BMI'], bins=20, kde=True, color='blue', alpha=0.6)
plt.title("BMI Distribution", fontsize=16)
plt.xlabel("BMI", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.axvline(data['BMI'].mean(), color='red', linestyle='--', label='Mean BMI')
plt.axvline(data['BMI'].median(), color='green', linestyle='--', label='Median BMI')
plt.legend()
plt.show()

# 3. Height vs Weight Scatterplot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Height', y='Weight', hue='BMI Class', data=data, palette='cool', edgecolor='w', s=100)
plt.title("Height vs Weight with BMI Class", fontsize=16)
plt.xlabel("Height (cm)", fontsize=14)
plt.ylabel("Weight (kg)", fontsize=14)
plt.legend(title="BMI Class", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# 4. BMI Class Distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=data, x='BMI Class', palette='viridis', hue='BMI Class', dodge=False)
plt.title("BMI Class Distribution", fontsize=16)
plt.xlabel("BMI Class", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.xticks(ticks=range(len(label_encoder.classes_)), labels=label_encoder.classes_)
plt.show()

# 5. Correlation Matrix Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap", fontsize=16)
plt.show()

# 6. Feature Importance Barplot
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
feature_importances = rf_classifier.feature_importances_

plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importances, y=X.columns, palette='coolwarm')
plt.title("Feature Importance", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.show()

# --- Section 6: Save Figures (Optional) ---
output_path = 'graphs/'
# Replace plt.show() with plt.savefig() to save plots