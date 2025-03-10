# --- Section 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visual styles for graphs
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["figure.figsize"] = (10, 6)

# --- Section 2: Load and Prepare Dataset ---
file_path = 'bmi.csv'
data = pd.read_csv(file_path)

# Encode categorical variables
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})  

# Define features and target variable
X = data[['Height', 'Weight']]
y = data['BMI']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predictions
y_pred = lr_model.predict(X_test)
residuals = y_test - y_pred  # Calculate residuals

# --- Section 3: Create Graphs ---

# 1. Scatterplot with Regression Line (Height vs BMI)
plt.figure(figsize=(8, 6))
sns.regplot(x=data["Height"], y=data["BMI"], scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.title("Height vs BMI Regression Line")
plt.xlabel("Height (cm)")
plt.ylabel("BMI")
plt.show()

# 2. Residual Plot (Fixed: Removed lowess=True to avoid errors)
plt.figure(figsize=(8, 6))
sns.residplot(x=y_pred, y=residuals, color="green")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residual Plot")
plt.xlabel("Predicted BMI")
plt.ylabel("Residuals")
plt.show()

# 3. Predicted vs Actual BMI
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='purple', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Perfect prediction line
plt.title("Predicted vs Actual BMI")
plt.xlabel("Actual BMI")
plt.ylabel("Predicted BMI")
plt.show()

# 4. Histogram of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=20, kde=True, color="blue", alpha=0.6)
plt.axvline(residuals.mean(), color='red', linestyle='--', label='Mean Residual')
plt.title("Distribution of Residuals")
plt.xlabel("Residual Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# 5. Feature Importance using Linear Regression Coefficients
coef_df = pd.DataFrame(lr_model.coef_, index=X.columns, columns=["Coefficient"]).reset_index()
coef_df.rename(columns={"index": "Feature"}, inplace=True)

plt.figure(figsize=(8, 6))
sns.barplot(x="Coefficient", y="Feature", data=coef_df, hue="Feature", palette='coolwarm', legend=False)
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.ylabel("Feature")
plt.show()

