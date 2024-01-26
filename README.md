# machine_learning_project-supervised-learning

## Project/Goals
The goal of this project was to use supervised learning techniques to build a machine learning model that can predict whether a patient has diabetes or not, based on certain diagnostic measurements. The following stages were performed on the data set provided:
- Exploratory data analysis
- Preprocessing
- Feature engineering
- Training a machine learning model


## Process
### EDA

``` python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Load diabetes dataset
diabetes_data = pd.read_csv('diabetes.csv')

# Check for missing values
missing_values = diabetes_data.isnull().sum()
print("Missing Values:\n", missing_values)

# Visualize the relationship between predictor variables and the outcome variable
sns.pairplot(diabetes_data, hue='Outcome', diag_kind='kde')
plt.show()

# Calculate and visualize the correlation matrix
correlation_matrix = diabetes_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# Visualize the distribution of each predictor variable
diabetes_data.hist(figsize=(12, 10))
plt.show()

# Visualize boxplots to identify outliers
plt.figure(figsize=(15, 10))
diabetes_data.boxplot()
plt.show()

# Visualize relationships between predictor variables
sns.pairplot(diabetes_data)
plt.show()

average_age = diabetes_data['Age'].mean()
print("Average Age:", average_age)

average_glucose_diabetes = diabetes_data[diabetes_data['Outcome'] == 1]['Glucose'].mean()
average_glucose_no_diabetes = diabetes_data[diabetes_data['Outcome'] == 0]['Glucose'].mean()

print("Average Glucose Level for Diabetes:", average_glucose_diabetes)
print("Average Glucose Level for No Diabetes:", average_glucose_no_diabetes)

average_bmi_diabetes = diabetes_data[diabetes_data['Outcome'] == 1]['BMI'].mean()
average_bmi_no_diabetes = diabetes_data[diabetes_data['Outcome'] == 0]['BMI'].mean()

print("Average BMI for Diabetes:", average_bmi_diabetes)
print("Average BMI for No Diabetes:", average_bmi_no_diabetes)

# Visualize the distribution of predictor variables for diabetes and no diabetes
sns.boxplot(x='Outcome', y='value', data=pd.melt(diabetes_data, id_vars=['Outcome']))
plt.show()
```

### Preprocessing

``` python
# Check for missing values
missing_values = diabetes_data.isnull().sum()
print("Missing Values:\n", missing_values)

#Replace missing values with the mean or median of the respective columns:
diabetes_data.fillna(diabetes_data.median(), inplace=True)

# Visualize boxplots to identify outliers
plt.figure(figsize=(15, 10))
diabetes_data.boxplot()
plt.show()

# Handle outliers (if necessary)
from scipy.stats.mstats import winsorize
diabetes_data['Insulin'] = winsorize(diabetes_data['Insulin'], limits=[0.05, 0.05])
```


### Feature Engineering

### Training A Machine Learning Model



