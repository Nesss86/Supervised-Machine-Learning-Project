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
```

### Preprocessing

### Feature Engineering

### Training A Machine Learning Model



