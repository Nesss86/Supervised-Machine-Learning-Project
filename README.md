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
<img src="images/Correlation Matrix.png" alt="Notebook">

<img src="images/Boxplot for Outliers.png" alt="Notebook">

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

``` python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Features to scale/normalize (excluding the target variable 'Outcome')
features_to_scale = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Standard Scaling
scaler = StandardScaler()
diabetes_data[features_to_scale] = scaler.fit_transform(diabetes_data[features_to_scale])

# Create a new feature representing BMI categories
diabetes_data['BMI_Category'] = pd.cut(diabetes_data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
```

### Training A Machine Learning Model
``` python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Separate features (X) and target variable (y)
X = diabetes_data.drop('Outcome', axis=1)
y = diabetes_data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create transformers for numeric and categorical columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # You can use other strategies for categorical data
])

# Apply transformers to appropriate columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
    ])

# Apply SMOTE for oversampling the minority class
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(preprocessor.fit_transform(X_train), y_train)
```

## Results

The initiative was to try two models to determine which would be the best to use for this prediction. 

The logisitc regression model was peformed first and yielded the following results:
<img src="images/Logistic Regression.png" alt="Notebook">

The Random Forest Model was perfomed second and yielded the following results:
<img src="images/Random Forest Model.png" alt="Notebook">


## Conclusion
- The strongest positive correlations were pregnancies and age. As both of those increased, the likelihood of diabetes incresed
- There appear to be outliers for skin thickness and insulin
- The average age of people with diabetes was 33
- While the random forest model had higher accuracy, it was determined that the logisitic regression model had better recall, F-1 score, ROC-AUC, and would be the better choice for supervised machine learning for this dataset.









