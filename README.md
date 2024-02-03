# machine_learning_project-supervised-learning

## Project/Goals
The goal of this project was to use supervised learning techniques to build a machine learning model that can predict whether a patient has diabetes or not, based on certain diagnostic measurements. The following stages were performed on the data set provided:
- Exploratory data analysis
- Preprocessing
- Feature engineering
- Training a machine learning model


## Process
### EDA

- Looked for missing values and outliers
- Generated summary statistics to look at distribution
- Generated a Correlation matrix to see which variables had the strongest relationship

<img src="images/Correlation Matrix.png" alt="Notebook">

<img src="images/Boxplot for Outliers.png" alt="Notebook">

### Preprocessing


 - Checked for missing values
 - Checked for outliers
 - Looked at the summary statistics to see the distribution


<img src="images/Summary Statistics.png" alt="Notebook">



### Feature Engineering

 - Standard Scaler was chosen for feature engineering as it is more sensitive to outliers

``` python

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









