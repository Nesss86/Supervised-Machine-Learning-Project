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
```

### Preprocessing

### Feature Engineering

### Training A Machine Learning Model



