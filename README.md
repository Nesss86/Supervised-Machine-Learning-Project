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

I decided to use two models and evaluated them both to determine which was best for this particular project. I started with a logistic regression model and then tried the more complex model (random forest) to see if it showed any improvements in performance. It is considered good practice to use multiple models as well as incorporating some hyperparameter tuning to further improve. Cross validation is always a benefit as is assists with the following:

- Assessing model performance
- Reduce Overfitting
- Optomize Hyperparameter Tuning
- Handle Imbalanced Datasets
- Model Selection


## Results

The initiative was to try two models to determine which would be the best to use for this prediction. 

The logisitc regression model was peformed first and yielded the following results:
<img src="images/Logistic Regression Model.png" alt="Notebook">

<img src="images/Confusion Matrix - Logistic Regression .png" alt="Notebook">

The Random Forest Model was perfomed second and yielded the following results:
<img src="images/Random Forest Model.png" alt="Notebook">

<img src="images/Confusion Matrix - Random Forest.png" alt="Notebook">


## Conclusion
- The strongest positive correlations were pregnancies and age. As both of those increased, the likelihood of diabetes incresed
- There appear to be outliers for skin thickness and insulin
- The average age of people with diabetes was 33
- While the random forest model had higher accuracy, it was determined that the logisitic regression model had better recall, F-1 score, ROC-AUC, and would be the better choice for supervised machine learning for this dataset.









