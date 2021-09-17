## Exploratory Data Analysis + (Feature Engineering,Dimensionality Reduction)

#### The Entire Process of EDA is divided into 3 parts :
#### 1] Data Analysis
#### 2] Feature Engineering
#### 3] Dimensionality Reduction (Feature Selection,Feature Elimination,Feature extraction)

***
#### NOTE : This is a general process to perform EDA,depending on your dataset and task at hand,you may use all or only a few of the steps defined below.
***

#### Exploratory Data Analysis is performed only on structured datasets.These structured datasets have 2 types of features :
#### A) Numerical (of 2 types : Discrete,Continuous)
#### B) Categorical (of 2 types : Nominal,Ordinal)
***

# DATA ANALYSIS
### In this step we take a high level look at our dataset and gain insights on topics mentioned below:
* Shape of Dataset (number of rows and columns)
* Number of Numerical & Categorcial Features
* Features with most missing/null values
* Cardinality of Each Categorical Features 
* Distribution of Numerical Features
* Detecting Features with most Outliers
* Finding relationship between Independent & Dependent variables

#### NOTE : In Data Analysis our task is only to find insights and not to make any changes (like removing rows/columns or imputing missing values).After this stage we take these insights and make changes to the dataset in the feature engineering stage.Though this is not a hard rule and you can take any required steps as per your task.

### Import Python Packages
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Data Analysis Step-By-Step ("df" is dataset dataframe)

* Describe the Dataset using comman pandas functions
```python
df.info()     #prints info about features(dtype,non-null values,memory usage etc)
df.describe()   #describe statistics of features (mean,median,mode,count etc)
df.corr()     #computes pairwise pearson's correlation among features
```

* Finding variables with missing/null values and percentage of missing/null values.
```python
features_with_na = [feature for feature in df.columns if df[feature].isnull().sum() > 1]

for feature in features_with_na :
    print(feature,"  ",np.round(df[feature].isnull().mean(),4),"% missing values")
```
##### NOTE : In order to decide if we should remove or keep the features with missing/null values we need to find their relationship with the dependent variable and see if they are of any significance.

* Plot Bar-Graphs of TARGET_FEATURE against features with missing/null values
```python
for feature in features_with_na:
    data = df.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    data[feature] = np.where(data[feature].isnull(), 1, 0)
    
    data.groupby(feature)["TARGET_FEATURE"].median().plot.bar()
    plt.title(feature)
    plt.show()
```

## Analyze Numerical Variables

* Find number of numerical features in dataset
```python
# list of all numerical variables in dataset
numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']

print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df[numerical_features].head()
```

##### NOTE : Numerical variables are usually of 2 type : Continous & Discrete Variables,Analyze them seperately

### Analyze Discrete Variables
* Find number of Discrete features in dataset
```python
# list of all discrete variables in dataset
discrete_features=[feature for feature in numerical_features if len(df[feature].unique())<25]

print('Number of Discrete variables: ', len(discrete_features))

# visualise the numerical variables
df[discrete_features].head()
```

* Find the realtionship between them and dependent variable by plotting graphs
```python
for feature in discrete_features:
    data=df.copy()
    data.groupby(feature)["TARGET_FEATURE"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("TARGET_FEATURE")
    plt.title(feature)
    plt.show()
```

### Analyze Continuous Variables
* Find number of Continuous features in dataset
```python
# list of all continuous variables in dataset
continuous_features=[feature for feature in numerical_features if feature not in discrete_features]

print('Number of Continuous variables: ', len(continuous_features))

# visualise the numerical variables
df[continuous_features].head()
```


































