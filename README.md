## Exploratory Data Analysis + (Feature Engineering,Dimensionality Reduction)

#### The Entire Process of EDA is divided into 3 parts :
#### 1] Data Analysis
#### 2] Feature Engineering
#### 3] Dimensionality Reduction (Feature selection,Feature extraction)

#### NOTE : This is a general process to perform EDA,depending on your dataset and task at hand,you may use all or only a few of the steps defined below.

***
### Useful Links : 
#### Top 50 matplotlib charts : https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
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

***

## Data Analysis Step-By-Step 
### ("df" is dataset dataframe,Target_Feature is continuous eg:price)

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

***

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

* Plot Scatterplots of TARGET_FEATURE against each Continuous Feature
```python
# Plotting Scatterplots of each continuous feature against TARGET_FEATURE
for feature in continuous_features:
    data=df.copy()
    plt.scatter(df[feature],df["TARGET_FEATURE"])
    plt.xlabel(feature)
    plt.ylabel("TARGET_FEATURE")
    plt.show()
```


* Plot Scatterplots of Continuous Features against each other
```python
# NOTE : Dont do this if you have more than 3 variables,since it'll create a very large numbers 
# of plots and it wont be possible to visualize them all at once. 
# You can use "df.corr()" to find the correlation among the features

# Eg : If you have 10 continuous features,then it'll create (10x10)=100 scatterplots

# Plotting Scatterplots of each continuous feature against TARGET_FEATURE
for feature1 in continuous_features:
    for feature2 in continuous_features:
            data=df.copy()
            plt.scatter(df[feature1],df[feature2])
            plt.xlabel(feature1)
            plt.ylabel(feature2)
            plt.show()
```


##### NOTE : Plotting Scatterplots of Continuous features against target variable and against each other help is determine the "correlation" among them.This is particularly helpful for Regression problem,where we only want to "keep features that have high-correlation with the target variable and low-correlation among themselves".


* Find the Distribution of values in each Continuous feature (create histograms)
```python
## NOTE : If the distributions are skewed we may need to transform them into another format like (Standard Distribution).
## Such data can be handled by following ways :
# 1] Log Transform (mostly used)
# 2] Box Cox Transform
# 3] Square Root Transform

for feature in continuous_features:
    data=df.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.show()
```

##### NOTE : It is very important to know the distribution of values in continuous features,in case the distribution is "Skewed" (Right Skewed or Left Skewed) we need to transform it into another format like Standard-Distribution during Feature-Engneering.This is a MUST step in a Regression Problem.

### Outliers in Continuous Variables

* Finding outliers distribution in each feature by creating their box-plot
```python

# NOTE : This works only for continuous variables and not for categorical

for feature in continuous_features:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        # applying log Transformation
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
```

* Finding outliers using Z-Score or Interquartile-Range
> see this blog : https://notes88084.blogspot.com/2021/04/exploratory-data-analysis.html

***

## Analyze Categorical Variables

* Find number of categorical features in dataset
```python
# list of all categorical variables in dataset
categorical_features=[feature for feature in df.columns if df[feature].dtypes=='O']

print('Number of categorical variables: ', len(categorical_features))

# visualise the numerical variables
df[categorical_features].head()
```

* Find the Cardinality i.e number of categories in each categorical feature
```python
for feature in categorical_features:
    print('The feature is {} and no. of categories are {}'.format(feature,len(df[feature].unique())))
```

##### NOTE : Categorical variables are usually of 2 type : Nominal & Ordinal.Depending on the Cardinality we can choose if we need to use one-hot encoding or label-encoding. Mostly we use one hot encoding for nominal variable,label encoding for ordinal variable

* Find the relationship between categorical variables and dependent feature
(Here the target_feature is continuous,eg-price)
```python
for feature in categorical_features:
    data=df.copy()
    data.groupby(feature)["TARGET_FEATURE"].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel("TARGET_FEATURE")
    plt.title(feature)
    plt.show()
```

***


# FEATURE ENGINEERING
#### In this stage we use the insights from previous stage to transform data into more suitable format,below are the things we do here:
* Handle missing/null values
* Categorical variables : remove rare labels
* Encode categorical features (label & one-hot encoding)
* Handle outliers in numerical features
* Handle Skewed Distribution in continuous features
* Feature scaling : standarise the variables to the same range


## Handle missing values

* Finding numerical variables with missing/null values and percentage of missing/null values.
```python

numerical_with_nan=[feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes!='O']

print(numerical_with_nan,"\n")

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(df[feature].isnull().mean(),4)))
```

* Finding categorical variables with missing/null values and percentage of missing/null values.
```python

categorical_with_nan=[feature for feature in df.columns if df[feature].isnull().sum()>1 and df[feature].dtypes=='O']

print(categorical_with_nan,"\n")

for feature in categorical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(df[feature].isnull().mean(),4)))
```


#### Common ways to deal with missing values :
* Deletion of rows/columns with missing values
* Replace missing values with Mean/Median/Mode of respective feature.
* Create a prediction model to predict missing values
* KNN Imputation

> see this blog : https://notes88084.blogspot.com/2021/04/exploratory-data-analysis.html

<img src="/Images/missing_values_graph.png" width="58%">


## Handle outliers in numerical features
#### Outliers can be of 2 types :
>* Artificial : outliers created unintentionally due to error during data collection.
>* Natural : outlier which is not artificial.

<img src="/Images/outliers_sensitive.png" width="58%">

#### Common ways to deal with outliers :
* Delete outliers
* Replacing outliers with Mean/Median/Mode (only if outlier is artificial)
* Treat Seperately - If there are significant number of outliers, we should treat them separately.One of the approach is to treat both groups as two different datasets and build individual model for both groups and then combine the output.
> see this blog : https://notes88084.blogspot.com/2021/04/exploratory-data-analysis.html


## Handle Skewed Distribution in continuous features

#### If the distributions are skewed (right or left skewed) we may need to transform them into another format like (Standard Distribution).Some common transformations are mentioned below :
* Log Transform (mostly used)
* Box Cox Transform
* Square Root Transform


#### (Applying Logarithmic Transformation to continuous features where target_feature is also continuous)
```python
for feature in continuous_features:
    data=df.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data["TARGET_FEATURE"]=np.log(data["TARGET_FEATURE"])
```
        
#### NOTE : After transformation plot histograms again to verify the change.


## Categorical variables : remove rare labels
##### If you have a categorical variable with large number of categories,then it is possible that not all of those categories are contributing so much in prediction.We need to remove these categories.By remove we mean we'll give these categories a new label,this will group these different categories into a single category.

> see this video : https://youtu.be/AtXNo2c-TYk
```python
# list of categorical features
categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']

# For each categorical feature replace all the catergories with "Rare_Value" which 
# are present in less than 1% on entire dataset samples.

for feature in categorical_features:
    temp=df.groupby(feature)["TARGET_FEATURE"].count()/len(df)
    temp_df=temp[temp>0.01].index # 0.01 means 1%
    df[feature]=np.where(df[feature].isin(temp_df),df[feature],'Rare_var')
```


## Feature Scaling
#### Here we bring the values of all features to a common scale so that difference in magnitude of feature values dont affect the accuracy of the model.When the values in some features/columns are very high as compared to rest of the features,then these features tend to have high impact in model prediction, even if they are far less crucial in determining the output hence feature-scaling must be applied here.

#### Note : Not all models need feature scaling. Tree based algorithms are fairly insensitive to feature scaling.Algorithms that compute the distance between variables are highly biased towards larger values & hence need feature scaling (eg:K-means,SVM)

The 2 main types of feature scaling methods :

* Normalization : features will be rescaled to range of [0,1]
* Standardization : features will be rescaled so that they’ll have the properties of a standard normal distribution with mean, μ=0 and standard deviation, σ=1.

<img src="/Images/feature_scaling.png" width="58%">

> see this blog : https://notes88084.blogspot.com/2021/04/exploratory-data-analysis.html

#### Normalization with SkLearn
```python
from sklearn.preprocessing import MinMaxScaler

# choose features on which to perform scaling.
# Perform scaling only on continuous features which will be used in prediciton.
feature_scale=[feature for feature in df.columns if feature not in ['Id','Name'] and df[feature].dtypes!='O']

scaler=MinMaxScaler()
scaler.fit(df[feature_scale])
scaled_data = scaler.transform(df[feature_scale]) # returns numpy array
```

#### Standardization with SkLearn
```python
from sklearn.preprocessing import StandardScaler

# choose features on which to perform scaling.
# Perform scaling only on continuous features which will be used in prediciton.
feature_scale=[feature for feature in df.columns if feature not in ['Id','Name'] and df[feature].dtypes!='O']

scaler = StandardScaler()
scaler.fit(df[feature_scale])
scaled_data = scaler.transform(df[feature_scale]) # returns numpy array
```


## Encoding categorical features
#### Many times the data set will contain categorical variables, these variables are typically stored as text values. In order to use those categorical features in a model we have to convert them into numerical format.

There are 2 main types of encoding :
* Label Encoding or Ordinal Encoding ***(used on ordinal categorical features)***
* One-Hot Encoding ***(used on nominal categorical features)***

### Label-Encoding with SkLearn
```python
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

# perform encoding only on categorical features
categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']

for feature in categorical_features:
    # encode categorical feature
    df[feature] = label_encoder.fit_transform(df[feature])

print(df.head())
```


### One-Hot Encoding with Pandas

#### NOTE : If you have categorical features with high cardinality i.e large number of categories,AVOID using one-hot encoding as it'll create a sparse input matrix and significantly increase number of columns/features. Also avoid falling into a "DUMMY VARIABLE TRAP"

```python
# list of categorical features in dataset
categorical_features=[feature for feature in df.columns if df[feature].dtype=='O']

# encode categorical features
new_encoded_columns = pd.get_dummies(df[categorical_features])

# Concatinating with original dataframe
df = pd.concat([df,new_encoded_columns],axis="columns")

# dropping the categorical variables since they are redundant now.
df = df.drop(categorical_features,axis="columns")
```
<img src="EDA Examples/dummy_variable_trap.png" width="58%">


***
# DIMENSIONALITY REDUCTION
#### Dimensionality reduction consist of 2 main parts :
* Feature Selection :  we try to select the most optimal features for the model from a given set of existing large number of features.
* Feature Extraction : we reduce the number of features in a dataset by creating new features from the existing ones.

**Feature Selection includes 3 types of methods :**
* Filter methods
* Wrapper methods
* Embedded methods

**Feature Extraction methods :** 
1. Principal Component Analysis (PCA)
2. Linear Discriminant Analysis (LDA)
3. t-SNE (Non-Linear)
4. Auto-Encoder

> see this blog : https://notes88084.blogspot.com/2021/04/dimensionality-reduction.html

***

## Filter Methods (feature selection)
#### In Filter methods the features are selected on the basis of their scores in various statistical tests for their correlation with the target variable.
**some common filter methods :** 
1. Variance Threshold
2. Pearson's Correlation
3. Chi Square Test
4. Information gain


### Variance Threshold (filter method)
#### This technique is a quick and lightweight way of eliminating features with very low variance (features with not much useful information).It removes all features whose variance doesn't meet some threshold.By default, it removes all zero-variance features(features that have the same or constant value in all samples).This feature selection algorithm looks only at the features (X), not the desired outputs (y), and can thus be used for unsupervised learning.
#### NOTE : This estimator only works with numeric data and it will raise an error if there are categorical features present in the dataframe.

```python
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
THRESHOLD = 3
df = pd.read_csv("train.csv")

# Takes a dataframe & threshold,returns a dataframe with low-variance columns removed.
def remove_features(df, threshold):
    # list of all numerical features in dataset
    numerical_features = [feature for feature in df.columns if df[feature].dtype != 'O']

    # dataframe of numerical features
    df_numerical = df[numerical_features]

    vt = VarianceThreshold(threshold)
    vt.fit(df_numerical)

    # list of selected columns
    selected_columns = df_numerical.columns[vt.get_support()]

    # list of columns not selected
    columns_to_remove = []
    for column in df_numerical.columns:
        if column not in selected_columns:
            columns_to_remove.append(column)

    print("Number of Columns Removed : ", len(columns_to_remove))
    print("List of Removed Columns : ", columns_to_remove)

    # removing columns from original dataset
    df = df.drop(columns_to_remove, axis="columns")
    return df
   
df = remove_features(df,5)
df.head(10)
```

***






## Wrapper Methods (feature selection)
#### In wrapper methods, we try to use a subset of features and train a model using them. Based on the inferences that we draw from the previous model, we decide to add or remove features from your subset.

#### Caution : wrapper methods are usually computationally very expensive.

**some common wrapper methods :** 
1. Forward selection
2. Backward elimination
3. Recursive feature elimination
4. Genetic Algorithms

#### Note: One of the most advanced algorithms for feature selection are "genetic algorithms".These are stochastic methods for function optimization based on the mechanics of natural genetics and biological evolution.


***





## Embedded Methods (feature selection)
#### Embedded methods combine the qualities’ of filter and wrapper methods. It’s implemented by algorithms that have their own built-in feature selection methods.The most typical embedded technique is decision tree algorithm. Decision tree algorithms select a feature in each recursive step of the tree growth process and divide the sample set into smaller subsets.

**some common embedded methods :**
1. Decision Tree
2. Lasso Regression (L1)
3. Ridge Regression (L2)


***



















































































