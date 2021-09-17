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

# Data Analysis
### In this step we take a high level look at our dataset and gain insights on topics mentioned below:
* Shape of Dataset (number of rows and columns)
* Number of Numerical & Categorcial Features
* Features with most missing/null values
* Cardinality of Each Categorical Features 
* Distribution of Numerical Features
* Detecting Features with most Outliers
* Finding relationship between Independent & Dependent variables

#### NOTE : In Data Analysis our task is only to find insights and not to make any changes (like removing rows/columns or imputing missing values).After this stage we take these insights and make changes to the dataset in the feature engineering stage.Though this is not a hard rule and you can take any required steps as per your task.


## Data Analysis Step-By-Step ("df" is dataset dataframe)

* Describe the Dataset using comman pandas functions
```python
df.info()     #prints info about features(dtype,non-null values,memory usage etc)
df.describe()   #describe statistics of features (mean,median,mode,count etc)
df.corr()     #computes pairwise pearson's correlation among features
```
















































