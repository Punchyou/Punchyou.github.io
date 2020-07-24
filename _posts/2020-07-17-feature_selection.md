---
layout: post
title: Feature Selection (In Progress)
author: Maria Pantsiou
date: '2020-07-17 14:35:23 +0530'
category: dataScience
summary: Feature Selection (In Progress)
thumbnail: ico_sphere_bg.png

---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
        tex2jax: {
            skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
            displayMath: [['$$','$$']],
            inlineMath: [['$','$']],
        },
    });
</script>

# Feature Selection Algorithms
*This article is about selecting the best set of features for your model. I did some research on this topic and ended up finding really interesting resources that refer to different algotihms that can be parts of a robust feature selection tool. Take a look on what I've found:*


# Why Feature Selection?
1. **When having too many features the model overfits due to the curse of dimentionality.** If we have more columns in the data than the number of rows, we will be able to fit our training data perfectly, but that won’t generalize to the new samples. And thus we learn absolutely nothing
2. **Occam’s Razor**: We want our models to be simple and explainable, and having multiple features doesn't help to that
3. **Garbage In Garbage out**: Poor-quality input will produce Poor-Quality output


Three categories of feature selection process:
1. **Filter Based**: An example of such a metric could be correlation/chi-square
2. **Wrapper-based**: Wrapper methods consider the selection of a set of features as a search problem. Example: *Recursive Feature Elimination*
3. **Embedded**: Embedded methods use algorithms that have built-in feature selection methods. For instance, *Lasso* and *RF* have their own feature selection methods.

## Implementation
The problem that we want to solve is to find the best features that can tell us if a football player is "great" or not, based on dataset of existing football players. Our training dataset consists of a column that contains names of famous football players to date, columns with different information about those players and an *Overall* column (range 46 - 94) based on which we will decide where a player is "great" or not. In our case, if the *Overall* score is more than 87, then we consider the player "great".

*You can find the code of this article [here](https://github.com/Punchyou/my_examples_and_notes/blob/master/feature_selection/feature_selection.py), but you can also continue reading to see in more detail the steps I followed.*


Use the following code to import all the libraries we'll use thoughout this article's code and to prepare the example dataset for this implementation:

```py
import scipy.stats as ss
import math 
import pandas as pd
import numpy as np
import seaborn as sns
import re
from matplotlib import pyplot as plt
from scipy import stats
from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

sns.set(style="ticks") #remove grid lines
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
flatui = sns.color_palette(flatui)

#upload dataset
player_df = pd.read_csv("https://raw.githubusercontent.com/amanthedorkknight/fifa18-all-player-statistics/master/2019/data.csv").drop("Unnamed: 0", axis=1)

# make categorical into numberical
numerical_cols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
categorical_cols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
player_df = player_df[numerical_cols+categorical_cols]
traindf = pd.concat([player_df[numerical_cols], pd.get_dummies(player_df[categorical_cols])],axis=1)
features = traindf.columns
traindf = traindf.dropna()
traindf = pd.DataFrame(traindf,columns=features)

# goals >= 87 => great player
y = traindf['Overall']>=87
X = traindf.copy()
del X['Overall']
X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x)) # lightGBS doesn't handle non ascii characters

# set parameters
feature_name=list(X.columns)
max_feats_num = 30
X_norm = MinMaxScaler().fit_transform(X) # Transform features by scaling each feature to a given range

```
#### Feature Importance Methods:
## 1. Pearson Correlation
**Filter based method.** 

The Pearson correlation coefficient to measure the strength of the relationship between two variables is defined as:

 $r=\frac{\sum(x_t-\overline{x})(y_t-\overline{y})}{\sqrt{\sum(x_t-\overline{x})^2}\sqrt{\sum(y_t-\overline{y})^2}}$

We check the absolute value of the Pearson’s correlation between the target and numerical features in our dataset. We keep the top n features based on this criterion:

```python
def pears_correl_selector(X:pd.DataFrame, y:pd.Series, max_feats_num:int):
    """
    Filter based selection.
    Check the absolute value of the Pearson’s correlation between the target and numerical features in our dataset.
    Return the top max_feats_num features.
    
    Returns
    --------
    cor_feature: list
    """
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for feature in feature_name:
        corellation = np.corrcoef(X[feature], y)[0, 1]
        cor_list.append(corellation)
    # replace NaN with 0
    cor_list = [0 if np.isnan(cor) else cor for cor in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-max_feats_num:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if feat in cor_feature else False for feat in feature_name]
    return cor_feature, cor_support
```

## 2. Chi-Squared Statistical Test
**Filter based method.**

We calculate the chi-square metric between the target and the numerical variable and only select the variable with the maximum chi-squared values. If a feature is independent to the target it is uniformative for classifying observations.

$X^2=\sum_{i=1}^n\frac{(O_i-E_i)^2}{E_i}$

$O_i$: the number of observations in a class $i$

$E_i$: the number of expected observations in class $i$ if there was no relationship between the feature and target

We generally calculate the squares as they willl eventually tell us if the difference between the expected and observation is statistically significant. It also works in a hand-wavy way with non-negative numerical and categorical features.

Find the code bellow:
```python
def chi_squared_selector(X:pd.DataFrame, X_norm: pd.DataFrame, y:pd.Series, max_feats_num:int):
    """
    Calculate the chi-square metric between the target and the numerical variable.
    Return the max_feats_num variables with the maximum chi-squared values
    
    Return
    --------
    chi_feature: list
    """
    chi_selector = SelectKBest(chi2, max_feats_num) # Select features according to the k highest scores.
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support() #Get a mask, or integer index, of the features selected
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_feature, chi_support
```

## 3. Recursive Feature Elimination
**Wrapper based method.** Considers the selection of features as a set of features as a search problem.

The `sklearn` RSE: The goal is to select features by **recursively considering smaller and smaller sets of features**. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a `coef_ attribute` or through a `feature_importances_ attribute`. Then, the least important features are pruned from current set of features. That procedure is recursively repeated on the pruned set **until the desired number of features to select** is eventually reached.

We can use any estimator with the method. Here, we use `LogisticRegression`, and the RFE observes the `coef_ attribute` of the `LogisticRegression` object:


```python
def rfe_selector(X:pd.DataFrame, X_norm: pd.DataFrame, y:pd.Series, max_feats_num:int):
    """
    Wrapper based selection. Recursively consider smaller and smaller sets of features until the max_feats_num number of features is eventually reached.
    Returns a list of the top max_feats_num features.
    
    Return
    --------
    rfe_feature: list
    """
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=max_feats_num, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_feature, rfe_support
```

## 4. Lasso: Select from Model
**Embedded method**. Model with a built-in feature selection method.
Definition:

$a\sum_{i=1}^n|w_i|$


Notes:
1. Lasso Regressor uses **L1 Norm** as regulizer: the sum of the magnitudes of the vectors in a space. It is the sum of absolute difference of the components of the vectors. Adds penalty equivalent to absolute value of the magnitude of coefficients (many weights are forced to zero, but the 'relevant' variables are allowed to have nonzero weights). This peranly factor determines how many features are retained; using cross-validation to choose the penalty factor helps assure that the model will generalize well to future data samples. The degree of sparsity is controlled by the penality term, and some procedure must be used to select it (cross-validation is a common choice).

2. Unlike [Ridge Regression](https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/), lasso's norm regularizer drives parameters to zero.
3. Higher values of **alpha** means fewer features have non-zero values

The code is the following:

```python
def lasso_selector(X:pd.DataFrame, X_norm: pd.DataFrame, y:pd.Series, max_feats_num:int):
    """
    Embeded feature selection. Use **L1 Norm** as regulizer.
    
    Return
    --------
    rfe_feature: list
    """
    lr_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'), max_features=max_feats_num) # select lasso by setting penalty=1
    lr_selector.fit(X_norm, y)
    lr_support = lr_selector.get_support()
    lr_feature = X.loc[:,lr_support].columns.tolist()
    return lr_feature, lr_support
```

## 5. Random Forest
**Embeded method.**

We can make use of the Random Forest classifier from `scikit-learn`. In this case, we exploit the `feature_importance_` feature of the algorithm.
The feature importance is calculating by using node impurities in each decision tree. In Random forest, the final feature importance is the average of all decision tree feature importance.
The code is the following:

```python
def random_forest_selector(X:pd.DataFrame, y:pd.Series, max_feats_num:int):
    """
    Embeded feature selection. Calculate feature importance using node impurities in each decision tree.
    
    Return
    --------
    rf_feature_feature: list
    """
    rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=max_feats_num)
    rf_selector.fit(X, y)
    rf_support = rf_selector.get_support()
    rf_feature = X.loc[:, rf_support].columns.tolist()
    return rf_feature, rf_support
```

## 6. Light GBM
**Embeded method.**
LightGBM is Microsoft's solution to **gradient boosting**. It is designed to be distributed and efficient.

The code is the following:
```python
def lightGBM_selector(X:pd.DataFrame, y:pd.Series, max_feats_num:int):
    """
    Embeded feature selection. Used XGBoost as a baseline with improved performance/
    
    Return
    --------
    lgb_feature_feature: list
    """
    
    lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    lgb_selector = SelectFromModel(lgbc, max_features=max_feats_num)
    lgb_selector.fit(X, y)
    lgb_support = lgb_selector.get_support()
    lgb_feature = X.loc[:,lgb_support].columns.tolist()
    return lgb_feature, lgb_support
```

## Comparing Important Feature Sets

We can compare all the methods mentioned above by having all the results gathered in a dataset:

```python
def create_feature_selection_df(feature_name, pearson_cor_support, chi_squared_support, rfe_support, lasso_support, rf_support, lgb_support, max_feats_num):
    feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_squared_support, 'RFE':rfe_support, 'Logistics': lasso_support,
                                    'Random Forest':rf_support, 'LightGBM':lgb_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df)+1)
    return feature_selection_df
```

Now we have all the results in a dataframe and we check if we get a feature based on all the methods. On top of this implementation we could create an evaluation function that finds the best feature selection algorithm for a specific dataset. In the next update of this article I plan to another algorithm that does that.

sources:
1. [The 5 Feature Selection Algorithms every Data Scientist should know](https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2)
2. [Feature Selection Using Football Data](https://www.kaggle.com/mlwhiz/feature-selection-using-football-data)
3. [Notes On Using Data Science & Machine Learning To Fight For Something That Matters
](https://chrisalbon.com/)
4. [Chi-Square Tests: Crash Course Statistics #29](https://www.youtube.com/watch?v=7_cs1YlZoug)
5. [Why LASSO for feature selection?](https://stats.stackexchange.com/questions/367155/why-lasso-for-feature-selection)
6. [Welcome to LightGBM’s documentation!](https://lightgbm.readthedocs.io/en/latest/)