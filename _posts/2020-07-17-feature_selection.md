---
layout: post
title: Feature Selection (In Progress)
author: Maria Pantsiou
date: '2020-07-17 14:35:23 +0530'
category: dataScience
summary: Feature Selection (In Progress)
thumbnail:

---

# Feature Selection Algorithms
*This article is about selecting the best set of features for your model. I did some research on this topic and ended up finding really interesting resources that mention different algotihms or a compination of some of them to create a robust feature selection tool. Take a look on what I've found:*


# Why Feature Selection?
1. **When having too many features the model overfits due to the curse of dimentionality.** If we have more columns in the data than the number of rows, we will be able to fit our training data perfectly, but that won’t generalize to the new samples. And thus we learn absolutely nothing
2. **Occam’s Razor**: We want our models to be simple and explainable, and having multiple features doesn't help to that
3. **Garbage In Garbage out**: Poor-quality input will produce Poor-Quality output


Three categories of feature selection process:
1. **Filter Based**: An example of such a metric could be correlation/chi-square
2. **Wrapper-based**: Wrapper methods consider the selection of a set of features as a search problem. Example: *Recursive Feature Elimination*
3. **Embedded**: Embedded methods use algorithms that have built-in feature selection methods. For instance, *Lasso* and *RF* have their own feature selection methods.

## Implementation
Use the following code to prepare the example dataset for this implementation:

```py

```


sources:
1. [The 5 Feature Selection Algorithms every Data Scientist should know](https://towardsdatascience.com/the-5-feature-selection-algorithms-every-data-scientist-need-to-know-3a6b566efd2)