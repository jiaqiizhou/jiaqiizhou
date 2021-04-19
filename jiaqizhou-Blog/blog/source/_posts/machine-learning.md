---
title: 机器学习-基础
tags: 
    - DataWhale
    - machine learning
---
## 导论
* 机器学习
    * 有监督学习 ：有因变量，有特征向量，预测结果。  
        *  回归：因变量是连续型变量，如：房价、体重等。
        *  分类：因变量是离散型变量，如：是否患癌症，西瓜是好瓜 or 坏瓜。
    * 无监督学习 ：无因变量，有特征向量，寻找数据中的结构。

## 投票法的原理分析
投票法是一种遵循少数服从多数原则的集成学习模型，通过多个模型的集成降低方差，从而提高模型的鲁棒性。理想情况下，投票法的预测效果优于任何一个基模型的预测效果。  
投票法在回归模型与分类模型上均可使用：  
* 回归投票法：预测结果是所有模型预测结果的平均值。  
* 分类投票法：预测结果是所有模型中出现最多的预测结果。

分类投票法又可以划分为硬投票与软投票：  
* 硬投票：预测结果是所有投票结果最多出现的类。
* 软投票：预测结果是所有投票结果中出现概率加和最大的类。

例子：  
 硬投票：对于某个样本 ，模型1的预测结果是A， 模型2的预测结果是B，模型3的预测结果是B。 硬投票法的预测结果是B。  
 软投票：model 1 类型A的概率是99%，model 2 类型A的概率是49%，model 3类型A的概率是49%。 A的预测概率的平均是（99+49+49）/3 = 65.67%。
 软投票考虑到预测概率这一额外信息，因此比硬投票法更加准确的预测结果。  

在投票法中，需要考虑到不同的基模型可能产生的影响。理论上，基模型可以是任何已被训练好的模型，在实际应用上，想要投票法产生较好的结果，需要满足两个条件：  
* 基模型之间的效果不能差别过大，当某个及模型相对于其他基模型效果过差时，该模型很可能成为噪声。
* 基模型之间应该有较小的同质性， 例如在基模型预测效果近似的情况下，基于树模型与线性模型的投票，往往优于两个树模型或两个线性模型。

当投票集合中使用的模型能预测出清晰的类别标签时，适合使用硬投票。  
当投票集合使用的模型能预测类别的概率时，适合使用软投票。软投票同样可以用于那些本身并不预测类成员概率的模型，只要他们可以输出类似于概率的预测分数值（SVM，k-最近邻和决策树）

## 投票法案例
sklearn中两种投票方法 VotingRegressor和VotingClassifier两个投票方法。这两种模型的操作方式相同，并采用相同的参数，使用模型需要提供一个模型列表，列表中每个模型采用Tuple的结构表示，第一个元素代表名称，第二个元素代表模型，需要保证每个模型必须拥有唯一的名称。

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 创建1000个样本，20个特征的随机数据集
# test classification dataset
from sklearn.datasets import make_classification
# define dataset
def get_dataset():
    X, y = make_classification(n_samples = 1000, n_features = 20, n_informative = 15, n_redundant=5, random_state = 2)
    # summarize the dataset
    return X,y
X,y = get_dataset() 

# 使用多个KNN模型作为基模型演示投票法，其中每个模型采用不同邻居值K参数：
# get a voting ensemble of models
from sklearn.neighbors import KNeighborsClassifier
def get_voting():
    # define the base models
    models = list()
    models.append(('knn1', KNeighborsClassifier(n_neighbors = 1)))
    models.append(('knn3', KNeighborsClassifier(n_neighbors = 3)))
    models.append(('knn5', KNeighborsClassifier(n_neighbors = 5)))
    models.append(('knn7', KNeighborsClassifier(n_neighbors = 7)))
    models.append(('knn9', KNeighborsClassifier(n_neighbors = 9)))
    # define the voting ensemble
    ensemble = VotingClassifier(estimators = models, voting = "hard")
    return ensemble

# 创建模型列表，包括每个基模型和硬投票模型
# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn1'] = KNeighborsClassifier(n_neighbors = 1)
    models['knn3'] = KNeighborsClassifier(n_neighbors = 3)
    models['knn5'] = KNeighborsClassifier(n_neighbors = 5)
    models['knn7'] = KNeighborsClassifier(n_neighbors = 7)
    models['knn9'] = KNeighborsClassifier(n_neighbors = 9)
    models['hard_voting'] = get_voting()
    return models

# evaluate a give model using cross_valiation
# 分层10倍交叉验证三次重复的分数列表的形式返回
from sklearn.model_selection import  RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
def evaluate_model(model,X,y):
    cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3, random_state = 1)
    scores = cross_val_score(model, X, y ,scoring = "accuracy", cv = cv, n_jobs = -1, error_score = "raise")
    return scores

import numpy as np
models = get_models()
results, names = list(), list()
for name, model in models.items():
    scores = evaluate_model(model,X,y)
    results.append(scores)
    names.append(name)
    print(">%s %.3f (%.3f)"%(name, np.mean(scores), np.std(scores)))
from matplotlib import pyplot
pyplot.boxplot(results, labels = names, showmeans = True)
pyplot.show()
```
![输出结果](voting_result.jpg)