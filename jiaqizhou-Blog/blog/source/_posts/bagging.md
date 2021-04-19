---
title: 集成学习-bagging
mathjax: true
tags: 
    - DataWhale
    - ensemble  
---
与投票法不同，Bagging不仅仅集成模型最后的预测结果，同时采用一定策略来影响基模型训练，保证基模型可以服从一定的假设。集成学习我们希望各个模型之间具有较大的差异性，而在实际操作中的模型往往是同质的，因此一个简单的思路是通过不同的采样增加模型的差异性。
## bagging的原理分析
bagging的核心在于<font color = 'red'>自主采样（bootstrap）</font>，即有放回的从数据集中进行采样，也就是说，同样一个样本可能被多次进行采样。一个自主采样的例子是希望估计全国所有人口年龄的平均值，那么我们在全国所有人口中随机抽取不同的集合（这些集合可能存在交集），计算每个集合的平均值，然后将所有平均值的均值作为估计值。
### bagging的基本流程
* 随机取出一个样本放入采样集合中，再把这个样本放回初始数据集，重复K次采样。最终获得一个大小为K的样本集合。
* 同样的方法，可以采样出T个含K个样本的采样集合，然后基于每个采样集合训练出一个基学习器，再将这些基学习器进行结合，就是Bagging的基本流程。

对于回归问题的预测是通过预测取平均值来进行的，对于分类问题的预测是通过预测取多数票预测来进行的。 Bagging方法之所以有效，是因为每个模型都是再略微不同的训练数据集上拟合完成的，这又使得每个基模型之间存在略微的差异，使每个基模型拥有略微不同的训练能力。  
bagging同样是一种降低方差的技术，因此它在不剪枝决策树、神经网络等易受样本扰动的学习器上效果更将明显，在实际的使用中，加入列采样的Bagging技术对高维小样本往往有神奇的效果。

行采样，列采样是随机森林、XGBoost等集成模型中常用的trick。主要作用是加快训练速度和防止过拟合。  
随机森林：1）建树前对样本随机抽样（行采样）2）每个特征分裂随机采样生成特征候选集（列采样）3）根据增益公式选取最优分裂特征和对应特征分裂建树。建树过程完全独立。能够完全并行化。
## bagging的案例分析
Sklearn 提供了BaggingRegressor和BaggingClassifier两种Bagging方法的API，这两种方法的默认基模型是树模型。树一般指的是决策树，是一种树形结构，树的每个非叶子节点表示对样本在一个特征上的判断，节点下方的分支代表对样本的划分。决策树的建立过程是对一个数据不断划分的过程，每次划分中，首先要选择用于划分的特征，之后要确定划分的方案（类别/阈值）。我们希望通过划分，决策树的分支节点所包含的样本“纯度”尽可能高，节点划分过程中所用的指标主要是<font color = 'red'>信息增益</font>和<font color = 'red'>GINI系数</font>  

### 决策树分裂

信息增益： 衡量的是划分前后不确定性程度的减小，信息不确定程度一般使用信息熵来度量，计算方式是：
$H(Y) = -\sum{p_{i}logp_{i}}$
i表示样本的标签，p表示该样本出现的概率，当我们对样本进行划分之后，计算样本的条件熵：
$H(Y|X) = -\sum_{x \in X}{p(X = x)H(Y|X = x)}$
其中X表示用于划分特征的取值。信息增益定义为信息熵与条件熵的差值：
$IG = H(Y) - H(Y|X)$ 
信息增益IG越大，说明使用该特征划分数据所获得的信息量变化越大，子节点的样本”纯度“越高。
同样，也可以利用Gini指数来衡量数据的不纯度，计算方法 $Gini = 1 - \sum p_{i}^{2}$
对样本做出划分后，计算划分后的Gini指数 $Gini_{x} = \sum_{x \in X}p(X = x)[1 - \sum p_{i}^2]$
选择使得划分后Gini指数最小的特征。
#### 分裂举例
![决策树](决策树.jpg)

根据天气、温度和风力等级判断是否打网球：首先通过计算信息增益 or Gini指数确定首先根据天气情况对样本进行划分，之后对于每个分支，继续考虑除天气之外的其他特征，直到样本的类型被完全分开，所有特征都已使用，或达到树的最大深度为止。
Bagging的一个典型应用是随机森林，”森林“是许多树bagging组成，具体实现上，用于每个决策树训练的样本和构建决策树的特征都是通过随机采样得到的，RF的预测结果是多个决策树输出的组合（投票）

![决策树](决策树过程.jpg)

### sklearn bagging 代码
```python
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=1000, n_features=20, n_informative=15,n_redundant=5, random_state=5)
print(X.shape,y.shape)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
import numpy as np
# define the model
model = BaggingClassifier()
# evaluate the model
# 使用重复的分层k-fold交叉验证来评估模型，一共重复3次，每次有10个fold，
cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3, random_state = 1)
n_scores = cross_val_score(model, X, y, scoring = "accuracy", cv = cv, n_jobs = -1, error_score = "raise")
# report performance
print("Accuracy : %.3f(%.3f)"%(np.mean(n_scores), np.std(n_scores)))
```
(1000, 20) (1000,)  
Accuracy : 0.858(0.034)

### **<font face = "微软雅黑"  size = 3 color = #BA55D3 > sklearn.ensemble.BaggingClassifier </font>**  
#### Parameters
* base_estimator : object, default=None  
The base estimator to fit on random subsets of the dataset. If None, then the base estimator is a DecisionTreeClassifier.
* n_estimators : int, default=10   
The number of base estimators in the ensemble.
* max_samples : int or float, default=1.0  
The number of samples to draw from X to train each base estimator (with replacement by default, see bootstrap for more details).
* max_features : int or float, default=1.0  
The number of features to draw from X to train each base estimator ( without replacement by default, see bootstrap_features for more details).
    * If int, then draw max_features features.
    * If float, then draw max_features * X.shape[1] features.
* bootstrap : bool, default=True  
Whether samples are drawn with replacement. If False, sampling without replacement is performed.  

### **<font face = "微软雅黑"  size = 3 color = #BA55D3 > sklearn.datasets.make_classifsication </font>**  

Generate a random n-class classification problem.  
创建一个长度为2*class_sep 且边长为 2 * class_sep 的正态分布于n_informative-维超立方体的顶点的点簇（std =1），并为每个类分配相等数量的簇。  
n_informative 特征， n_redundant 线性的信息特征组合，n_repeated 重复，从信息和冗余特征中随机替换。
 
#### Parameters
* n_samples: int, default = 100, the number of samples.
* n_features: int, default = 20, The total number of features. 
* n_informative:int， default = 2
* n_redundant:int， default = 2
* n_repeated:int， default = 0
* n_classes: int, default = 2
* n_cluster_per_class, int, default = 2
* weights: array-like of shape(n_classes, ) or (n_classes-1, ), default = None
* flip_y: float, default = 0.01
* class_sep：float, default = 1.0
* hypercube: bool, default = True
* shift:float, ndarray of shape (n_features,) or None, default=0.0
* scale: float, ndarray of shape (n_features,) or None, default=1.0
* shuffle: bool, default=True
* random_state: int, RandomState instance or None, default=None

#### Returns

* X : ndarray of shape \(n\_samples, n\_features\)  
    The generated samples
* y : ndarray of shape \(n\_samples\)  
    The integer labels for class membership of each sample.

### **<font face = "微软雅黑"  size = 3 color = #BA55D3 > sklearn.model_selection.RepeatedStratifiedKFold </font>**  

#### 为什么使用交叉验证（监督学习器性能评估方法）
一般情况下，我们将原始数据集分为训练数据集和验证数据集。防止训练过程中数据泄露，从训练数据集中划分出一部分数据，validation set（验证数据集），用来评估模型。最后再用测试集检验模型的泛化能力。  
但是把原始数据集分割之后，用来训练模型的数据集大大减小，同时训练结果也更大的依赖于训练数据集和测试数据集占原始数据集的比重。 解决方法即 cross-validation 交叉检验，缩写cv。  
k-fold cv的基本方法中，训练集被划分为k个较小的集合k-folds. 分别让一个fold作为测试集，余下部分作为训练集，进行k次训练，共计得到k个参数。最终使用均值作为最终的模型参数。  

![交叉验证](cv.jpg)

缺点：相同大小的数据集，需要进行更多的运算。
优点：最大特点是不浪费validation set大小的数据，尤其是在样本集不够大的情况下。
```python
class sklearn.model_selection.RepeatedStratifiedKFold(*, n_splits=5, n_repeats=10, random_state=None)
```
#### Parameters
* n_splits : int, default=5, Number of folds. Must be at least 2.
* n_repeats : int, default=10, Number of times cross-validator needs to be repeated.
* random_state : int, RandomState instance or None, default=None

### **<font face = "微软雅黑"  size = 3 color = #BA55D3 > sklearn.model_selection.cross_val_score </font>**  
Evaluate a score by cross-validation
#### Parameters
* estimator：estimator object implementing ‘fit’.
* X: array-like of shape \( n\_samples, n\_features\).
* y: array-like of shape \(n\_samples,\) or \(n\_samples, n\_outputs\), default=None.
* groups: array-like of shape \(n\_samples,\), default=None.
* scoring: str or callable, default=None.
* cv: int, cross-validation generator or an iterable, default=None
    * None, to use the default 5-fold cross validation
    * int, to specify the number of folds in a \(Stratified\)KFold
    * CV splitter
    * An iterable yielding \(train, test\) splits as arrays of indices.

#### Returns
* scores : ndarray of float of shape=(len(list(cv)),)
    * Array of scores of the estimator for each run of the cross validation.


