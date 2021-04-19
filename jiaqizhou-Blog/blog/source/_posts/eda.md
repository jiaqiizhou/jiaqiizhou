---
title: 智慧海洋-数据分析
date: 2021-04-17 09:28:21
tags: 
    - DataWhale
    - EDA
    - AIS
---

### 数据分析的目的

* EDA的主要价值在于熟悉整个数据集的基本情况(缺失值、异常值)，来确定所获得数据集可以用于接下来的机器学习或者深度学习使用。
* 了解特征之间的相关性、分布，以及特征与预测值之间的关系。

``` python
class Load_Save_Data():
    def __init__(self,file_name=None):
        self.filename = file_name
    def load_data(self,Path=None): 
        if Path is None:
            assert self.filename is not None,"Invalid Path...." 
        else:
            self.filename = Path
        with open(self.filename,"wb") as f:
            data = pickle.load(f) 
        return data
    def save_data(self,data,path): 
        if path is None:
            assert self.filename is not None,"Invalid path...." 
        else:
            self.filename = path
        with open(self.filename,"wb") as f:
            pickle.dump(data,f)
```
``` python
# 定义读取数据的函数
def read_data(Path,Kind=""): """
    :param Path:待读取数据的存放路径 :param Kind:'train' of 'test' """
    # 替换成数据存放的路径
    filenames = os.listdir(Path)
    print("\n@Read Data From"+Path+".........................") with mp.Pool(processes=mp.cpu_count()) as pool:
        data_total = list(tqdm(pool.map(read_all_data.read_train_file if Kind == "train" else read_all_data.read_test_file,filenames),total= len(filenames
    print("\n@End Read total Data............................") load_save = Load_Save_Data()
    if Kind == "train":
    load_save.save_data(data_total,"./data_tmp/total_data.pkl") return data_total
```
```python
# 训练数据读取
# 存放数据的绝对路径
train_path = "D:/code_sea/data/train/hy_round1_train_20200102/" data_train = read_data(train_path,Kind="train")
data_train = pd.concat(data_train)
# 测试数据读取
# 存放数据的绝对路径
test_path = "D:/code_sea/data/test/hy_round1_testA_20200102/" data_test = read_data(test_path,Kind="test")
data_test = pd.concat(data_test)
```
### 总体了解数据
```python
data_test.shape
data_train.shape
data_train.columns
```
查看一下具体的列名，赛题理解部分已经给出具体的特征含义，这里方便阅读再给一下

* 渔船ID:渔船的唯一识别，结果文件以此ID为标识 
* x:渔船在平面坐标系下的x轴坐标 
* y:渔船在平面坐标系下的y轴坐标 
* 速度:渔船当前时刻的航速，单位节 
* 方向:渔船当前时刻的航首向，单位度 
* time:数据上报时刻，单位月日 时:分 
* type:渔船label，作业类型
```python
pd.options.display.max_info_rows = 2699639 
data_train.info(）
```
```python
data_train.describe([0.01,0.025,0.05,0.5,0.75,0.9,0.99])
```
```python
data_train.head(3).append(data_train.tail(3))
```
### 查看数据集中特征缺失值、唯一值等

### 数据特性和特征分布
#### 三类渔船轨迹可视化

```python
# 把训练集的所有数据,根据类别存放到不同的数据文件中
def get_diff_data():
    Path = "./data_tmp/total_data.pkl" 
    with open(Path,"rb") as f:
        total_data = pickle.load(f) 
    load_save = Load_Save_Data()
    kind_data = ["刺网","围网","拖网"]
    file_names = ["ciwang_data.pkl","weiwang_data.pkl","tuowang_data.pkl"] 
    for i,datax in enumerate(kind_data):
        data_type = [data f o r data i n total_data i f data["type"].unique()[0] = = datax] load_save.save_data(data_type,"./data_tmp/" + file_names[i])
get_diff_data()
```
```python
# 从存放某个轨迹类别的数据文件中，随机读取某个渔船的数据
def get_random_one_traj(type=None): """
    :param type:"ciwang","weiwang" or "tuowang"
    """
    np.random.seed(10)
    path = "./data_tmp/"
    with open(path + type + ".pkl","rb") as f1:
        data = pickle.load(f1) length = len(data)
        index = np.random.choice(length) return data[index]
```