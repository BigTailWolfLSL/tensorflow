from sklearn import datasets# 可能需要先安装scikit-learn
from pandas import DataFrame
import pandas as pd

x_data = datasets.load_iris().data # 返回所有输入特征->数据
y_data = datasets.load_iris().target # 返回iris数据集所有标签->结果
print("x_data from datasets: \n", x_data)
print("y_data from datasets: \n", y_data)

x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度']) #增加标签
pd.set_option('display.unicode.east_asian_width', True) #设置列名对齐
print("x_data add index: \n", x_data)

x_data['类别'] = y_data
print("x_data add a column: \n", x_data) # 在x_data中加一列为类别的数据