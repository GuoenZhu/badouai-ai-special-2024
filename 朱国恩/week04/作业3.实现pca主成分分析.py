import numpy as np
import pandas as pd

#读取数据
Data = pd.read_excel("农村居民人均可支配收入来源2016.xlsx")
#print(Data)
X = Data.iloc[:,1:]
print(X)
#计算相关系数矩阵
R = X.corr()
print("相关系数矩阵为\n",R)

#数据规范化处理
#1．导入均值-方差规范化模块StandardScaler
from sklearn.preprocessing import StandardScaler
#2．利用StandardScaler创建均值-方差规范化对象scaler
scaler = StandardScaler()
#3．调用scaler对象中的fit()拟合方法，对待处理的数据X进行拟合训练
scaler.fit(X) 
#4．调用scaler对象中的transform()方法，返回规范化后的数据集X（覆盖原未规范化的X）
X=scaler.transform(X)  
#5.输出规范化后的数据集X
print(X)

#对标准化后的数据X作主成分分析
#1.导入主成分分析模块PCA
from sklearn.decomposition import PCA   
#2.利用PCA创建主成分分析对象pca
pca = PCA(n_components=0.95) #这里设置累计贡献率为95%以上。
#3.调用pca对象中的fit()方法，对待分析的数据进行拟合训练
pca.fit(X)
#4.调用pca对象中的transform()方法，返回提取的主成分
Y = pca.transform(X)
print(Y)

#(5)通过pca对象中的components_属性、explained_variance_属性、explained_variance_ratio_属性，
#返回主成分分析中对应的特征向量、特征值和主成分方差百分比（贡献率），
tzx1 = pca.components_
print("主成分分析中对应的特征向量\n",tzx1)

tz = pca.explained_variance_
print("特征值\n",tz)

gx1 = pca.explained_variance_ratio_
print("主成分方差百分比\n",gx1)

#主成分表达式及验证
Y00  = sum(X[0,:]*tzx1[0,:])
print(Y00)
