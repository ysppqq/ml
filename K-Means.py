#encoding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

#生成用于聚类的2维数据1500个，默认有3个中心点
n_samples = 10
random_state =170
X,y = make_blobs(n_samples=n_samples,random_state=random_state)

#对X进行聚类
k_means = KMeans(n_clusters=3,random_state=random_state).fit(X)
print k_means.labels_
print k_means.cluster_centers_

#用已有中心点对X中的所有样本计算归属类别
y_pred1 = k_means.predict(X)
print y_pred1

#上述过程可简化为下方函数调用
y_pred2 = KMeans(n_clusters=3,random_state=random_state).fit_predict(X)
print y_pred2


plt.subplot(221)
plt.scatter(X[:,0],X[:,1],c=y_pred1)
plt.show()