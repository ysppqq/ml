#encoding:utf-8
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

#生成测试样本
centers = [[10,10],[-10,-10],[10,-10]]
X,labels_true = make_blobs(n_samples=1000,centers=centers,cluster_std=0.4,random_state=0)
#以下与scale(X)作用相似
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3,min_samples=10).fit(X)
#数据的聚类标签
print db.labels_
#官方文档说明为核心点的序号，但实测是所有点的序号？
print len(db.core_sample_indices_)


