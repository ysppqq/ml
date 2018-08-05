#encoding:utf-8
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba

#数字型特征预处理
L = []
for i in range(10):
    tmp=[]
    for n in range(i,i+5):
        tmp.append(n)
    L.append(tmp)
#print L
#print np.array(L)
#标准化
L_scared = preprocessing.scale(L)
#正则化
L_normalized = preprocessing.normalize(L)
#归一化
L_minmaxscaler = preprocessing.MinMaxScaler().fit_transform(L)

#print L_scared
#print L_normalized
#print L_minmaxscaler


#文本型特征提取
#初始化对象，用于从字典型特征中提出
measurements = [
    {'city':'Dubai','temperature':33.},
    {'city':'Beijing','temperature':21.},
    {'city':'Londan','temperature':12.}
]
vec = DictVectorizer()
r = vec.fit_transform(measurements)
print r.toarray()
print vec.get_feature_names()

#初始化对象，用于从列表类文本中生成词汇表和特征
vectorizer = CountVectorizer(min_df=1)
corpus = [
    'today is saturday!',
    'Are you OK?',
    'Thank you.'
]
#对给定文本进行处理
x = vectorizer.fit_transform(corpus)
#特征化结果
print x.toarray()
#获取分词结果
print vectorizer.get_feature_names()

#对其他文本提取特征
test = ['hello','I like you you!']
x= vectorizer.transform(test)
print x.toarray()

#获取词汇表
vocabulary = vectorizer.vocabulary_
print vocabulary

#利用已有词汇表初始化新的对象，用于特征化其他文本
new_vec = CountVectorizer(min_df=1,vocabulary=vocabulary)

test = ['hello','I like you you!']
r = new_vec.fit_transform(test)
print new_vec.get_feature_names()
print r.toarray()

#对中文分词应在初始化对象是传入中文分词器，此处使用jieba
vectorizer = CountVectorizer(min_df=1, tokenizer=jieba.cut)
corpus = [
    '你好!',
    '我晚上请你吃饭。',
    '今天天气真好！'
]
#对给定文本进行处理
x = vectorizer.fit_transform(corpus)
#特征化结果
print x.toarray()
print vectorizer.get_feature_names()
#获取分词结果
for i in vectorizer.get_feature_names():
    print i

