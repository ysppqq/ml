#encoding:utf-8
from sklearn.feature_extraction.text import CountVectorizer
import jieba

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
