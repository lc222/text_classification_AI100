# -*- coding: utf-8 -*-
import xgboost as xgb
import csv
import jieba
#jieba.load_userdict('wordDict.txt')
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# 读取训练集
def readtrain(path):
    with open(path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        column1 = [row for row in reader]
    content_train = [i[1] for i in column1] # 第一列为文本内容，并去除列名
    opinion_train = [int(i[0])-1 for i in column1] # 第二列为类别，并去除列名
    print '训练集有 %s 条句子' % len(content_train)
    train = [content_train, opinion_train]
    return train

def stop_words():
    stop_words_file = open('stop_words_ch.txt', 'r')
    stopwords_list = []
    for line in stop_words_file.readlines():
        stopwords_list.append(line.decode('gbk')[:-1])
    return stopwords_list

# 对列表进行分词并用空格连接
def segmentWord(cont):
    stopwords_list = stop_words()
    c = []
    for i in cont:
        text = ""
        word_list = list(jieba.cut(i, cut_all=False))
        for word in word_list:
            if word not in stopwords_list and word != '\r\n':
                text += word
                text += ' '
        c.append(text)
    return c

def segmentWord1(cont):
    c = []
    for i in cont:
        a = list(jieba.cut(i))
        b = " ".join(a)
        c.append(b)
    return c

train = readtrain('data/training.csv')
train_content = segmentWord1(train[0])
train_opinion = np.array(train[1])     # 需要numpy格式
print "train data load finished"
test = readtrain('data/testing.csv')
test_content = segmentWord(test[0])
print 'test data load finished'



vectorizer = CountVectorizer()
tfidftransformer = TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(train_content))
weight = tfidf.toarray()
print tfidf.shape
test_tfidf = tfidftransformer.transform(vectorizer.transform(test_content))
test_weight = test_tfidf.toarray()
print test_weight.shape


dtrain = xgb.DMatrix(weight, label=train_opinion)
dtest = xgb.DMatrix(test_weight)  # label可以不要，此处需要是为了测试效果
param = {'max_depth':6, 'eta':0.5, 'eval_metric':'merror', 'silent':1, 'objective':'multi:softmax', 'num_class':11}  # 参数
evallist  = [(dtrain,'train')]  # 这步可以不要，用于测试效果
num_round = 100  # 循环次数
bst = xgb.train(param, dtrain, num_round, evallist)
preds = bst.predict(dtest)
with open('XGBOOST_OUTPUT.csv', 'w') as f:
    for i, pre in enumerate(preds):
        f.write(str(i + 1))
        f.write(',')
        f.write(str(int(pre) + 1))
        f.write('\n')
