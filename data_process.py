# -*- coding: utf-8 -*-
import jieba
import nltk
from math import log
import numpy as np
import json

N = 4774
feature_words = [[u'物业管理', u'物业', u'房地产', u'顾问', u'中介', u'住宅', u'商业', u'开发商', u'招商', u'营销策划'],
                 [u'私募', u'融资', u'金融', u'贷款', u'基金', u'股权', u'资产', u'小额贷款', u'投资', u'担保'],
                 [u'软件', u'互联网', u'平台', u'信息化', u'软件开发', u'数据', u'移动', u'信息', u'系统集成', u'运营'],
                 [u'制造', u'安装', u'设备', u'施工', u'机械', u'工程', u'自动化', u'工业', u'设计', u'装备'],
                 [u'药品', u'医药', u'生物', u'原料药', u'药物', u'试剂', u'GMP', u'片剂', u'制剂', u'诊断'],
                 [u'材料', u'制品', u'塑料', u'环保', u'新型', u'化学品', u'改性', u'助剂', u'涂料', u'原材料'],
                 [u'养殖', u'农业', u'种植', u'食品', u'加工', u'龙头企业', u'产业化', u'饲料', u'基地', u'深加工'],
                 [u'医疗器械', u'医疗', u'医院', u'医用', u'康复', u'治疗', u'医疗机构', u'临床', u'护理'],
                 [u'汽车', u'零部件', u'发动机', u'整车', u'模具', u'C36', u'配件', u'总成', u'车型'],
                 [u'媒体', u'制作', u'策划', u'广告', u'传播', u'创意', u'发行', u'影视', u'电影', u'文化'],
                 [u'运输', u'物流', u'仓储', u'货物运输', u'货运', u'装卸', u'配送', u'第三方', u'应链', u'集装箱']]

# 读取停词表
def stop_words():
    stop_words_file = open('stop_words_ch.txt', 'r')
    stopwords_list = []
    for line in stop_words_file.readlines():
        stopwords_list.append(line.decode('gbk')[:-1])
    return stopwords_list

def jieba_fenci(raw, stopwords_list):
    # 使用结巴分词把文件进行切分
    word_list = list(jieba.cut(raw, cut_all=False))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    # word_set用于统计A[nClass]
    word_list.remove('\n')
    word_set = set(word_list)
    return word_list, word_set

def process_file(train_path, test_path):
    '''
    本函数用于处理样本集中的所有文件。并返回处理结果所得到的变量
    :param floder_path: 样本集路径
    :return: A：CHI公示中的A值，嵌套字典。用于记录某一类中包含单词t的文档总数。第一层总共9个key，对应9类新闻分类
                第二层则是某一类中所有单词及其包含该单词的文档数（而不是出现次数）。{{1：{‘hello’：8，‘hai’：7}}，{2：{‘apple’：8}}}
            TFIDF：用于计算TFIDF权值。三层嵌套字典。第一层和A一样，key为类别。第二层的key为文件名（这里使用文件编号代替0-99）.第三层
                    key为单词，value为盖单词在本文件中出现的次数。用于记录每个单词在每个文件中出现的次数。
            train_set:训练样本集。与测试样本集按7:3比例分开。三元组（文档的单词表，类别，文件编号）
            test_set:测试样本集。三元组（文档的单词表，类别，文件编号）
    '''
    stopwords_list = stop_words()
    # 用于记录CHI公示中的A值
    A = {}
    tf = []
    i=0
    # 存储训练集/测试集
    count = [0]*11
    train_set = []
    test_set = []
    with open(train_path, 'r') as f:
        for line in f:
            tf.append({})
            label = int(line.split(',')[0])-1
            if label not in A:
                A[label] = {}
            count[label] += 1
            content = ""
            for aa in line.split(',')[1:]:
                content += aa
            word_list, word_set = jieba_fenci(content, stopwords_list)
            train_set.append((word_list, label))
            for word in word_set:
                if A[label].has_key(word):
                    A[label][word] += 1
                else:
                    A[label][word] = 1
            for word in word_list:
                if tf[i].has_key(word):
                    tf[i][word] += 1
                else:
                    tf[i][word] = 1
            i += 1
        print "处理完数据"

    tf2 = []
    j = 0
    with open(test_path, 'r') as g:
        for line in g:
            tf2.append({})
            label = int(line.split(',')[0])-1
            content = ""
            for aa in line.split(',')[1:]:
                content += aa
            word_list, word_set = jieba_fenci(content, stopwords_list)
            test_set.append((word_list, label))
            for word in word_list:
                if tf2[j].has_key(word):
                    tf2[j][word] += 1
                else:
                    tf2[j][word] = 1
            j += 1
    return A, tf, tf2, train_set, test_set, count


def calculate_B_from_A(A):
    '''
    :param A: CHI公式中的A值
    :return: B，CHI公职中的B值。不是某一类但是也包含单词t的文档。
    '''
    B = {}
    for key in A:
        B[key] = {}
        for word in A[key]:
            B[key][word] = 0
            for kk in A:
                if kk != key and A[kk].has_key(word):
                    B[key][word] += A[kk][word]
    return B

def feature_select_use_new_CHI(A, B, count):
    '''
    根据A，B，C，D和CHI计算公式来计算所有单词的CHI值，以此作为特征选择的依据。
    CHI公式：chi = N*（AD-BC）^2/((A+C)*(B+D)*(A+B)*(C+D))其中N,(A+C),(B+D)都是常数可以省去。
    :param A:
    :param B:
    :return: 返回选择出的1000多维特征列表。
    '''
    word_dict = []
    word_features = []
    for i in range(0, 11):
        CHI = {}

        M = N - count[i]
        for word in A[i]:
            #print word, A[i][word], B[i][word]
            temp = (A[i][word] * (M - B[i][word]) - (count[i] - A[i][word]) * B[i][word]) ^ 2 / (
            (A[i][word] + B[i][word]) * (N - A[i][word] - B[i][word]))
            CHI[word] = log(N / (A[i][word] + B[i][word])) * temp
        #每一类新闻中只选出150个CHI最大的单词作为特征
        a = sorted(CHI.iteritems(), key=lambda t: t[1], reverse=True)[:100]
        b = []
        for aa in a:
            b.append(aa[0])
        word_dict.extend(b)
        for word in word_dict:
            if word not in word_features:
                word_features.append(word)
    return word_features

def document_features(word_features, TF, data, num):
    '''
    计算每一篇新闻的特征向量权重。即将文件从分词列表转化为分类器可以识别的特征向量输入。
    :param word_features:
    :param TFIDF:
    :param document: 分词列表。存储在train_set,test_set中
    :param cla: 类别
    :param num: 文件编号
    :return: 返回该文件的特征向量权重
    '''
    document_words = set(data)
    features = {}
    for i, word in enumerate(word_features):
        if word in document_words:
            features[word] = 1#TF[num][word]#*log(N/(A[cla][word]+B[cla][word]))
        else:
            features[word] = 0
    return features

A, tf, tf2, train_set, test_set, count = process_file('data/training.csv', 'data/testing.csv')
B = calculate_B_from_A(A)
print "开始选择特征词"
word_features = feature_select_use_new_CHI(A, B, count)
#print word_features
print len(word_features)
for word in word_features:
    print word

print "开始计算文档的特征向量"
documents_feature = [(document_features
                      (word_features, tf, data[0], i), data[1])
                     for i, data in enumerate(train_set)]

print "测试集"
test_documents_feature = [document_features(word_features, tf2, data[0], i)
                          for i, data in enumerate(test_set)]

json.dump(documents_feature, open('tmp/documents_feature.txt', 'w'))
json.dump(test_documents_feature, open('tmp/test_documents_feature.txt', 'w'))


# print "开始训练分类器"
# classifier = nltk.NaiveBayesClassifier.train(documents_feature[:4000])
# test_error = nltk.classify.accuracy(classifier, documents_feature[4000:4773])
# print "test_error:", test_error
# classifier.show_most_informative_features(20)
# results = classifier.prob_classify_many([fs for fs in test_documents_feature])
# print results
# with open('output/TFIDF_out.csv', 'w') as f:
#     for i in range(2381):
#         f.write(str(i+1))
#         f.write(',')
#         f.write(str(results[i] + 1))
#         f.write('\n')