#coding=utf8
import jieba

def stop_words():
    stop_words_file = open('stop_words_ch.txt', u'r')
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

stopwords_list = stop_words()

result = []
qq = 0
with open('data/testing.csv', u'r') as f:
    for line in f:
        content = ""
        for aa in line.split(',')[1:]:
            content += aa
        word_list, word_set = jieba_fenci(content, stopwords_list)
        label = 2
        count = 0
        for i, cla in enumerate(feature_words):
            tmp = 0
            for word in word_list:
                if word in cla:
                    tmp += 1
            if tmp > count:
                count = tmp
                label = i+1
        if count == 0:
            qq += 1
        result.append(label)
print qq
with open('output/feature_match_output.csv', u'w') as g:
    for i, res in enumerate(result):
        g.write(str(i+1))
        g.write(',')
        g.write(str(res))
        g.write('\n')