#coding=utf8
import nltk
import json
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

documents_feature =json.load(open('tmp/documents_feature.txt', 'r'))
test_documents_feature = json.load(open('tmp/test_documents_feature.txt', 'r'))

print "开始训练分类器"
classifier = nltk.NaiveBayesClassifier.train(documents_feature)
#classifier = nltk.DecisionTreeClassifier.train(documents_feature)
#classifier = SklearnClassifier(SVC(), sparse=False).train(documents_feature[:4000])
#test_error = nltk.classify.accuracy(classifier, documents_feature)
#print "test_error:", test_error
#classifier.show_most_informative_features(20)
results = classifier.classify_many([fs for fs in test_documents_feature])

with open('output/TFIDF_out.csv', 'w') as f:
    for i in range(2381):
        f.write(str(i+1))
        f.write(',')
        f.write(str(results[i]+1))
        f.write('\n')