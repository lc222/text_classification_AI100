import pandas as pd

data = pd.read_csv('data/training.csv')
for i in range(11):
    print data[data['label'] == i+1].count()
