# 66.80%
# https://archive.ics.uci.edu/ml/datasets/online+news+popularity
# all numbers



import pandas 


import pandas as pd

df = pd.read_csv('OnlineNewsPopularity.csv')
print(df)


with open("OnlineNewsPopularity.csv") as f:
    data = f.readlines()


print(data[2])