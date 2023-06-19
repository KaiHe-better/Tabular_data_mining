#  87.36%
# https://archive.ics.uci.edu/ml/datasets/adult
#  numbers + text

import pandas 


import pandas as pd

df = pd.read_csv('adult.data')
print(df)


with open("adult.data") as f:
    data = f.readlines()


print(data[2])