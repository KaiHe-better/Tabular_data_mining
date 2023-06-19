#  76.19%
# https://archive.ics.uci.edu/ml/datasets/in-vehicle+coupon+recommendation
#  numbers + text

import pandas 


import pandas as pd

df = pd.read_csv('in-vehicle-coupon-recommendation.csv')
print(df)


with open("in-vehicle-coupon-recommendation.csv") as f:
    data = f.readlines()


print(data[2])