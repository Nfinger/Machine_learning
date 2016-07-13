import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
reg = LinearRegression()


df = pd.read_csv("sphist.csv")
df["Date"] = pd.to_datetime(df["Date"])
# df = df.sort_values(df.index,inplace = False,ascending=False)
df = df.iloc[::-1]
#5 day mean
m = df["Close"].rolling(window=5)
df["day_5"] = m.mean()

#30 day mean
m = df["Close"].rolling(window=20)
df["day_30"] = m.mean()

#365 day mean
m = df["Close"].rolling(window=252)
df["day_365"] = m.mean()

df = df.iloc[::-1]
df = df[df["Date"] > datetime(year=1951, month=1, day=2)]
df = df.dropna(axis=0)

train = df[df["Date"] < datetime(year=2013, month=1, day=1)]
test = df[df["Date"] >= datetime(year=2013, month=1, day=1)]

predictors = train[["day_5","day_30","day_365"]]
to_fit = train[["Close"]]
to_predict = test[["Close"]]
predict = dict()
for item in predictors:
	reg.fit(train[[item]],to_fit)
	predict[item] = reg.predict(to_predict)
mse = dict()
for key,value in predict.items():
	_sum = 0
	diff = (value - to_predict) ** 2
	_sum = diff.sum()
	mse[key] = _sum/len(value)

import matplotlib.pyplot as plt

# Make a scatterplot with the actual values in the training set
plt.scatter(train["day_5"], train["Close"])
plt.plot(train["day_5"], reg.predict(train[["Close"]]))
plt.show()