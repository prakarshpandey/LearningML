import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL') # data frame

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecastCol = 'Adj. Close'
df.fillna(-99999, inplace = True)

forecastOut = int(math.ceil(0.001 * len(df)))

df['label'] = df[forecastCol].shift(-forecastOut)
df.dropna(inplace=True)

# features: factors that determine the output of {label}
X = np.array(df.drop(['label'], 1))

# labels: stuff we want to predict
y = np.array(df['label'])

# need to include training data
X = preprocessing.scale(X)

print(len(X), len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
