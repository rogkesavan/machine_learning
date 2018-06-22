
# coding: utf-8

# In[189]:


#import libaries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib import pyplot
from datetime import datetime
from matplotlib import pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)


# load data 
names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Marketcap']
Bitcoin = pd.read_csv('cryptocurrency_data/bitcoin_dataset1.csv', names=names)
ETH = pd.read_csv('cryptocurrency_data/ethereum_price.csv', names=names)





trace1 = go.Scatter(
                    x=Bitcoin['Date'], y=Bitcoin['Marketcap'], # Data
                    mode='lines', name='Bitcoin' # Additional options
                   )
trace2 = go.Scatter(x=ETH['Date'], y=ETH['Marketcap'], mode='lines', name='ETH' )


layout = go.Layout(title='Simple Plot from csv data',
                   plot_bgcolor='rgb(230, 230,230)')

fig = go.Figure(data=[trace1, trace2], layout=layout)

# Plot data in the notebook
py.iplot(fig, filename='simple-plot-from-csv')


# In[192]:


from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn.ensemble import RandomForestClassifier




#we need only Open column for our prediction 
#df = Bitcoin[['Open']]
df = ETH[['Open']]
#print(df)





# predicting ____ days into future
forecast_out = int(5)
#  label column with data shifted 100 units up
df['Prediction'] = df[['Open']].shift(-forecast_out) 





# Defining Features & Labels
X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)
# set X_forecast equal to last ___
X_forecast = X[-forecast_out:] 
X = X[:-forecast_out] # remove last ____ from X
y = np.array(df['Prediction'])
y = y[:-forecast_out]




#Linear Regression
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)
# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)




forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)





History = pd.read_csv('cryptocurrency_data/history.csv')

trace = go.Scatter(y=forecast_prediction)
#trace1 = go.Scatter(y=History)
data = [trace]
layout = dict(title = 'Train and Test Loss during training',
              xaxis = dict(title = 'Days'), yaxis = dict(title = 'Price'))
fig = dict(data=data, layout=layout)
py.iplot(fig, filename='simple-plot-from-csv')

