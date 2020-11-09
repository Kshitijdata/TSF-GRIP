#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# In[ ]:





# In[2]:


data = pd.read_csv('http://bit.ly/w-data')
print('Shape of the dataset is: ', data.shape)
data.head()


# In[3]:


# Checking for any missing values:

data.isnull().sum()


# In[4]:



X = data.iloc[:, :1].values
y = data.iloc[:, 1].values


# In[5]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[6]:


plt.plot(X_train, y_train, 'o', label = 'Training Set')
plt.plot(X_test, y_test, 'go', label = 'Testing Set')
plt.legend()
plt.title('Hours Studied vs. Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# In[7]:


def mean(list):
    return float(sum(list))/len(list)

def slope(X_train, y_train, X_mean, y_mean):
    return sum((X_train - X_mean) * (y_train - y_mean))/sum((X_train - X_mean) ** 2)

def intercept(X_mean, y_mean, m):
    return y_mean - m * X_mean


# In[8]:


X_mean = mean(X_train)
y_mean = mean(y_train)
m = slope(X_train[:, 0], y_train, X_mean, y_mean)
b = intercept(X_mean, y_mean, m)
print('Slope, m: ', m)
print('Intercept, b: ', b)


# In[9]:


plt.plot(X_train, X_train * m + b, 'c-')

# Plotting the data points from training set
plt.plot(X_train, y_train, 'o')
plt.title('Regression Graph')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# In[10]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print ("Slope: ", regressor.coef_[0])
print ("Intercept: ", regressor.intercept_)


# In[11]:


sns.regplot(X_train, y_train, ci = None, line_kws={'color':'c'})
plt.title('Regression Graph')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.show()


# In[12]:


print('The error between slope of our model and scikit-learn\'s model: ', abs(regressor.coef_[0] - m))
print('The error between intercept of our model and scikit-learn\'s model: ', abs(regressor.intercept_ - b))


# In[13]:


y_pred = regressor.predict(X_test)


# In[14]:


df = pd.DataFrame({'Hours': X_test[:,0], 'Actual Score': y_test, 'Predicted Score': y_pred})  
df


# In[15]:


print('Number of hours: 9.65')
print('Predicted score: ', regressor.predict([[9.65]])[0])


# In[16]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Value:', metrics.r2_score(y_test, y_pred))


# In[ ]:




