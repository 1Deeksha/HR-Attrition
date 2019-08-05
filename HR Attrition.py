#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


hr = pd.read_csv("hr.csv")


# In[4]:


hr.head()


# In[5]:


hr.tail()


# In[6]:


feats = ['department','salary']
hr_final = pd.get_dummies(hr,columns=feats,drop_first=True)
print(hr_final)


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x= hr_final.drop(['left'], axis=1).values
y= hr_final['left'].values


# In[9]:


print(x)


# In[10]:


print(y)


# In[11]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)


# In[12]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train= sc.fit_transform(x_train)
x_test= sc.transform(x_test)


# In[13]:


print(x_train)


# In[14]:


print(x_test)


# In[15]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[16]:


classifier= Sequential()


# In[17]:


classifier.add(Dense(9, kernel_initializer = "uniform", activation= "relu", input_dim=18))


# In[18]:


classifier.add(Dense(1, kernel_initializer = "uniform", activation= "sigmoid"))


# In[19]:


classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics= ["accuracy"])


# In[20]:


classifier.fit(x_train, y_train, batch_size=10, epochs = 10)


# In[21]:


y_pred = classifier.predict(x_test)


# In[22]:


y_pred = (y_pred > 0.5)


# In[23]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[24]:


new_pred = classifier.predict(sc.transform(np.array([[0.26,0.7,3.,
                                                     238.,6.,0.,
                                                     0.,0.,0.,0.,0.,
                                                     0.,0.,0.,1.,0.,
                                                     0.,1.]])))


# In[25]:


new_pred = (new_pred > 0.5)
new_pred


# In[26]:


new_pred = (new_pred > 0.6)
new_pred


# In[27]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


# In[28]:


def make_classifier():
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu", input_dim = 18))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer= "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier


# In[29]:


classifier = KerasClassifier(build_fn = make_classifier, batch_size=10, nb_epoch=1)


# In[31]:


accuracies = cross_val_score(estimator = classifier, X = x_train, y=y_train, cv=10)


# In[32]:


mean = accuracies.mean()
mean


# In[33]:


variance = accuracies.var()
variance


# In[35]:


from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dropout(rate = 0.1))
classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])


# In[37]:


from sklearn.model_selection import GridSearchCV
def make_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(9, kernel_initializer = "uniform", activation = "relu"))
    classifier.add(Dense(1, kernel_initializer = "uniform", activation = "sigmoid"))
    classifier.compile(optimizer = optimizer, loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier


# In[38]:


classifier = KerasClassifier(build_fn = make_classifier)


# In[40]:


params = {
    'batch_size' : [20,35],
    'epochs': [2,3],
    'optimizer': ['adam', 'rmsprop']
}


# In[41]:


grid_search = GridSearchCV(estimator=classifier,
                          param_grid=params,
                          scoring= "accuracy",
                          cv=2)


# In[43]:


grid_search = grid_search.fit(x_train, y_train)


# In[47]:


best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_


# In[48]:


best_param


# In[49]:


best_accuracy


# Applying RFE (Recursive feature elimination) and then building logistics regression model to predict which variable might impact the employees attrition rate.
# 

# In[50]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[51]:


pd.crosstab(hr.department, hr.left).plot(kind='bar')
plt.title('Employee attrition per Department')
plt.xlabel('Department')
plt.ylabel('Employee Attrition')
plt.savefig('Employee attrition per Deparment')


# In[30]:


pd.crosstab(hr.salary, hr.left).plot(kind='bar')
plt.title('Employee attrition as per Salary')
plt.xlabel('Salary')
plt.ylabel('Employee Attrition')
plt.savefig('Employee attrition as per Salary')


# In[52]:



pd.crosstab(hr.salary, hr.promotion_last_5years).plot(kind='bar')
plt.title('promotion_last_5years for Salary categories')
plt.xlabel('Salary')
plt.ylabel('promotion_last_5years')
plt.savefig('promotion_last_5years for Salary categories')


# In[53]:


hr_final.shape


# In[57]:


#applying RFE to filter best suitable variables for our model

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[58]:


model = LinearRegression()
rfe = RFE(model, 5)
rfe = rfe.fit(x, y)
print(rfe.support_)
print(rfe.ranking_)


# We have 5 variables for our model, which are marked as true in the support_array and "1" in the ranking_array. They are: 'satisfaction_level', 'Work_accident', 'promotion_last_5years', 'salary_low', 'salary_medium'.
# 
# Now, we will build Logistic regression model by usig these stated 5 variables.

# In[59]:


vars=['satisfaction_level', 'Work_accident', 'promotion_last_5years', 'salary_low', 'salary_medium']
X = hr_final[vars]
Y = hr_final['left']


# In[60]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
model.fit(X_train, Y_train)


# In[61]:


from sklearn.metrics import accuracy_score
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(Y_test, model.predict(X_test))))


# In[62]:


from sklearn.metrics import classification_report
import seaborn as sns
print(classification_report(Y_test, model.predict(X_test)))


# In[63]:


pred_Y = model.predict(X_test)
model_cm = metrics.confusion_matrix(pred_Y, Y_test, [1,0])
model_cm


# In[64]:


sns.heatmap(model_cm, annot=True, fmt= '.2f', xticklabels = ["Left", "Stayed"], yticklabels =["Left", "Stayed"])
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Logistic Regression')
plt.savefig('HR_Logistic_regression_CM')


# In[ ]:




