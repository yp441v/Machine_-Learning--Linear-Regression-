#!/usr/bin/env python
# coding: utf-8

# In[61]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[40]:


df  = pd.read_csv('new_insurance_data (1).csv')
df


# ### EDA

# In[41]:


df.shape


# In[42]:


df.info()


# In[43]:


# checked null values 
df.isnull().sum()


# In[44]:


# Dropped null Values 
df = df.dropna()


# In[45]:


df.isnull().sum()


# In[46]:


# check duplicates 
df.duplicated().sum()

#   df.drop_duplicates()  : to remove duplicates 


# ### Visualisation 

# In[47]:


# for reference : all the columns in one output 
col = df.columns
col


# In[48]:


# Histograph 
plt.figure(figsize=(5, 3))
plt.hist(df['age'])
plt.xlabel('age')
plt.ylabel('count')
plt.show()


# In[49]:


# statistical comopnent  analysis 
df.describe().T


# In[50]:


# Detecting Outliers and IQR


# In[51]:


col


# In[52]:


# Code that gives box plots for all the columns whre datatype isn't object 

for itr in col:
    if (df[itr].dtypes == 'int64' or df[itr].dtypes == 'float64'):
        plt.boxplot(df[itr])
        plt.xlabel(itr)
        plt.ylabel('Outliers')
        plt.show()


# In[37]:


# removal of outliers using IQR method 



# In[ ]:


# creat Q3, Q1,IQR
# Use the condition of outliers : All the datapoints Below lower fence and Above Upper Fence  are known as ouotliers
# So use the opposite/ Reversal ofthe above condition in order to remove 

# outliers = df['age']< lf & df['age']> uf
     # reverse the above condition 
# no outliers /only pure data = df['age']>lf & df['age']<uf


# In[ ]:


# make a not ao all the columns with outliers 

# bmi, past_consultations, Hospital_expenditure, NUmber_of_past_hospitalizations, Anual_Salary, charges 


# In[35]:


# For bmi column

Q1 = df.bmi.quantile(0.25)
Q3 = df.bmi.quantile(0.75)

IQR = Q3-Q1
IQR

# Filtering out the outliers by applying the reversal of the Technique 

df = df[(df.bmi >= Q1 -1.5*(IQR)) & (df.bmi <= Q3+1.5*(IQR))]  



# In[55]:


col


# In[63]:


# Using For Loop Removal of Outliers 
for i in col:
  if (df[i].dtypes == 'int64' or df[i].dtypes == 'float64'):
    Q1 = df[i].quantile(0.25)
    Q3 =  df[i].quantile(0.75)
    IQR = Q3-Q1
    df = df[(df[i]>= Q1 - (1.5 * IQR)) & (df[i] <= Q3 + (1.5 * IQR))]


# In[64]:


# verify 

for itr in col:
    if (df[itr].dtypes == 'int64' or df[itr].dtypes == 'float64'):
        plt.boxplot(df[itr])
        plt.xlabel(itr)
        plt.ylabel('Outliers')
        plt.show()


# In[69]:


corr = df.corr(numeric_only = True)
corr


# In[92]:


# Plotting the heatmap
plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt=".2f")
plt.show()


# ## Model Building 

# In[131]:


# 1. Locate the Trget Columns and store it as 'y' (Dependent Variable/ Target Variable)

y = df.iloc[:,-1]    # last column is always the Trget column 


# In[132]:


# 2. To store all the feature other than the target in the independent variavle'x

x = df.loc[:, ['children','Claim_Amount','past_consultations','Hospital_expenditure','age' ]]


# In[133]:


df


# In[134]:


# 3. Splitting the Variable I.e x and y

#  variables x and y are split into 2 different cataories making it 4 variable in the end 
     #   x-train , x_test, y_train , y_test
  

      # training data is for actual values 
      # Testing data is for predicted values  



# In[135]:


# 4.  import the Algorith of Linear Regression 

from sklearn.linear_model import LinearRegression 


# In[136]:


#  created the Model and stored in the variable 'lr'
lr = LinearRegression()
lr


# In[137]:


# .fit methos is used to provide date to the machine (training date )
lr.fit(x_train,y_train)


# In[138]:


# using the test variables : perfomr prediction of results : predicted outcomes 


# In[139]:


# create a variavle  and make use .predict with x_test
# this gives us the output values that are predicted  outputs that recieved from trainig data 


# In[140]:


pred = lr.predict(x_test)
pred


# In[141]:


# Compare with y_test for better understanding of
# accuracy of the model built for the comparison of expected outputs and predicted outputs

# In short calculate r2s


# In[142]:


from sklearn.metrics import *


# In[143]:


score = r2_score(y_test, pred)
score


# In[144]:


# conclusion 

#With an R-squared score of 69%, the model explains 69% of the variance in healthcare charges or insurance premiums based on the selected features. 
#While this indicates a good level of predictive capability, there's still room for improvement. 
#Further refinement and validation are recommended for better generalization and decision-making.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




