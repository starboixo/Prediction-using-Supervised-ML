#!/usr/bin/env python
# coding: utf-8

# # AMOGH G. LONARE 

# # TASK 1 : Prediction using Supervised ML
#   Predict the percentage of an student based on the no. of study hours.

# # IMPORTING LIBRARIES

# In[34]:


import numpy as np


# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


link = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data = pd.read_csv(link)
print("Data is imported succesfully")
data.head()


# In[37]:


data.shape

data.describe()
# In[38]:


data.info()


# In[39]:


data.isnull().sum()


# In[40]:


data.plot(x="Hours",y="Scores",style="o")
plt.title("Study Hours v/s Percentage Scores")
plt.ylabel("Scores in Percentage")
plt.xlabel("No of Hours")
plt.show()


# In[41]:


sns.regplot(x=data["Hours"],y=data["Scores"],data=data)
plt.title("Study Hours vs Percentage Scores")
plt.xlabel("No of Hours")
plt.ylabel("Scores in Percentage")
plt.show()


# In[ ]:





# In[42]:


sns.regplot(x=data["Hours"],y=data["Scores"],data=data)
plt.title("Study Hours vs Percentage Scores")
plt.xlabel("No of Hours")
plt.ylabel("Scores in Percentage")
plt.show()


# It can be see that there is a +ve correlationship between the study hours and the percentage scores.

# In[43]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[45]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[46]:


print("Coefficient :", model.coef_)
print("Intercept :", model.intercept_)


# In[47]:


line = model.coef_*x - model.intercept_

plt.scatter(x,y)
plt.plot(x, line, color="red", label="Regression Line")
plt.legend()
plt.show()


# In[48]:


y_preds = model.predict(X_test)


# In[49]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_preds})
df


# In[50]:


print("Training Score :", model.score(X_train, y_train))
print("Testing Score :", model.score(X_train, y_train))


# In[51]:


df.plot(kind="bar", figsize=(9,5))
plt.grid(which="major",linewidth='0.5', color='blue')
plt.grid(which="minor",linewidth='0.5', color='red')


# # Task 2 : What if Student studies for 9.25 hours/day

# In[52]:


hours = 9.25
test = np.array([hours])
test = test.reshape(-1,1)
new_pred = model.predict(test)
print(f"No. of hours = {hours}")
print(f"Predicted Acore = {new_pred[0]}")


# # Final Step To Evaluating This Model

# In[56]:


from sklearn import metrics
print("Mean Absolute Error :", metrics.mean_absolute_error(y_test, y_preds))
print("Mean Squared Error :", metrics.mean_squared_error(y_test, y_preds))
print("Root Mean Squared Error :", np.sqrt(metrics.mean_squared_error(y_test, y_preds)))
print("R-2 :", metrics.r2_score(y_test,y_preds))


# # Thank You

# In[ ]:




