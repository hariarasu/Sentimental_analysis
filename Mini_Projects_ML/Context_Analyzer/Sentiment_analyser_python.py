#!/usr/bin/env python
# coding: utf-8

# In[49]:


import spacy
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from spacy.tokenizer import Tokenizer
import pandas as pd


# In[2]:


df2=pd.read_csv("C:/Users/vvhem/Jupyter_content/ML_Model_Dept/filtered.csv")


# In[3]:


df2.isna().sum()


# In[4]:


df2.isnull().sum()


# In[5]:


df2=df2.dropna()


# In[45]:


df2.result.value_counts()


# In[8]:


X=df2["refined_reviews"]
y=df2["result"]


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


# In[10]:


X_train.head(5)


# In[11]:

vectorizer=TfidfVectorizer(
    min_df=5,
    max_df=0.8,
    sublinear_tf=True,
    use_idf=True)



# In[12]:


train_vector=vectorizer.fit_transform(X_train)
test_vector=vectorizer.transform(X_test)


# In[13]:

classifier_model=LinearSVC()
classifier_model.fit(train_vector,y_train)




# In[14]:





# In[73]:


sample_review="bad"
review_vector=vectorizer.transform([sample_review])


# In[74]:


classifier_model.predict(review_vector)


# In[64]:


pred=classifier_model.predict(test_vector)
print(classification_report(y_test,pred,output_dict=True))


# In[75]:


import pickle
with open('Vectorizer','wb') as file:
    pickle.dump(vectorizer,file)


# In[76]:


with open('Classifier_model','wb') as file:
    pickle.dump(classifier_model,file)


# In[77]:


with open ('Vectorizer','rb') as file:
    model=pickle.load(file)


# In[78]:


the=model.transform([sample_review])


# In[79]:


with open('Classifier_model','rb') as file:
    model2=pickle.load(file)


# In[82]:


model2.predict(the)


# In[ ]:

def classification(text):
    return(model2.predict(model.transform([text])))


