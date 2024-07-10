#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the libraries


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[4]:


# data collection and data pre_processing


# In[5]:


df=pd.read_csv(r"C:\Users\pc\Downloads\movies.csv")
df.head(5)


# In[6]:


df.shape


# In[7]:


# selecting the relevant features for recommendation system


# In[50]:


selected_features=['genres','keywords','tagline','cast','director']
print(selected_features)


# In[51]:


# replacing the null values with null string


# In[52]:


for features in selected_features:
    df[features]=df[features].fillna('')


# In[53]:


# combining all 5 selected features


# In[54]:


combined_features=df['genres']+''+df['keywords']+''+df['tagline']+''+df['cast']+''+df['director']
combined_features


# In[55]:


# converting text data to feature vector


# In[56]:


vectorizer=TfidfVectorizer()


# In[57]:


feature_vector=vectorizer.fit_transform(combined_features)
print(feature_vector)


# In[58]:


# getting the similarity score using cosine similarity


# In[59]:


similarity=cosine_similarity(feature_vector)
print(similarity)


# In[60]:


print(similarity.shape)


# In[61]:


# getting the movie name from the user


# In[70]:


movie_name=input('Enter your favourite movie name: ')


# In[71]:


#creating a list with the the movie name given in the data set


# In[72]:


list_of_all_titles=df['title'].tolist()
print(list_of_all_titles)


# In[73]:


# finding the close match given by the user


# In[74]:


find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)
print(find_close_match)


# In[75]:


close_match=find_close_match[0]
print(close_match)


# In[76]:


#finding the index no. of the movie


# In[77]:


index_of_the_movie=df[df.title==close_match]['index'].values[0]
print(index_of_the_movie)


# In[78]:


# getting the list of similar score


# In[79]:


similarity_score=list(enumerate(similarity[index_of_the_movie]))
print(similarity_score)


# In[80]:


len(similarity_score)


# In[81]:


# sorting the movie based on there similrity score


# In[82]:


sorted_similar_movies=sorted(similarity_score,key =lambda x:x[1],reverse=True)
print(sorted_similar_movies)


# In[83]:


#print the movie based on the index


# In[84]:


print('movies suggested for you: \n')
i=1

for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=df[df.index==index]['title'].values[0]
    if (i<16):
        print(i,'.',title_from_index)
        i+=1


# In[85]:


# MOVIE RECOMMENDATION SYSTEM


# In[86]:


movie_name=input('Enter your favourite movie name: ')

list_of_all_titles=df['title'].tolist()

find_close_match=difflib.get_close_matches(movie_name,list_of_all_titles)

close_match=find_close_match[0]

index_of_the_movie=df[df.title==close_match]['index'].values[0]

similarity_score=list(enumerate(similarity[index_of_the_movie]))

sorted_similar_movies=sorted(similarity_score,key =lambda x:x[1],reverse=True)

print('movies suggested for you: \n')
i=1

for movie in sorted_similar_movies:
    index=movie[0]
    title_from_index=df[df.index==index]['title'].values[0]
    if (i<16):
        print(i,'.',title_from_index)
        i+=1


# In[ ]:




