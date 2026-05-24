import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer



movies = pd.read_csv('tmdb_5000_movies.csv')
credits= pd.read_csv('tmdb_5000_credits.csv')
data= movies.merge(credits, on='title')
data=data[['id','title','genres','cast','crew','overview','popularity','revenue','keywords']].copy()
data_cleaned= data.copy()
data_cleaned=data_cleaned.dropna()

import ast
def extractor(obj):
    list=[]
    for i in ast.literal_eval(obj):
        list.append(i['name'])
    return list

data_cleaned['genres']= data_cleaned['genres'].apply(extractor)

data_cleaned['keywords']= data_cleaned['keywords'].apply(extractor)

def extractor3(obj):
    list=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            list.append(i['name'])
            counter+=1
        else:
            break
    return list

data_cleaned['cast']= data_cleaned['cast'].apply(extractor3)

def fetch_director(obj):
    list=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            list.append(i['name'])
    return list

data_cleaned['crew'] = data_cleaned['crew'].apply(fetch_director)
data_cleaned['overview'] = data_cleaned['overview'].apply(lambda x:x.split())

data_cleaned['genres'] = data_cleaned['genres'].apply(lambda x: [i.replace(" ","") for i in x ])
data_cleaned['cast'] = data_cleaned['cast'].apply(lambda x: [i.replace(" ","") for i in x])
data_cleaned['crew'] = data_cleaned['crew'].apply(lambda x: [i.replace(" ","") for i in x])
data_cleaned['keywords'] = data_cleaned['keywords'].apply(lambda x: [i.replace(" ","") for i in x])


data_cleaned['tags']= data_cleaned['crew']+ data_cleaned['cast'] + data_cleaned['genres'] + data_cleaned['keywords']+ data_cleaned['overview']
data_cleaned.columns

final_data = data_cleaned[['id','title','crew','tags']]

ps=PorterStemmer()

def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)
        
final_data['tags']= final_data['tags'].apply(lambda x: " ".join(x))

final_data['tags']= final_data['tags'].apply(lambda x: x.lower())

final_data['tags']=final_data['tags'].apply(stem)

final_data['tags'][0]

cv= CountVectorizer(max_features= 5000, stop_words= 'english')
vectors=cv.fit_transform(final_data['tags']).toarray()

similarity= cosine_similarity(vectors)

def recommended(movie):
    movie_index= final_data[final_data['title']==movie].index[0]
    distances= similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse= True, key= lambda x:x[1])[1:6]

    for i in movie_list:
        print(final_data.iloc[i[0]].title)

recommended('Avatar')

print(final_data.columns)
pickle.dump(final_data, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

