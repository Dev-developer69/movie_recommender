import streamlit as st
import pickle
import pandas as pd
import requests
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


if os.path.exists('similarity.pkl'):
    similarity = pickle.load(open('similarity.pkl', 'rb'))
else:
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    vectors = tfidf.fit_transform(movies['tags']).toarray()
    similarity = cosine_similarity(vectors)


movies = pickle.load(open('movies.pkl', 'rb'))

API_KEY = "api_key"


st.markdown("""
     <style>
            @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;500&display=swap');

                
            .stApp{
                background: #1a1a2e !important ; 
            }
            #MainMenu{
                visibility: hidden !important
            }
            h1 {
                font-family: 'Bebas Neue', cursive !important;
                color: #E50914 !important ; 
                font-size: 50px !important;
                letter-spacing: 4px;
            }
            /* Dropdown font */
            .stSelectbox div, .stSelectbox label {
                font-family: 'Poppins', sans-serif !important;
                color: white !important;
                font-size: 16px;
            }
            
            /* Buttons */
            .stButton button {
                font-family: 'Poppins', sans-serif !important;
                font-weight: 500;
            }
            /* Recommend Button */
            .stButton button {
                font-family: 'Poppins', sans-serif !important;
                font-weight: 500;
                background-color: #FF6B9D;  
                color: white;
                border: none;
                border-radius: 25px;   
                padding: 6px 25px;       
                margin-top: 15px;         
                font-size: 16px;
                letter-spacing: 1px;
                transition: 0.3s ease-in-out;         
            }
            
            /* Hover effect */
            .stButton button:hover {
                background-color: #b20710;
                transform: scale(1.15);
                cursor: pointer;
            }
                </style>
    """, unsafe_allow_html=True)



def fetch_poster(movie_id):
    for attempt in range(4): 
        try:
            url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
            data = requests.get(url, timeout=10).json()
            poster_path = data.get('poster_path')
            if poster_path:
                return "https://image.tmdb.org/t/p/w500" + poster_path
            else:
                return "https://placehold.co/500x750?text=No+Poster"
        except:
            time.sleep(0.8)
    return "https://placehold.co/500x750?text=No+Poster"

def recommended_by_movie(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    if movie_index.empty:
        return None
    names = []
    posters = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]].id
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movie_id))
    return names, posters

def recommended_by_director(director):
    director = str(director).strip()
    director_movies = movies[movies['crew'].apply(lambda x: director in str(x))]
    
    if director_movies.empty:
        return None, None
    
    movie_index = director_movies.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    names, posters = [], []
    for i in movie_list:
        names.append(movies.iloc[i[0]].title)
        posters.append(fetch_poster(movies.iloc[i[0]].id))
    return names, posters

st.title("🎬 Movie Recommender System")

if 'search_type' not in st.session_state:
    st.session_state.search_type = None

col1,col2 = st.columns(2)
with col1:
    if st.button('Search by Movie',type='primary',key='btn1', width='stretch'):
        st.session_state.search_type = 'Movie'

with col2:
    if st.button('Search by Director',type='primary',key='btn2', width='stretch'):
        st.session_state.search_type = 'Director'

 
if st.session_state.search_type == "Movie":
    options = ["-- Select a Movie --"] + list(movies['title'].values)
    selected = st.selectbox("Movie choose karo:", options)

elif st.session_state.search_type == 'Director':
    options = ["-- Select a Director--"] + list(movies['crew'].values)
    selected = st.selectbox("Director ka naam btao:",options)

elif st.session_state.search_type ==None:
    st.info('Choose any option')


if st.button("Recommend"):
    if st.session_state.search_type == "Movie":
        names, posters = recommended_by_movie(selected)
        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.text(names[i])
                st.image(posters[i])
    else:
        names, posters = recommended_by_director(selected)
        if names is None:
            st.error("Director nahi mila dataset mein!")
        else:
            cols = st.columns(5)
            for i in range(5):
                with cols[i]:
                    st.text(names[i])
                    st.image(posters[i])
