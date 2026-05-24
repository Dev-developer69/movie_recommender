import requests
import pandas as pd
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer

API_KEY = "ae649c6d4d194c3c135e8a5be84a628b"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json"
}

# ========== Step 1: Safe API call ==========
def safe_get(url, retries=3):
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            return response.json()
        except:
            print(f"Retry {i+1}...")
            time.sleep(2)
    return None

# ========== Step 2: TMDB se latest movies fetch karo ==========
def fetch_latest_movies(pages=10):
    movies = []
    for page in range(1, pages+1):
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&page={page}"
        data = safe_get(url)
        if not data:
            continue

        for movie in data['results']:
            detail_url = f"https://api.themoviedb.org/3/movie/{movie['id']}?api_key={API_KEY}&append_to_response=keywords,credits"
            detail = safe_get(detail_url)
            if not detail:
                continue

            movies.append({
                'id': detail.get('id'),
                'title': detail.get('title'),
                'overview': detail.get('overview', ''),
                'genres': [g['name'] for g in detail.get('genres', [])],
                'keywords': [k['name'] for k in detail.get('keywords', {}).get('keywords', [])],
                'cast': [c['name'] for c in detail.get('credits', {}).get('cast', [])[:5]],
                'crew': [c['name'] for c in detail.get('credits', {}).get('crew', []) if c['job'] == 'Director'],
            })
            time.sleep(0.3)

        print(f"Page {page}/{pages} done ✅")

    return pd.DataFrame(movies)

print("Fetching latest movies from TMDB...")
new_df = fetch_latest_movies(pages=10)
print(f"Fetched: {len(new_df)} new movies")

# ========== Step 3: Purana dataset load karo ==========
old_df = pd.read_pickle('movies.pkl')
print(f"Old dataset: {len(old_df)} movies")

# ========== Step 4: Tags banao naye movies ke liye ==========
def make_tags(row):
    genres   = row['genres']   if isinstance(row['genres'], list)   else []
    keywords = row['keywords'] if isinstance(row['keywords'], list) else []
    cast     = row['cast']     if isinstance(row['cast'], list)     else []
    crew     = row['crew']     if isinstance(row['crew'], list)     else []
    overview = row['overview'].split() if isinstance(row['overview'], str) else []
    return " ".join(genres + keywords + cast + crew + overview).lower()

new_df['tags'] = new_df.apply(make_tags, axis=1)

# ========== Step 5: Merge karo ==========
common_cols = ['id', 'title', 'tags']
old_df  = old_df[common_cols]
new_df  = new_df[common_cols]

merged_df = pd.concat([old_df, new_df]).drop_duplicates(subset='title').reset_index(drop=True)
print(f"Total after merge: {len(merged_df)} movies ✅")

# ========== Step 6: Stemming ==========
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(word) for word in str(text).split()])

print("Stemming tags...")
merged_df['tags'] = merged_df['tags'].apply(stem)

# ========== Step 7: Similarity matrix ==========
print("Calculating similarity matrix (thoda time lagega)...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(merged_df['tags']).toarray()
similarity = cosine_similarity(vectors)
print("Similarity matrix ready ✅")

# ========== Step 8: Pickle save karo ==========
pickle.dump(merged_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
print(f"Pickle files updated ✅")
print(f"Final dataset: {len(merged_df)} movies")