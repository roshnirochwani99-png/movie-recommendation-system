import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- BUILD MODEL ON STARTUP ----------

movies = pd.read_csv("data/movies.csv")
movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['title'] + " " + movies['genres']

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------- FASTAPI APP ----------

app = FastAPI()

# Allow Next.js (localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/movies")
def get_movies():
    """
    Return list of all movie titles.
    """
    return {"movies": movies['title'].tolist()}


@app.get("/recommend")
def recommend_movie(title: str, n: int = 5):
    """
    Recommend n movies similar to the given title.
    """
    title_lower = title.lower()
    all_titles = movies['title'].str.lower()

    if title_lower not in all_titles.values:
        raise HTTPException(status_code=404, detail="Movie not found")

    idx = all_titles[all_titles == title_lower].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    movie_indices = [i[0] for i in sim_scores]

    recs = movies[['title', 'genres']].iloc[movie_indices]

    return {
        "base_movie": title,
        "recommendations": [
            {"title": row['title'], "genres": row['genres']}
            for _, row in recs.iterrows()
        ]
    }
