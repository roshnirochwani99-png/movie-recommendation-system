import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- STEP 1: Load dataset ----
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

print("Movies dataset shape:", movies.shape)

# ---- STEP 2: Clean and create content column ----
movies['genres'] = movies['genres'].fillna('')
movies['content'] = movies['title'] + " " + movies['genres']

print("Content column created:\n", movies.head())

# ---- STEP 3: TF-IDF Vectorization ----
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['content'])

print("TF-IDF matrix shape:", tfidf_matrix.shape)

# ---- STEP 4: Cosine Similarity ----
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("Similarity matrix shape:", similarity_matrix.shape)

def recommend_movie(title, top_n=5):
    """
    Recommend top_n movies similar to the given title.
    """
    # Convert input title to lowercase for matching
    title = title.lower()
    
    # Make a lowercase Series of all titles
    all_titles = movies['title'].str.lower()
    
    # Check if movie exists
    if title not in all_titles.values:
        print("Movie not found in database. Check spelling or try another movie.")
        return []
    
    # Get index of the movie
    idx = all_titles[all_titles == title].index[0]
    
    # Get similarity scores for this movie with all others
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort movies based on similarity score (highest first)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # First one will be the movie itself → skip it
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the indices of top similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the recommended movies (title + genres)
    return movies[['title', 'genres']].iloc[movie_indices]

if __name__ == "__main__":
    # Print some movie titles so you know what to type
    print("\nSample movie titles from the dataset:")
    print(movies['title'].head(10))

    # Ask user for a movie name
    user_movie = input("\nEnter a movie title exactly as shown above (e.g., Toy Story (1995)): ")

    # Get recommendations
    recs = recommend_movie(user_movie, top_n=5)

    # If list is not empty, print recommendations
    if len(recs) > 0:
        print(f"\nTop 5 recommendations similar to '{user_movie}':\n")
        for idx, row in recs.iterrows():
            print(f"- {row['title']}  |  Genres: {row['genres']}")
