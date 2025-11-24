import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- FUNCTIONS ----------

@st.cache_data
def load_and_prepare_data():
    # Load the movies dataset
    movies = pd.read_csv("data/movies.csv")

    # Clean and create content column
    movies['genres'] = movies['genres'].fillna('')
    movies['content'] = movies['title'] + " " + movies['genres']

    # TF-IDF and similarity matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['content'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return movies, similarity_matrix


def recommend_movie(title, movies, similarity_matrix, top_n=5):
    """
    Recommend top_n movies similar to the given title.
    """
    title = title.lower()
    all_titles = movies['title'].str.lower()

    if title not in all_titles.values:
        return None  # movie not found

    idx = all_titles[all_titles == title].index[0]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n + 1]  # skip the same movie
    movie_indices = [i[0] for i in sim_scores]

    return movies[['title', 'genres']].iloc[movie_indices]


# ---------- STREAMLIT UI ----------

def main():
    st.title("🎬 Movie Recommendation System")
    st.write("Get movie recommendations based on a movie you like.")

    # Load data + model
    with st.spinner("Loading data and building model..."):
        movies, similarity_matrix = load_and_prepare_data()

    # Movie selection
    movie_list = movies['title'].tolist()
    selected_movie = st.selectbox("Select a movie:", movie_list)

    top_n = st.slider("Number of recommendations:", min_value=3, max_value=15, value=5)

    if st.button("Recommend"):
        recs = recommend_movie(selected_movie, movies, similarity_matrix, top_n=top_n)

        if recs is None:
            st.error("Movie not found in the database. Try another one.")
        else:
            st.subheader(f"Movies similar to: **{selected_movie}**")
            for _, row in recs.iterrows():
                st.markdown(f"- **{row['title']}**  \n  Genres: {row['genres']}")


if __name__ == "__main__":
    main()
