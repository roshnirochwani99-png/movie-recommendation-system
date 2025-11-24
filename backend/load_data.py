import pandas as pd

# Load movies and ratings CSV files
movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

# Show dataset shapes
print("Movies dataset shape:", movies.shape)
print("Ratings dataset shape:", ratings.shape)

# Check for missing values in genres column
movies['genres'] = movies['genres'].fillna('')

# Create a new combined column named "content"
movies['content'] = movies['title'] + " " + movies['genres']

# Display first few rows to confirm
print("\nSample Movie Data with Content Column:\n")
print(movies.head())
