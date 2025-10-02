import streamlit as st
import pandas as pd
import pickle
from fuzzywuzzy import process
import os

# Basic Streamlit configuration
st.set_page_config(page_title="IMDb Movie Recommender", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
body {
    background-color: #1c1c1c;
    color: #ffffff;
}
.stButton>button {
    background-color: #F5C518;
    color: #000000;
    font-weight: bold;
}
.stTextInput>div>input {
    background-color: #333333;
    color: #ffffff;
}
.card {
    background-color: #2c2c2c;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
}
.card-title {
    font-weight: bold;
    font-size: 16px;
    color: #F5C518;
}
.card-text {
    color: #ffffff;
    font-size: 14px;
}
hr {
    border-top: 2px solid #F5C518;
}
</style>
""", unsafe_allow_html=True)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load preprocessed data
clean_data = pd.read_pickle(os.path.join(BASE_DIR, "clean_data.pkl"))
with open(os.path.join(BASE_DIR, "cosine_sim.pkl"), "rb") as f:
    cosine_sim = pickle.load(f)

# Recommendation function with fuzzy matching
def recommend_movies_fuzzy(title, top_n=5):
    # Find the closest match to the input title
    titles_list = clean_data['Series_Title'].tolist()
    best_match, score = process.extractOne(title, titles_list)
    
    if score < 50:
        return [], best_match  # Return empty if no good match found
    
    # Get index of the matched movie
    idx = clean_data[clean_data['Series_Title'] == best_match].index[0]
    
    # Calculate similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the movie itself
    
    movie_indices = [i[0] for i in sim_scores]
    
    # Return recommended movies as a list of dictionaries
    return clean_data.iloc[movie_indices].to_dict(orient='records'), best_match

# Title and input
st.title("🎬 IMDb Movie Recommender")
st.write("Type the name of a movie and get recommendations based on plot, genre, director and actors.")
st.markdown("<hr>", unsafe_allow_html=True)

# User inputs
input_title = st.text_input("Enter movie title")
top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, step=1)

# Button to get recommendations
if st.button("Get Recommendations"):
    if input_title:
        recommendations, matched_title = recommend_movies_fuzzy(input_title, top_n)
        if recommendations:
            st.subheader(f"Top {top_n} recommendations for '{matched_title}'")
            for movie in recommendations:
                st.markdown(f"""
                <div class="card">
                    <div class="card-title">{movie['Series_Title']}</div>
                    <div class="card-text">
                        <strong>Genre:</strong> {movie['Genre']}<br>
                        <strong>Director:</strong> {movie['Director']}<br>
                        <strong>IMDB Rating:</strong> {movie['IMDB_Rating']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No movie found with a similar title. Please try again.")