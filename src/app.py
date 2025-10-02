import streamlit as st
import pandas as pd
import pickle
import os
from fuzzywuzzy import process

# Ruta base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar datos preprocesados
clean_data = pd.read_pickle(os.path.join(BASE_DIR, "clean_data.pkl"))
with open(os.path.join(BASE_DIR, "cosine_sim.pkl"), "rb") as f:
    cosine_sim = pickle.load(f)

# Función de recomendación con fuzzy matching
def recommend_movies(query, top_n=5):
    # buscar título más parecido
    matches = process.extract(query, clean_data['Series_Title'], limit=1)
    if not matches:
        return [], None
    
    best_match, score = matches[0]
    if score < 50:  # muy bajo, probablemente no relevante
        return [], None

    idx = clean_data[clean_data['Series_Title'] == best_match].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return clean_data.iloc[movie_indices].to_dict(orient='records'), best_match


# ---------------- Streamlit UI ----------------

st.set_page_config(page_title="🎬 IMDb Movie Recommender", layout="wide")

st.title("🎬 IMDb Movie Recommender")
st.write("Type the name of a movie and get recommendations based on plot, genre, director and actors.")

# Input del usuario
query_title = st.text_input("Enter a movie title:", "")
top_n = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)

if st.button("Get Recommendations"):
    if query_title:
        recommendations, best_match = recommend_movies(query_title, top_n)
        
        if recommendations:
            st.success(f"Showing recommendations for **{best_match}**")
            for movie in recommendations:
                st.subheader(movie['Series_Title'])
                st.write(f"**Genre:** {movie['Genre']}")
                st.write(f"**Director:** {movie['Director']}")
                st.write(f"**IMDb Rating:** {movie['IMDB_Rating']}")
                st.markdown("---")
        else:
            st.error("No similar movies found. Try another title.")
    else:
        st.warning("Please enter a movie title.")