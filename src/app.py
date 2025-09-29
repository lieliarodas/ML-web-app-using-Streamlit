from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import os

# Ruta base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cargar datos preprocesados
clean_data = pd.read_pickle(os.path.join(BASE_DIR, "clean_data.pkl"))
with open(os.path.join(BASE_DIR, "cosine_sim.pkl"), "rb") as f:
    cosine_sim = pickle.load(f)

# Función de recomendación
def recommend_movies(title, top_n=5):
    if title not in clean_data['Series_Title'].values:
        return []
    
    idx = clean_data[clean_data['Series_Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return clean_data.iloc[movie_indices].to_dict(orient='records')

# Crear app Flask
app = Flask(__name__)

# Página principal con formulario
@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    query_title = ""
    top_n = 5

    if request.method == "POST":
        query_title = request.form.get("title")
        top_n = int(request.form.get("top_n", 5))
        recommendations = recommend_movies(query_title, top_n)

    return render_template("index.html",
                           recommendations=recommendations,
                           query_title=query_title,
                           top_n=top_n)

# Endpoint API
@app.route("/recommend", methods=["GET"])
def recommend_endpoint():
    movie_title = request.args.get("title")
    if not movie_title:
        return jsonify({"error": "Please provide a movie title"}), 400
    top_n = int(request.args.get("top_n", 5))
    recommendations = recommend_movies(movie_title, top_n)
    return jsonify(recommendations)

# Desarrollo local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
