from utils import db_connect
engine = db_connect()

from flask import Flask, request, jsonify
import pandas as pd
import pickle

clean_data = pd.read_pickle("clean_data.pkl")
with open("cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

def recommend_movies(title, top_n=5):
    
    idx = clean_data[clean_data['Series_Title'] == title].index[0]
    
   
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1] 
    
    
    movie_indices = [i[0] for i in sim_scores]
    return clean_data.iloc[movie_indices].to_dict(orient='records')


app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend_endpoint():
    movie_title = request.args.get("title")
    top_n = int(request.args.get("top_n", 5))
    recommendations = recommend_movies(movie_title, top_n)
    return jsonify(recommendations)


if __name__ == "__main__":
    app.run(debug=True)