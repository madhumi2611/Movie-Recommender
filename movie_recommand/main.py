from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# additional imports
from typing import Dict
from sklearn.preprocessing import normalize

import requests
from typing import List, Optional
import os

app = FastAPI(title="Movie Recommendation System")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TMDB API Configuration
TMDB_API_KEY = "6f21e4fa36d1e82c2c5b2ff1ccfb8219"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

# Global variables for model
df = None
tfidf_matrix = None

def load_model():
    """Load the pickled model and dataframe"""
    global df, tfidf_matrix, kmeans_model, agg_model, available_models
    svd_model = None
    kmeans_model = None
    agg_model = None
    available_models = [ "linear_kernel"]  
    try:
        with open("movie_recommender.pkl", "rb") as f:
            df, tfidf_matrix = pickle.load(f)
        print(f"Model loaded successfully! Dataset size: {len(df)}")
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")

    try:
        with open("movie_recommender_clusters_kmeans.pkl", "rb") as f:
            kmeans_model = pickle.load(f)
        print("KMeans loaded: movie_recommender_clusters_kmeans.pkl")
        available_models.append("kmeans")
    except Exception:
        kmeans_model = None

    try:
        with open("movie_recommender_clusters_hier_agg.pkl", "rb") as f:
            d = pickle.load(f)
            agg_model = d.get('agg_model')
            hier_idx = d.get('hier_idx', None)
        print("Agglomerative model loaded: movie_recommender_clusters_hier_agg.pkl")
        available_models.append("agg")
    except Exception:
        agg_model = None

    print("Available models:", available_models)

def recommend_by_model(movie_index: int, n: int = 12, model: str = "tfidf"):
    """
    Return recommended movie indices depending on `model`.
    model options: "tfidf", "kmeans", "agg", "popularity", "hybrid"
    """
    if model == "tfidf" or kmeans_model is None:
        # default TF-IDF cosine-based (existing)
        sims = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()
        top_idx = sims.argsort()[-n-1:-1][::-1]
        return top_idx

    if model == "popularity": 
        # return top-n by popularity in linear kernel excluding the seed movie
        seed_id = df.index[movie_index]
        pop_sorted = df.sort_values('pop_score', ascending=False).index.tolist()
        # remove seed if in list
        pop_sorted = [i for i in pop_sorted if i != movie_index]
        return np.array(pop_sorted[:n], dtype=int)

    if model == "kmeans" and kmeans_model is not None:
        # find cluster of the movie and rank others in cluster by TF-IDF similarity (or popularity)
        # if you saved kmeans on full df you can use kmeans_model.labels_
        try:
            labels = kmeans_model.labels_
            movie_cluster = labels[movie_index]
            same_cluster_idx = np.where(labels == movie_cluster)[0]
            # compute similarity within cluster and pick top-n excluding seed
            sims = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix[same_cluster_idx]).flatten()
            order = sims.argsort()[::-1]
            chosen = same_cluster_idx[order]
            chosen = [i for i in chosen if i != movie_index]
            return np.array(chosen[:n], dtype=int)
        except Exception:
            # fallback to tfidf
            sims = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()
            top_idx = sims.argsort()[-n-1:-1][::-1]
            return top_idx

    if model == "agg" and agg_model is not None:
        # we assume agg labels saved to df column 'agg_<k>'
        agg_cols = [c for c in df.columns if c.startswith("agg_")]
        if not agg_cols:
            # fallback
            return recommend_by_model(movie_index, n, model="tfidf")
        agg_col = agg_cols[-1]  # take last one
        movie_cluster = int(df.iloc[movie_index][agg_col])
        same_cluster_idx = df[df[agg_col] == movie_cluster].index.to_numpy()
        # rank by tfidf similarity inside cluster
        sims = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix[same_cluster_idx]).flatten()
        order = sims.argsort()[::-1]
        chosen = same_cluster_idx[order]
        chosen = [i for i in chosen if i != movie_index]
        return np.array(chosen[:n], dtype=int)

    if model == "hybrid":
        # prefer kmeans; if cluster size small, fall back to popularity
        try:
            idxs = recommend_by_model(movie_index, n, model="kmeans")
            if len(idxs) >= n:
                return idxs[:n]
            # pad with popularity
            pop = recommend_by_model(movie_index, n*2, model="popularity")
            pad = [i for i in pop if i not in idxs and i != movie_index]
            return np.array(list(idxs) + pad[:(n-len(idxs))], dtype=int)
        except Exception:
            return recommend_by_model(movie_index, n, model="tfidf")

    # default fallback
    sims = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    top_idx = sims.argsort()[-n-1:-1][::-1]
    return top_idx


def get_tmdb_details(title: str, year: Optional[str] = None):
    """Fetch movie details from TMDB API"""
    try:
        # Search for movie
        search_url = f"{TMDB_BASE_URL}/search/movie"
        params = {
            "api_key": TMDB_API_KEY,
            "query": title,
            "year": year if year else None
        }
        response = requests.get(search_url, params=params, timeout=5)
        results = response.json().get('results', [])
        
        if not results:
            return None
        
        movie = results[0]
        movie_id = movie['id']
        
        # Get detailed movie info
        detail_url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        detail_params = {"api_key": TMDB_API_KEY, "append_to_response": "credits"}
        detail_response = requests.get(detail_url, params=detail_params, timeout=5)
        details = detail_response.json()
        
        return {
            "title": details.get('title'),
            "poster": f"{TMDB_IMAGE_BASE}{details.get('poster_path')}" if details.get('poster_path') else None,
            "backdrop": f"https://image.tmdb.org/t/p/w1280{details.get('backdrop_path')}" if details.get('backdrop_path') else None,
            "overview": details.get('overview'),
            "release_date": details.get('release_date'),
            "rating": details.get('vote_average'),
            "vote_count": details.get('vote_count'),
            "runtime": details.get('runtime'),
            "genres": [g['name'] for g in details.get('genres', [])],
            "cast": [c['name'] for c in details.get('credits', {}).get('cast', [])[:5]],
            "director": next((c['name'] for c in details.get('credits', {}).get('crew', []) if c['job'] == 'Director'), None)
        }
    except Exception as e:
        print(f"Error fetching TMDB details: {e}")
        return None

def recommend(movie_index, top_n=10):
    """Generate movie recommendations"""
    sim = linear_kernel(tfidf_matrix[movie_index], tfidf_matrix).flatten()
    top_idx = sim.argsort()[-top_n-1:-1][::-1]
    return top_idx

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return FileResponse("index.html")

@app.get("/api/models")
async def list_models():
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"available_models": available_models}


@app.get("/api/search", response_model=List[dict])
async def search_movies(q: str):
    """Search for movies with autocomplete suggestions"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not q or len(q) < 2:
        return []
    
    # Case-insensitive search
    mask = df['title'].str.lower().str.contains(q.lower(), na=False)
    results = df[mask].head(10)
    
    return [
        {
            "id": int(row['id']) if 'id' in row else idx,
            "title": row['title'],
            "year": row.get('release_date', '')[:4] if pd.notna(row.get('release_date')) else '',
            "rating": float(row.get('vote_average', 0)) if pd.notna(row.get('vote_average')) else 0
        }
        for idx, row in results.iterrows()
    ]

@app.get("/api/movie/{title}")
async def get_movie_details(title: str):
    """Get detailed information about a specific movie"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Find movie in dataset
    movie_data = df[df['title'].str.lower() == title.lower()]
    
    if movie_data.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_row = movie_data.iloc[0]
    movie_index = movie_data.index[0]
    
    # Get year from release_date
    year = None
    if pd.notna(movie_row.get('release_date')):
        year = str(movie_row['release_date'])[:4]
    
    # Fetch TMDB details
    tmdb_data = get_tmdb_details(movie_row['title'], year)
    
    # Combine dataset info with TMDB data
    result = {
        "title": movie_row['title'],
        "overview": movie_row.get('overview', ''),
        "release_date": movie_row.get('release_date', ''),
        "rating": float(movie_row.get('vote_average', 0)) if pd.notna(movie_row.get('vote_average')) else 0,
        "vote_count": int(movie_row.get('vote_count', 0)) if pd.notna(movie_row.get('vote_count')) else 0,
        "runtime": int(movie_row.get('runtime', 0)) if pd.notna(movie_row.get('runtime')) else None,
        "poster": None,
        "backdrop": None,
        "genres": [],
        "cast": [],
        "director": None
    }
    
    # Override with TMDB data if available
    if tmdb_data:
        result.update({k: v for k, v in tmdb_data.items() if v is not None})
    
    return result

@app.get("/api/recommend/{title}")
async def get_recommendations(title: str, n: int = 12, model: str = "tfidf"):
    """Get movie recommendations based on a title and chosen model"""
    if df is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if model not in available_models:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not available. Choose from {available_models}")

    # Find movie index
    movie_data = df[df['title'].str.lower() == title.lower()]
    if movie_data.empty:
        raise HTTPException(status_code=404, detail="Movie not found")

    movie_index = movie_data.index[0]

    # Use selected model to get indices
    rec_indices = recommend_by_model(movie_index, n=n, model=model)
    recommendations = []

    for idx in rec_indices:
        row = df.iloc[idx]
        year = str(row.get('release_date', ''))[:4] if pd.notna(row.get('release_date')) else None

        # Try to get TMDB poster (optional / slower). Consider skipping for speed or caching results.
        tmdb_data = get_tmdb_details(row['title'], year) if TMDB_API_KEY else None

        rec = {
            "title": row['title'],
            "year": year,
            "rating": float(row.get('vote_average', 0)) if pd.notna(row.get('vote_average')) else 0,
            "poster": tmdb_data.get('poster') if tmdb_data else None,
            "overview": row.get('overview', '')[:150] + '...' if len(str(row.get('overview', ''))) > 150 else row.get('overview', '')
        }
        recommendations.append(rec)

    return {"model": model, "recommendations": recommendations}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)