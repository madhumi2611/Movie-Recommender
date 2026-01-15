#   Movie Recommender System

##  Overview
This project is a **Movie Recommendation System** built using **Python** and **Machine Learning**.  
It recommends movies to users based on the similarity of movie features such as genres, cast, crew, and overviews.  
The project also includes a simple **web interface** for easy interaction.

---

##  Features
-  Content-based recommendation system  
-  Pre-trained ML models (`.pkl` files) for instant suggestions  
-  Web interface using HTML  
-  Based on the TMDB Movie Dataset  
-  Secure handling of API keys using a `.env` file  

---


##  Model Details
The recommender system uses **content-based filtering**.  
It calculates **cosine similarity** between feature vectors extracted from:
- Movie genres  
- Cast and crew  
- Overview text  
- Keywords  

Pre-trained `.pkl` files store model data and clustering information for fast and accurate recommendations.

---

##  Getting Started

###  Clone the repository
```bash
git clone https://github.com/Crimson-ray/MOVIE_RECOMMENDER.git
cd movie_recommand
```

###  Create a virtual environment
```bash
python -m venv venv
```

###  Activate the environment
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```

###  Install dependencies
```bash
pip install -r requirements.txt
```

###  Add environment variables
Create a `.env` file in the root folder:
```
API_KEY=your_tmdb_api_key_here
```

###  Run the application
```bash
python main.py
```




