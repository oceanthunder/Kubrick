import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import save_npz, load_npz
import pickle
import gradio as gr

#for loading the omdb api key
from dotenv import load_dotenv
import os
load_dotenv()

# For fetching posters (OMDB API)
def getPoster(t):
    apiKey = os.getenv('OMDB_API_KEY')
    if not apiKey:
        raise ValueError("API key is missing. Please set OMDB_API_KEY.")
    url = f'http://www.omdbapi.com/?t={t}&apikey={apiKey}'

    try:
        response = requests.get(url)
        data = response.json()
        if data.get('Response') == 'True':
            return data.get('Poster', 'No poster available')
        else:
            return 'No poster available'
    except:
        return 'No poster available'

# Remove stop words and transform Overview into ovMatrix
def vecData(df):
    tfidf = TfidfVectorizer(stop_words='english')
    ovMatrix = tfidf.fit_transform(df['overview'])
    return ovMatrix

# Calculates how similar two words are in the ovMatrix
def computeSimilarities(ovMatrix):
    ovSim = cosine_similarity(ovMatrix).astype(float)
    return ovSim

# Recommends the top 5 movies based on the similarity scores from the ovMatrix.
def recommendMovies(movie, df, ovSim):
    movie = movie.strip().lower() 
    if movie not in df['title'].str.lower().values:  # Compare against lowercase titles in dataset
        return f"Movie '{movie}' not found.", []
    
    idx = df[df['title'].str.lower() == movie].index[0]
    simScores = ovSim[idx]  
    similarIdx = simScores.argsort()[-6:-1][::-1]  # Get the indices of the most similar movies
    
    similarMovies = []
    for i in similarIdx:
        m = df.iloc[i]
        poster = getPoster(m['title'])
        similarMovies.append({
            'Title': m['title'],
            'Poster': poster,
            'Overview': m['overview']
        })
    
    return "Recommended Movies:", similarMovies

# Save model components
def saveModel(df, ovMatrix, ovSim, filename):
    save_npz('ovMatrix.npz', ovMatrix)  # Save sparse matrix
    with open(filename, 'wb') as f:
        pickle.dump({'df': df, 'ovSim': ovSim}, f)

# Load model components
def loadModel(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    df = data['df']
    ovSim = data['ovSim']
    ovMatrix = load_npz('ovMatrix.npz')  # Load sparse matrix
    return df, ovMatrix, ovSim

# Load or create model
def initializeModel():
    try:
        df, ovMatrix, ovSim = loadModel('model.pkl')
    except FileNotFoundError:
        df = pd.read_csv('TMDB_movie_dataset_v11.csv')
        df['overview'] = df['overview'].fillna('')
        df = df[df['vote_count'] > 500] # this is the threshold, change it to reduce/increase the amount of movies
        ovMatrix = vecData(df)
        ovSim = computeSimilarities(ovMatrix)
        saveModel(df, ovMatrix, ovSim, 'model.pkl')
    return df, ovMatrix, ovSim

# Gradio interface
def recommend(movie):
    df, _, ovSim = initializeModel()
    message, recommendations = recommendMovies(movie, df, ovSim)
    
    results = []
    for rec in recommendations:
        title = rec['Title']
        poster_url = rec['Poster']
        overview = rec['Overview']
        
        if poster_url != 'No poster available':
            poster_link = f"<a href='{poster_url}' target='_blank'><img src='{poster_url}' style='width:120px; border-radius:10px; box-shadow:0 4px 8px rgba(0, 0, 0, 0.2);'></a>"
        else:
            poster_link = 'No poster available'

        results.append(f"<div style='margin-bottom:20px;'><strong style='font-size:18px;'>{title}</strong><br>{poster_link}<br><p style='font-size:14px; line-height:1.6; color:#444;'>{overview}</p></div>")
    
    return message, "<div style='display:flex; flex-wrap:wrap; gap:20px;'>" + "".join(results) + "</div>"

# Initialize model before running
initializeModel()

# Create Gradio app
with gr.Blocks() as app:
    gr.Markdown("""<h1 style='text-align:center; color:#4CAF50;'>🎬 Kubrick</h1>
    <p style='text-align:center; color:#555;'>Enter the name of a movie you liked and discover similar movies you'll love! (or hate, idk model isn't perfect :) </p>""")
    with gr.Row():
        with gr.Column(scale=1):
            movie_input = gr.Textbox(label="Enter Movie Title", placeholder="e.g., Full Metal Jacket", lines=1, max_lines=1)
            submit_button = gr.Button("🔍 Recommend")
        with gr.Column(scale=2):
            output_message = gr.Textbox(label="Message", interactive=False, max_lines=1)
            output_recommendations = gr.HTML(label="Recommendations")
    
    submit_button.click(recommend, inputs=[movie_input], outputs=[output_message, output_recommendations])

# Launch app
app.launch(share=True)
