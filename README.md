# Kubrick: Movie Recommender System

Kubrick is a thing I made that uses cosine similarity on the overviews of several movies and recommends based on that similarity score.
The output consists of the titles, posters and overviews of 5 recommendations.

Steps to run:

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Set up environment variables:**
   Create a `.env` file in the project root and add your OMDB API key:
    ```env
    OMDB_API_KEY=<your_omdb_api_key>
    ```

3. **Run the application:**
    ```bash
    python app.py
    ```

## Requirements:

If you want to make a model file with a higher/lower threshold, first download the [movies dataset](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies), move it to this project's directory, delete the pkl and npz lines, and then run app.py.
