# python -m flask run

import pandas as pd
import numpy as np
import pickle
from flask import Flask, flash, redirect, render_template, request, url_for

def get_similar_films(film_title, no_of_recommendations):
    # Get top 5 similar films (excluding itself)
    try:
        film_title_lower = film_title.lower() # Convert to lowercase for matching...
        film_no = df[df['title_lower'] == film_title_lower].index[0]
        similar_films = sorted(list(enumerate(similarity[film_no])), reverse=True, key=lambda x: x[1])[1:(no_of_recommendations + 1)]
        header = "Top {} similar films to '{}':".format(no_of_recommendations, df['title'][film_no]) # ...But use original case for display
    except IndexError:
        print(f"Film titled '{film_title}' not found in the database.")
        return film_title+" not found in database.", []
    return header, similar_films

# Load the movies DataFrame
f = open('model.pkl', 'rb')
df = pickle.load(f)
f.close()
# Load the similarity DataFrame
f = open('similarity.pkl', 'rb')
similarity = pickle.load(f)
f.close()

print('Running movie recommender...')
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    header = ""
    recommendations = []
    if request.method == 'POST':
        # Get form data
        title = request.form.get('movie_title')
        no_of_recommendations = int(request.form.get('no_of_recommendations'))

        # Input validation
        if not title or no_of_recommendations <= 0 or not no_of_recommendations:
            return render_template('index.html', error="Please provide a valid movie title and number of recommendations.")
        
        # Generate recommendations
        print(f"Received request for recommendations based on '{title}' ({no_of_recommendations} recommendations).")
        header, similar_films = get_similar_films(title, no_of_recommendations)
        recommendations = [df['title'][i[0]] for i in similar_films]

        # Redirect to results page
    return render_template('index.html', header=header, recommendations=recommendations)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)