from tavily import TavilyClient
from flask import Flask, flash, redirect, render_template, request, url_for

query = "Your research query here"
results_limit = 3

# Tavily online search
tavily_client = TavilyClient(api_key="TAVILY_API_KEY") # Replace with your actual Tavily API key
print('Running research agent...')
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])

def index():
    query = ""
    results = []
    if request.method == 'POST':
        query = request.form['query']
        results_limit = int(request.form['max_results'])
        if results_limit < 1 or results_limit > 10:
            flash('Please enter a number between 1 and 10 for maximum results.')
            return render_template('index.html', error="Please provide a valid number of results.")

        response = tavily_client.search(query=query, max_results=results_limit)
        # Display Tavily search results
        print("\nTavily Search Results:\n")
        for result in response['results']:
            print(f"Title: {result['title']}")
            results.append([result['title'], result['url'], result['score']])
        print("Search completed successfully.")
    return render_template('index.html', query=query, results=results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)