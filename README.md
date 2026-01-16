## Project 1: Forex Currency Prediction - 'app'

All files included, should dockerise easily.

This project was my introduction to creating web apps. I learnt a lot about how to organise data and how to process it within them to decrease loading times, as I made the mistake of performing a lot of the analysis in real time instead of processing beforehand. Some of the currencies have clearly incorrect predictions where the currency's value drops almost immediately to zero, so this needs to be addressed. Streamlit was very useful in displaying the graphical data easily, although some formatting options (such as giving upper and lower limits to the y-axis) proved very difficult.


## Project 2: Movie Recommendation System - 'movie-recommender'

'model.pkl' and 'similarities.pkl' were too large to upload to GitHub. The files should still be able to be generated using analysis.ipynb

I learnt from my previous project how to organise the analysis within my app, and this one runs much faster. Using Flask instead of Streamlit felt more intuitive and precise, despite the slight extra effort it took on my part. A lot of work went into cleaning and formatting the data appropriately across two datasets, and although the code could be cleaner, it works very well.


## Project 3: Web Search Agent - 'research-agent'

app.py should run once tavily and flask are installed and the TAVILY_API_KEY is given in app.py.

During this project, I used langchain to search documents and tavily to search the web to find relevent information based on a query given by the user. I was unable to successfully use crewai to construct specialised ai agents for writing and reviewing to give a summarised result due to version conflicts between packages. The flask web app in this repository only has the tavily web search part of my program, as the langchain similarity search takes a long time to run, especially with many documents. I have learned to more completely separate my python environments for each program I write.