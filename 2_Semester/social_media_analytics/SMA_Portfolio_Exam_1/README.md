# Data Science Coursework Project: Text Analysis and Word Embedments

## Project Overview
This project, as part of a data science coursework, involves analyzing text data using various Python libraries. The project is split into several exercises, each focusing on different aspects of text analysis and natural language processing, from web scraping to training word embeddings and building a content-based recommender system.

## Key Features and Methods

### Exercise 1: Web Scraping
- Developed a function to scrape articles from 'tagesschau.de' for a given date.
- The function extracts article elements like title, date, description, text, and tags, and stores them in a dataframe.

### Exercise 2: Word Cloud Generation
- Implemented a function to create word clouds for specified articles.
- The function allows for different types of word clouds ('noun' or 'entity') and custom dimensions.

### Exercise 3: Training Word Embeddings
- Used Word2Vec and FastText from the Gensim library to train word embedding models on the scraped text data.
- Preprocessed the text data and created a corpus for training the models.

### Exercise 4: Analysis and Application of Word Embeddings
- Analyzed the most similar words to a given word using both Word2Vec and FastText models.
- Compared and described differences in the results from both models.
- Developed a content-based recommender system to recommend similar articles based on word embeddings.

## Technologies Used
- Python
- Libraries: NumPy, Pandas, Requests, BeautifulSoup, Spacy, Matplotlib, WordCloud, NLTK, Gensim

---

[← Portfolio index](../../../README.md)
