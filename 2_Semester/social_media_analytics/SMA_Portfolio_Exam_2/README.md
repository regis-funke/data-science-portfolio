# Data Science Project: Analysis of German Parliament Press Releases

## Project Overview
This project analyzes the communications behavior of the German parliament, focusing on extracting prevalent topics and analyzing their distribution over time. The analysis period ranges from January 1, 2021, to January 16, 2023.

## Key Components

### Part 1: Scraping
- Scraped press releases from the German parliament's website.
- Data extracted includes dates, titles, and text contents of press releases.

### Part 2: Filtering
- Employed Latent Dirichlet Allocation (LDA) to filter out press releases about public events.
- Text cleaning and tokenization were done to prepare the data for LDA.

### Part 3: Analysis
- Created a dictionary and corpus from the tokenized texts.
- Determined the optimal number of topics for LDA.
- Conducted LDA to identify distinct topics within the press releases.
- Visualized topic distributions over time and intertopic distance maps.

## Technologies Used
- Python
- Libraries: NumPy, Pandas, BeautifulSoup, Gensim, NLTK, Matplotlib, Plotly Express, pyLDAvis

## Note
The project aims to provide insights into political communications by analyzing topics in press releases. For a more detailed understanding, a domain specialist's review is recommended.

---

[← Portfolio index](../../../README.md)
