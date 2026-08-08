# Python Challenge Part 2 - Data Sciences Coursework

This project is part of a data sciences coursework, focusing on data manipulation, analysis, and visualization using Python. It includes a series of exercises dealing with music chart data.

## Guidelines

- Include only the answers to the questions posed in the exercises.
- Code commenting is optional but can be useful for clarification.
- Different approaches can lead to correct results.
- The code will be checked, not just the output.
- Follow-up errors in the code are not counted as errors.

## Data

The project uses two datasets:

1. **charts.csv**: Contains information on the top 200 songs per country and week.
2. **songs.csv**: Contains metadata for each song.

### Data Loading and Inspection

#### Exercise 1.1

- Load `songs.csv` and display the first five observations.
- Show the number of rows and columns in the dataset.

#### Exercise 1.2

- Load `charts.csv`, convert column names to lower case, and sort the DataFrame by `chart_id` and `position`.
- Display the first five rows of the sorted DataFrame.

### Data Processing and Analysis

#### Exercise 1.3

- Display columns `song_name`, `artist_name`, and `danceability` of the 10 songs with the lowest `danceability`.

#### Exercise 1.4

- Calculate and report the average `valence` of all songs.

#### Exercise 2.1

- Merge `charts.csv` and `songs.csv` without losing any observations from `charts.csv`.
- Report the shape of the resulting DataFrame.

#### Exercise 2.2

- Count and report observations with merge keys in both `charts` and `songs`, only in `charts`, and only in `songs`.

#### Exercise 2.3

- Create a new column `duration_min` for song duration in minutes.
- Remove the `duration_ms` column and report the remaining column names.

#### Exercise 2.4

- Process `chart_id` to create four new columns: `type`, `country`, `interval`, and `date`.
- Convert `date` to a datetime column and display the first five rows.

### Advanced Analysis

#### Exercise 3.1

- Analyze the German and United States charts for the week prior to Christmas 2021.
- Count the number of songs appearing in both countries' charts.

#### Exercise 3.2

- Calculate the fraction of explicit songs per genre and the number of non-missing observations.
- Sort and display the results from most to least explicit.

#### Exercise 3.3

- Find and display all artist names performing the song "Last Christmas".

#### Exercise 3.4

- Create two bar plots showing the total number of streams per genre for Mexico and the United States.
- Include titles and axis labels in the subplots and an overall title for the figure.

## Technologies Used

- Python
- Pandas for data manipulation
- Plotly for data visualization

---

[← Portfolio index](../../../README.md)
