# Visualization Challenge

## Project Overview

Four visualizations of the Happy Planet Index (HPI), an alternative to GDP that scores countries on well-being and sustainability rather than output. The dataset holds four variables per country, which makes it a good test of whether a chart can carry all of them at once and stay readable. Data from [Happy Planet Index](https://happyplanetindex.org).

### Key Objectives:
1. Reveal relationships between multiple variables in the HPI dataset.
2. Analyze the evolution of Zimbabwe's HPI over time.
3. Visualize HPI data on a world map with a specific projection.
4. Show the distribution of HPI per continent.

## Installation

Before running the scripts, ensure you have R installed along with the following packages:
- `tidyverse`
- `sf`
- `rnaturalearth`
- `GGally`
- `plotly`
- `viridis`

## Data Source

The data is sourced from the Happy Planet Index website and includes variables like life expectancy, experienced wellbeing, ecological footprint, and GDP per capita.

## Structure

- `RF_challenge.Rmd`: The analysis. All four tasks — setup, the multi-variable
  relationships for 2019, Zimbabwe's HPI over time, the world map, and the
  distribution per continent — are in this single R Markdown file.
- `RF_challenge.html`: The knitted output, with the Plotly charts still
  interactive. [Read it in the browser](https://regis-funke.github.io/data-science-portfolio/1_Semester/data_viz/DV_Portfolio_Exam_1/RF_challenge.html)
  without installing anything.
- `happy-planet-index.csv`: The main dataset used for the analyses.

---

[← Portfolio index](../../../README.md)
