# RF DV Portfolio Exam 2

## Overview

This project, conducted during the Summer Semester 2022, delves into the analysis of inflation and its impact on consumer spending categories. Utilizing data from DESTATIS, the German Federal Office of Statistics, the project explores the historical development of inflation and its effects on different consumer spending areas since 1991.

## Data Source

The data for this project is sourced from DESTATIS ([German Federal Office of Statistics](https://www-genesis.destatis.de/genesis/online)). Data files used include:
- `61111-0004.xml`
- `61111-0002.xlsx`

These files were processed using a Python notebook (`RF_DV_data_prep.ipynb`) to generate six CSV tables for visualization purposes.

## Dependencies

The project utilizes R and its libraries for data processing and visualization:
- `tidyverse`: For data manipulation and visualization.
- `plotly`: For interactive plotting.

Ensure these libraries are installed in R before running the scripts.

## Project Structure

- `RF_DV_data_prep.ipynb`: Python notebook for data preparation.
- `RF_DV_Portfolio_Exam_2.Rmd`: Main R Markdown file containing all visualizations and analyses.

### Data Files

After preparation in Python, the following CSV files are used in R for visualizations:
- `61111-0004.csv`
- `61111-0004_pct.csv`
- `61111-0004_std.csv`
- `61111-0002.csv`
- `61111-0002_pct.csv`
- `61111-0002_std.csv`

## Key Insights

- Historical trends in consumer price index since 1992.
- Month-by-month inflation rate analysis.
- Continuous and cyclical price development in various consumer spending categories.
- Comparative analysis of specific categories like transportation, education, leisure, etc.

## Conclusion

The project concludes with insights on how rising inflation rates might affect different consumer groups, especially those with lower incomes. It also provides practical tips on managing expenses during times of inflation.

---

[← Portfolio index](../../../README.md)
