# Data Science Coursework: Trade Data Analysis

This project focuses on analyzing global trade statistics using the United Nations Comtrade data extraction API. It involves a series of Python scripts that request, process, analyze, and visualize international trade data, with a particular focus on Ukraine's export dynamics and global dependencies on certain products.

## Project Structure and Content

1. **Ukraine's Exports by Partner Country (2021)**:
   - This script requests total export data of Ukraine to all available partner countries for the year 2021.
   - The data is processed to identify Ukraine's top 10 export partners.
   - A bar chart is used to visualize these exports, highlighting the trade value in billion USD.

2. **Ukraine's Exports by Product (2021)**:
   - The script analyzes Ukraine's most important export products in 2021.
   - It retrieves data for a specific product classification (cc='AG4') and processes it to find the top 10 export products.
   - Results are visualized in a bar chart, showing the global export value of these products in billions.

3. **Dependency of the World on Ukraine's Exports**:
   - This analysis focuses on the global dependency on Ukraine's exports of a specific product (e.g., cooking oil).
   - It calculates the fraction of global exports of the product contributed by each country.
   - A pie chart is used to represent these fractions for the top 10 exporter countries.

4. **World Dependency Plot Function**:
   - A Python function `world_dependency_plot` is developed to automate the analysis of global dependency on exports from different countries.
   - The function takes a product code and a year as inputs and returns a visualization similar to the previous exercise.

5. **Requesting Data for Many Years**:
   - A Python function `comtrade_many_years` is created to handle requests for an arbitrary number of years.
   - It accounts for API rate limits by splitting the requests and concatenating the results into a single dataset.
   - This function is tested for a range of years to analyze Germany's (r=276) total exports to the world.

## Technologies Used

- Python
- Pandas for data manipulation
- JSON for handling API responses
- Urllib for URL handling
- Plotly Express for data visualization
- Itertools and Time for handling multiple requests

## Data source

United Nations Comtrade database.

---

[← Portfolio index](../../../README.md)
