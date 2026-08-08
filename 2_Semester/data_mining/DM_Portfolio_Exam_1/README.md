# Frequent Itemset Mining and Association Rule Learning Project

## Overview
This project conducts a comprehensive analysis of frequent itemsets and association rule mining using the APRIORI and FP-Growth algorithms. The focus is on assessing the performance of these algorithms under different conditions and datasets.

## Description
The project involves multiple exercises:
1. **Candidate Generation**: Exploring APRIORI candidate generation for creating frequent itemsets.
2. **Frequent Itemset Mining**: Applying APRIORI to different datasets to identify frequent itemsets and analyze their support values.
3. **Performance Measurement**: Timing the execution of APRIORI for different support thresholds and data structures.
4. **Comparison of Algorithms**: Conducting a comparative study between APRIORI and FP-Growth on various datasets with different support thresholds.
5. **Visualization**: Plotting the results to visually compare the performances of APRIORI and FP-Growth.

## Methodology
- **Data Preparation**: Two datasets, a grocery dataset and a larger dataset 'T10I4D100K', are prepared for analysis.
- **APRIORI and FP-Growth Implementation**: Both algorithms are applied to the datasets to extract frequent itemsets with varying support thresholds.
- **Performance Analysis**: The runtime and memory consumption of the algorithms are measured and compared.
- **Visualization**: Results are visualized using Plotly Express to provide insights into the performance trends.

## Key Findings
- APRIORI's performance is significantly affected by lower support thresholds, whereas FP-Growth shows better efficiency.
- For larger datasets and higher thresholds, APRIORI may outperform FP-Growth.
- Sparse and non-sparse data representations have varying impacts on algorithm performance.

## Technologies Used
- Python
- Pandas and Numpy for data manipulation
- MLxtend for implementing APRIORI and FP-Growth
- Plotly Express for data visualization
- Timeit for performance measurement

---

[← Portfolio index](../../../README.md)
