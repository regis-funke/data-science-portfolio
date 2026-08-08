# OCR Engines Performance Comparison

## Overview
This project is focused on comparing the performance of different Optical Character Recognition (OCR) engines to determine which performs best under various image quality conditions. The OCR engines compared include Pytesseract, EasyOCR, and docTR.

## Methodology
- **Data Source**: The base collection consists of 85 images from openfoodfacts.org, primarily of ingredient lists.
- **Ground Truths**: Ground truths are obtained using Abbyy Fine Reader and manually corrected. These are stored in text files corresponding to each image.
- **OCR Engines**: The project tests three OCR engines (Pytesseract, EasyOCR, and docTR) for their ability to accurately extract text from images.
- **Accuracy Measurement**: Accuracy is measured using Levenshtein Distance and Levenshtein Ratio. The distance indicates deviation from the ground truth, while the ratio provides a normalized measure of difference.
- **Image Processing**: Images are processed (e.g., cropped, modified) to isolate ingredient lists for analysis.

## Execution
- Images are loaded, and OCR is run on each, returning the output as a string.
- A separate function loads the corresponding ground truth text file for comparison.
- Levenshtein distance and ratio are calculated for each picture, using OCR output against its ground truth.
- Results are compiled into dataframes for analysis and comparison.

## Results Visualization
- Distribution of Levenshtein Ratios per engine is visualized to show the comparative accuracy of each OCR engine.
- Results are further prepared for visualization, showing top, middle, and bottom examples of OCR performance from each engine.

## Key Findings
- EasyOCR shows the highest share of results with ratios greater than 0.9, indicating superior performance on this set of images.
- Pytesseract performs well in terms of speed, but its accuracy is lower compared to EasyOCR.
- docTR shows the weakest performance, struggling with images of decreasing quality.

## Technologies Used
- Python
- Libraries: Pandas, Matplotlib, Pillow, Pytesseract, EasyOCR, python-doctr, python-Levenshtein, Plotly, OpenCV

---

[← Portfolio index](../../README.md)
