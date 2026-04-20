# Examining Sentiment Bias Across Cuisine Types in Yelp Reviews 

CS4120 | Sheryl Cheng, Arpitha Coorg, Melina Yang

## Overview
This project investigates whether systematic sentiment bias exists across cuisine types 
in Yelp reviews using three NLP sentiment analysis approaches: VADER, DistilBERT, and 
RoBERTa. We analyze 100,000 Yelp reviews spanning 10 cuisine types and control for 
restaurant price tier to isolate the effect of cuisine type on review sentiment.

## Research Question
Do reviews of non-American ethnic cuisine restaurants receive systematically lower 
sentiment scores compared to American cuisine restaurants, even when controlling for 
price tier?

## Dataset
Download the cleaned dataset here: https://drive.google.com/file/d/1eDUSLp9KEs4gtdPY3ttVy19iRQW5ho8R/view?usp=sharing

## Setup
1. Clone the repo:
   git clone https://github.com/melinayg/CS4120-Project.git
   
   cd CS4120-PROJECT

3. Install dependencies:
   pip install -r requirements.txt

## How to Run
Run notebooks in this order:

1. 1_data_cleaning.ipynb        - cleans raw Yelp data, assigns cuisine labels, filters reviews
2. 2_exploratory_analysis.ipynb - VADER baseline, word frequency, stereotype word analysis
3. 3_distilbert_model.ipynb     - DistilBERT sentiment scoring + aggregation + statistics
4. 4_distilbert_cv.ipynb        - 5-fold cross-validation to optimize DistilBERT threshold
5. 5_roberta_model.ipynb        - RoBERTa sentiment scoring + aggregation + statistics
6. 6_roberta_cv.ipynb           - 5-fold cross-validation to optimize RoBERTa threshold
7. 7_model_comparison.ipynb     - cross-model comparison, figures, final evaluation

## Dependencies
See requirements.txt. Main packages:
- pandas, numpy, json
- torch, transformers
- scipy, scikit-learn
- tqdm, vaderSentiment
- matplotlib, seaborn, plotly
- os, urllib, zipfile, ssl

## Key Findings
- American cuisine scored lowest in aggregate sentiment across all three models
- This disparity is concentrated at the budget ($) tier and largely disappears at 
  higher price points, suggesting establishment quality is the primary confound
- Persistent linguistic differences exist across cuisines (e.g. "authentic" used 
  more for ethnic cuisines, "cheap" associated with Chinese restaurants)
- Strong inter-model agreement (r > 0.80) validates findings across approaches
