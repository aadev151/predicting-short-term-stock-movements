# Main Findings

This document outlines the main fidings we've received in our project. This document serves as a fact repository for an LLM to use when generating the final report.

## Research Question

Can we predict stock movements using a combination of technical indicators and news sentiment analysis? All our models were trained and developed to predict the direction of stock price movement (up or down) in the short term (next day) based on historical price data and news sentiment.

## Test/Train Split

We decided to use a time-based split with the split at 2023-01-01. This gives an approximately 68-32 vs 80-20 split. However, this way the test dataset includes data from distinct market environents (post-Covid, 2024 election, 2025)

## Feature engineering

Refer to @notebooks/decision_tree.ipynb Feature Engineering section for the features that we decided to use. We used the same features in every model

## Baselines

In our analysis, we used 4 different baselines to compare our model's performance against.

1. Random guess -- essentially useless, with any model being able to outperform it
2. Always predict "up" -- since stocks has a positive long-run drift, more days are "Up" than. "Down"
3. Follow yesterday's direction (momentum)
4. Follow yesterday's market direction -- if S&P500 return today > 0, predict all stocks will be up tomorrow. This is a very important baseline to beat because idividual stock returns are largely driven by merket returns

### Baseline Results

========================================
  BASELINE REFERENCE  (test ≥ 2023-01-01)
========================================
Baseline                 Accuracy
----------------------------------------
Random Guess               0.5003
Always Up                  0.5223
Follow Yesterday           0.4961
Follow Market              0.4881
========================================

Interestingly enough, a random guess was able to beat even a robust (as we thought) follow the market strategy.

So even though we have established four different baselines, the Always Up one was the most accurate one. Now our models only need to beat the 52.23% accuracy to be considered useful. This is a very low bar, but it is what it is.

## Decision Tree

Even though accuracy on the validation set during cross-validation peaked at ~56%, when we selected our best-performing decision tree, we were only able to reach the accuracy of .5170 with the baseline of .5223.

Unfortunately, this did not succeed.

### Feature importances

Still under development
