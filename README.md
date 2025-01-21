## Smart-Product-Review-Analyzer
# Multi-Modal Data Analysis and Predictive Insights

This project involves analyzing and deriving insights from a dataset combining text data (product reviews) with numerical metadata. The workflow includes feature extraction, sentiment analysis, predictive modeling, and data visualization to provide actionable insights about product ratings.

---

## Table of Contents
1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Steps and Methodology](#steps-and-methodology)
4. [Results and Visualizations](#results-and-visualizations)
5. [Outputs](#outputs)
6. [BanaoAiTask2Report.pdf](#PDF-Report)
7. [Acknowledgments](#acknowledgments)

---

## Objective
The main objectives of this project are:
- To preprocess and analyze textual data from product reviews.
- To extract features such as sentiment scores, key phrases, and topics.
- To integrate textual features with numerical metadata for multi-modal analysis.
- To develop a predictive model to forecast product ratings and visualize insights.

---

## Dataset
The project utilizes the [Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews), which includes:
- **Text reviews**: User feedback in text format.
- **Metadata**: Product ratings, helpfulness votes, and other numerical data.

---

## Steps and Methodology

### 1. Data Preprocessing
- Cleaned text by removing URLs, special characters, and excessive whitespaces.
- Converted text to lowercase and applied tokenization, stopword removal, and lemmatization.

### 2. Sentiment Analysis
- Calculated sentiment polarity scores using the `TextBlob` library.
- Extracted key phrases using TF-IDF vectorization.

### 3. Topic Modeling
- Performed Latent Dirichlet Allocation (LDA) to identify key topics in reviews.

### 4. Feature Engineering
- Combined textual features (sentiment scores, key phrases) with numerical metadata.
- Engineered additional features from the dataset.

### 5. Predictive Modeling
- Developed a Gradient Boosting Regressor to predict product ratings.
- Split data into training and testing sets and evaluated model performance using metrics like Mean Squared Error (MSE) and R-squared.

### 6. Visualization
- Created visualizations to illustrate:
  - Sentiment scores vs. product ratings.
  - Feature importance in the predictive model.
  - Distribution of actual vs. predicted ratings.

---

## Results and Visualizations

### Key Visualizations
- **Residual Plot**: Highlights the difference between actual and predicted ratings.
- **Feature Importance**: Shows the significance of each feature in the model.
- **Distribution Plot**: Compares the distribution of actual vs. predicted ratings.

---

## Outputs
The following outputs are saved in the `output/` directory:
1. `step1_loaded_data.csv`: Raw dataset.
2. `step2_preprocessed_text.csv`: Preprocessed dataset with processed text.
3. `step3_sentiment_features.csv`: Dataset with sentiment scores and topics.
4. `step4_feature_engineering.csv`: Final feature set for modeling.
5. `step5_predictions.csv`: Predictions from the Gradient Boosting Regressor.
6. `residual_plot.png`: Residual plot.
7. `feature_importance.png`: Feature importance visualization.
8. `distribution_plot.png`: Actual vs. predicted ratings distribution.
9. `model_performance.txt`: Metrics (MSE, R-squared) for the predictive model.

---
## BanaoAiTask2Report
BanaoAiTask2Report.pdf
---

## Acknowledgments
- Dataset sourced from [Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews).
- Libraries used include `pandas`, `numpy`, `nltk`, `textblob`, `scikit-learn`, `matplotlib`, and `seaborn`.
- Special thanks to open-source contributors for making tools and datasets available for analysis.

---
