# Widen: News Intelligence Mini-Project

Data Science Internship @ Thiranex — a 4-week end-to-end data science project on a real-world news dataset, built as one connected pipeline instead of four separate tasks.

**Goal:** explore whether data science techniques (cleaning, classification, EDA, and a simple dashboard) can help someone understand a news topic at a glance — a small-scale precursor to a bigger idea (Widen).

**Dataset:** [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (Kaggle, ~210K HuffPost articles, by rmisra). A random sample of 30,000 cleaned rows is used to keep the project fast to run and easy to learn from.

## Progress

- [x] Week 1: Data Cleaning & Visualization
- [x] Week 2: Topic Classification Model
- [ ] Week 3: Exploratory Data Analysis
- [ ] Week 4: Streamlit Dashboard

## Week 1 — Data Cleaning & Visualization

First, I examined the dataset's structure — column names, row count, and data types — to understand what I was working with before deciding on further steps.

The main focus of the week was data cleaning: I identified missing values and made deliberate decisions on each — some (like blank author names) were filled with a placeholder since they weren't needed for later modeling, while others (like blank descriptions) were dropped since that column would be used as model input. I also removed exact duplicate rows.

For outlier detection, I used the IQR method on text length to flag unusually short or long articles, but rather than dropping everything the statistics flagged, I manually inspected the flagged rows first. This showed that very short texts were genuinely low-content or junk data worth removing, while unusually long texts were legitimate, detailed articles worth keeping — reinforcing that statistical flags are a starting point, not an automatic decision rule.

I also reviewed the dataset's 42 category labels and merged clearly overlapping ones (e.g. "ARTS" and "ARTS & CULTURE") down to 33 cleaner categories.

Finally, I visualized the cleaned data — a bar chart of category distribution, a histogram of text length, and a line chart of articles per year — before saving the cleaned dataset for the next stage.

**Files:** `notebooks/week1_cleaning_eda.ipynb`, `data/cleaned/news_cleaned.csv`

## Week 2 — Topic Classification Model

The goal this week was to predict an article's `category` directly from its `text`, using classical ML models rather than deep learning — a good baseline before anything more complex.

**Vectorization:** Since models can't work with raw strings, I converted the `text` column into numeric features using TF-IDF (Term Frequency – Inverse Document Frequency), which scores each word by how important it is to a specific document relative to how common it is across the whole dataset — so frequent-everywhere words like "the" end up near-zero, while distinctive words carry more weight. This produced a 24,000 × 36,925 sparse matrix from the training split.

I split the data into 80% train / 20% test *before* vectorizing, and fit the vectorizer only on the training text — fitting on the full dataset (including test data) would leak information about the test set into the model's vocabulary and IDF scores, making evaluation dishonest.

**Models compared:** I trained and evaluated three classifiers on the same TF-IDF features, to compare rather than assume which would perform best:

| Model | Accuracy |
|---|---|
| Logistic Regression | ~56.9% |
| Random Forest | ~49.1% |
| Multinomial Naive Bayes | ~35.0% |

Logistic Regression performed best, likely because it combines all 36,925 (mostly sparse) features into one weighted decision per category, rather than relying on one feature at a time (Random Forest) or assuming word independence (Naive Bayes) — both of which struggle more with sparse, high-dimensional text data.

**Evaluation:** Beyond accuracy, I built a confusion matrix (33×33, one row/column per category) for each model to see *where* errors were happening, not just how many. Logistic Regression's errors were fairly spread out across categories. Naive Bayes showed a clear bias toward over-predicting a few high-frequency categories (e.g. `POLITICS`, `WELLNESS`) regardless of the article's true category — consistent with its independence assumption breaking down on related word pairs (e.g. "stock market", "prime minister") that carry more meaning together than apart.

**Files:** `ThiranexDataScience.ipynb`
