# Widen: News Intelligence Mini-Project

Data Science Internship @ Thiranex — a 4-week end-to-end data science project on a real-world news dataset, built as one connected pipeline instead of four separate tasks.

**Goal:** explore whether data science techniques (cleaning, classification, EDA, and a simple dashboard) can help someone understand a news topic at a glance — a small-scale precursor to a bigger idea (Widen).

**Dataset:** [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (Kaggle, ~210K HuffPost articles, by rmisra). A random sample of 30,000 cleaned rows is used to keep the project fast to run and easy to learn from.

## Progress

- [x] Week 1: Data Cleaning & Visualization
- [x] Week 2: Topic Classification Model
- [x] Week 3: Exploratory Data Analysis
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

## Week 3 — Exploratory Data Analysis

This week's goal was to dig deeper into the dataset's statistical structure and use those findings to explain *why* the Week 2 models behaved the way they did, rather than treating EDA as a separate exercise from the classification work.

**Statistical summaries:** I computed mean, median, and skewness of `text_length` grouped by category. Skewness turned out to be the most revealing metric — most categories showed moderate right-skew (0.5–1.5), but `POLITICS` (5.04) and `THE WORLDPOST` (4.00) stood out with extreme skew, while `FIFTY` was essentially symmetric (-0.02) despite having the highest median length overall (303 words). This showed two distinct patterns hiding behind similar-looking summary stats: `POLITICS` has a tight cluster of typical articles plus a long tail of outliers, while `FIFTY` is just consistently longer-form content with no real outliers.

**Boxplots:** I visualized text length distributions across categories to confirm this. `POLITICS` and `THE WORLDPOST` showed small, tight boxes but a dense cloud of outlier points reaching up to ~1,500 words, while `FIFTY` showed a taller box (wider typical range) with almost no outliers — proving that box size (typical spread) and outlier density (frequency of extreme values) are separate signals that can tell different stories about the same category.

**Class imbalance:** Using `value_counts()` on `category`, I confirmed what the Week 2 confusion matrices had hinted at: `POLITICS` (32,436 articles) and `WELLNESS` (23,205) dominate the dataset, while categories like `LATINO VOICES` (1,022) and `GOOD NEWS` (1,039) have roughly 30x fewer samples. This directly explains Naive Bayes' bias toward over-predicting the high-frequency classes — with imbalanced priors and no strong mechanism to resist the shortcut, it defaults to the statistically "safe" high-frequency answer more often than Logistic Regression does.

**Key influencing factor:** Category consistently influences article length and distribution shape — some categories (`POLITICS`, `THE WORLDPOST`) mix short news briefs with rare long-form pieces, others (`FIFTY`) are uniformly longer, and others (`WEIRD NEWS`, ~114 words median) run consistently short. This structural variation, combined with class imbalance, offers a data-level explanation for the classification patterns observed in Week 2 — not just a model-level one.

**Files:** `ThiranexDataScience.ipynb`
