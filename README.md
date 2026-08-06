# Widen: News Intelligence Mini-Project

Data Science Internship @ Thiranex — a 4-week end-to-end data science project on a real-world news dataset, built as one connected pipeline instead of four separate tasks.

**Goal:** explore whether data science techniques (cleaning, classification, EDA, and a simple dashboard) can help someone understand a news topic at a glance — a small-scale precursor to a bigger idea (Widen).

**Dataset:** [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) (Kaggle, ~210K HuffPost articles, by rmisra). A random sample of 30,000 cleaned rows is used to keep the project fast to run and easy to learn from.

## Progress

- [x] Week 1: Data Cleaning & Visualization
- [ ] Week 2: Topic Classification Model
- [ ] Week 3: Exploratory Data Analysis
- [ ] Week 4: Streamlit Dashboard

## Week 1 — Data Cleaning & Visualization

First, I examined the dataset's structure — column names, row count, and data types — to understand what I was working with before deciding on further steps.

The main focus of the week was data cleaning: I identified missing values and made deliberate decisions on each — some (like blank author names) were filled with a placeholder since they weren't needed for later modeling, while others (like blank descriptions) were dropped since that column would be used as model input. I also removed exact duplicate rows.

For outlier detection, I used the IQR method on text length to flag unusually short or long articles, but rather than dropping everything the statistics flagged, I manually inspected the flagged rows first. This showed that very short texts were genuinely low-content or junk data worth removing, while unusually long texts were legitimate, detailed articles worth keeping — reinforcing that statistical flags are a starting point, not an automatic decision rule.

I also reviewed the dataset's 42 category labels and merged clearly overlapping ones (e.g. "ARTS" and "ARTS & CULTURE") down to 33 cleaner categories.

Finally, I visualized the cleaned data — a bar chart of category distribution, a histogram of text length, and a line chart of articles per year — before saving the cleaned dataset for the next stage.

**Files:** `notebooks/week1_cleaning_eda.ipynb`, `data/cleaned/news_cleaned.csv`
