# 🛒 Retail Analytics using Machine Learning
### Customer Segmentation, Campaign Effectiveness Analysis, and Product Recommendation System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![PanelOLS](https://img.shields.io/badge/PanelOLS-Econometrics-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

---

## 📌 Project Overview

This project presents an end-to-end **Retail Analytics Platform** developed using the **Dunnhumby "The Complete Journey" dataset**. It combines three major machine learning and analytics problems into a single business-oriented solution:

- 👥 Customer Segmentation
- 📈 Campaign Effectiveness Analysis
- 🛍️ Personalized Product Recommendation

An interactive **Streamlit Application** integrates all three modules, allowing users to explore customer behavior, analyze campaign performance, and generate personalized product recommendations.

---

## 🎯 Objectives

The project addresses three key business questions:

### 1. Customer Segmentation
Identify meaningful customer groups based on purchasing behavior to enable targeted marketing strategies.

### 2. Campaign Effectiveness Analysis
Estimate the incremental impact of marketing campaigns on customer spending using panel regression with fixed effects.

### 3. Product Recommendation
Recommend products to individual households using collaborative filtering based on purchasing similarity.

---

# 📊 Dataset

Dataset: **Dunnhumby – The Complete Journey**

The project uses multiple relational tables including:

- Transaction Data
- Product Information
- Household Demographics
- Campaign Information
- Coupon Details
- Coupon Redemptions
- Causal Data

Approximately:

- **2,500 Households**
- **2 Years of Transactions**
- **Millions of Purchase Records**

---

# 🏗️ Project Architecture

```
Retail Analytics
│
├── Customer Segmentation
│      │
│      ├── Feature Engineering
│      ├── PCA
│      └── K-Means Clustering
│
├── Campaign Effectiveness
│      │
│      ├── Household Panel Dataset
│      ├── Fixed Effects Panel Regression
│      ├── Campaign Lag Analysis
│      └── Incremental Sales Estimation
│
├── Product Recommendation
│      │
│      ├── Household-Item Matrix
│      ├── Cosine Similarity
│      ├── Collaborative Filtering
│      └── Top-N Recommendations
│
└── Streamlit Application
```

---

# 🚀 Module 1: Customer Segmentation

### Workflow

- Data Cleaning
- Feature Engineering
- Outlier Treatment
- Correlation Analysis
- Variance Inflation Factor (VIF)
- Principal Component Analysis (PCA)
- K-Means Clustering
- Cluster Profiling

### Features Used

- Total Sales
- Total Quantity
- Basket Count
- Average Basket Size
- Average Spend
- Product Diversity
- Discount Usage
- Coupon Usage

### Model

- PCA
- K-Means Clustering

### Final Result

Three interpretable customer segments:

- Loyal Customers
- One-Time Buyers
- Dormant Customers

---

# 📈 Module 2: Campaign Effectiveness

Instead of comparing simple pre- and post-campaign sales, the project estimates campaign impact using **Panel Regression**.

## Approach

- Household-week level panel dataset
- Fixed Effects Panel Regression
- Four-week campaign lag
- Campaign × Segment interaction effects
- Entity Fixed Effects
- Time Fixed Effects
- Clustered Standard Errors

### Variables

Dependent Variables

- Weekly Sales
- Weekly Quantity (Not considered for main model)

Independent Variables

- Campaign Type
- Campaign Lag
- Sales Lags
- Basket Count
- Product Diversity
- Campaign × Segment Interaction

### Model

PanelOLS

### Key Findings

- Campaign response varies significantly across customer segments.
- Loyal customers exhibit the strongest positive campaign lift.
- One-time buyers show little or no measurable lift.
- Fixed-effects regression substantially outperforms naive campaign comparisons.

---

# 🛍️ Module 3: Product Recommendation

A **User-Based Collaborative Filtering** recommender was developed.

## Workflow

Transaction Data

↓

Household × Sub-Commodity Matrix

↓

Cosine Similarity

↓

Nearest Neighbour Search

↓

Top-N Product Recommendations

### Recommendation Logic

For each household:

1. Find the most similar households.
2. Aggregate products purchased by those neighbours.
3. Remove products already purchased by the target household.
4. Recommend the highest-ranked unseen products.

---

# 📊 Streamlit Application

The project includes an interactive business app built with Streamlit.

## Features

### Overview

- Business KPIs
- Segment Distribution
- Campaign Summary

### Customer Segmentation

- Household Profile
- Segment Information
- Spending Analysis
- Purchase Behaviour

### Campaign Effectiveness

- Regression Coefficients
- Campaign Lift
- Segment-wise Campaign Performance
- Campaign Comparison

### Product Recommendation

- Similar Households
- Personalized Recommendations
- Recommendation Scores
- Explanation of Recommendations

---

# 📈 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Linearmodels (PanelOLS)
- Plotly
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

# 💡 Key Business Insights

- Customers exhibit distinct purchasing behaviours that can be effectively segmented.
- Marketing campaigns are not equally effective across all customer groups.
- Loyal customers generate the highest incremental campaign lift.
- Collaborative filtering enables personalized recommendations based on historical purchasing behaviour.
- Combining segmentation, campaign analytics, and recommendation systems provides a comprehensive retail decision-support framework.

---

# 📊 Future Improvements

- Hybrid Recommendation Systems
- Deep Learning Recommenders
- Campaign-Aware Recommendations
- Real-Time Recommendation Pipeline
- Cold-Start Solutions
- A/B Testing Framework
- Time-Series Sales Forecasting

---

# 🖥️ Running the Project

Clone the repository:

```bash
git clone https://github.com/yourusername/retail-analytics.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# 📷 Dashboard Preview

> Add screenshots of your Streamlit dashboard here.

```
Overview Dashboard

Customer Segmentation

Campaign Effectiveness

Recommendation Engine
```

---

# 📚 References

- Dunnhumby – The Complete Journey Dataset
- Scikit-Learn Documentation
- Streamlit Documentation
- Linearmodels Documentation
- Optuna Documentation

---

# 👨‍💻 Author

**Jeremiah Ranen R, Chatrapal Singh, Chaithra Kulal, Meesala Sandeep Kumar, Mrunal Patil, Vaishnavi Panthul**

Postgraduate Project

Retail Analytics using Machine Learning

Customer Segmentation • Campaign Effectiveness • Product Recommendation

---
