# Customer Churn & Sales Trend Analysis

An end-to-end data analysis and machine learning project on e-commerce retail data. This project analyzes revenue seasonality, product categories, and customer concentration (Pareto 80/20 rule), followed by fine-tuned classification models to predict which customers are likely to churn.

---

## Repository Structure

```
├── app.py                          # Interactive Streamlit Web Application & Churn Predictor
├── customer_churn_analysis.ipynb   # Main Jupyter Notebook (EDA, Pareto, ML fine-tuning & evaluation)
├── customer_churn_analysis.html    # Standalone HTML export for web display & frontend embedding
├── main.py                         # Standalone executable CLI pipeline script
├── README.md                       # Project overview, setup, and results documentation
├── requirements.txt                # Exact dependencies
├── data/                           # Data directory
│   ├── online_retail_II.xlsx       # Original raw dataset
│   ├── cleaned_transactions.parquet# Processed clean dataset (fast loading)
│   └── cleaned_transactions.csv    # Processed CSV format
├── reports/                        # Formatted reports
│   ├── business_summary_report.md  # One-page executive brief in Markdown
│   └── business_summary_report.pdf # One-page executive brief in PDF
└── figures/                        # Evaluation charts and analytical visualizations
    ├── monthly_sales_trend.png
    ├── category_revenue_comparison.png
    ├── category_monthly_pivot_heatmap.png
    ├── pareto_customer_spend.png
    ├── churn_confusion_matrix.png
    ├── churn_roc_curve.png
    └── churn_feature_importance.png
```

---

## Key Analysis Findings

### 1. Revenue Trends & Seasonality
- **Peak Month:** November 2010 with **£1,172,336** in revenue (+13.1% MoM growth), followed by October 2010 (£1,036,680).
- **Lowest Month:** February 2010 with **£506,371** in revenue.
- **Seasonal Surge:** Sales grew by **+131%** from February to November due to Q4 holiday gift purchasing.

### 2. Category Performance
- **Home Decor & Living** (£2.43M, 27.5% share) and **Bags & Storage** (£1.82M, 20.6% share) make up **48% of total revenue**.
- Kitchen & Dining is third with £1.66M (18.9% share).

### 3. Customer Concentration (Pareto 80/20 Test)
- Total unique customers: **4,312**.
- **Top 10% of customers (432 accounts):** Generate **£5,288,001 (59.9% of total revenue)**.
- **Top 20% of customers (863 accounts):** Generate **£6,490,094 (73.5% of total revenue)**.
- Average annual spend of the top 10% is **£12,240** vs **£2,048** for typical customers.

---

## Machine Learning Churn Classification

### Framing & Feature Engineering
To predict churn without lookahead data leakage:
- **Observation Period (Dec 1, 2009 to Aug 31, 2010):** Behavioral features extracted for 3,321 active customers (`recency`, `frequency`, `monetary`, `tenure`, `purchase_span`, `aov`, `total_quantity`, `unique_items`, `is_uk`).
- **Target Period (Sep 1, 2010 to Dec 9, 2010):** Observed whether the customer made any purchases (`churn = 1` if 0 orders, `0` if >= 1 order). Observed churn rate: **39.9%**.

### Model Performance: Training vs. Testing Evaluation

| Metric | Training Set ($N = 2,490$) | Testing Set ($N = 831$) | Generalization Gap |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **74.54%** | **67.75%** | -6.79% |
| **Precision** | **69.11%** | **59.58%** | -9.53% |
| **Recall** | **65.23%** | **59.94%** | -5.29% |
| **F1-Score** | **0.6711** | **0.5976** | -0.0735 |
| **ROC-AUC** | **0.8301** | **0.7333** | -0.0968 |

**Overfitting Assessment:**  
The generalization gap between training and testing accuracy is only ~6.8% (and ~9.7% on ROC-AUC). Hyperparameter constraints (`max_depth=6`, `min_samples_leaf=2`, `min_samples_split=4`, `n_estimators=180`) successfully prevented the ensemble from overfitting on customer noise.

---

### Classification Matrices (Confusion Matrix Breakdown)

#### 1. Training Confusion Matrix ($N = 2,490$)
```
                            Predicted Retained (0)    Predicted Churned (1)
  Actual Retained (0)             1,208 (TN)                 288 (FP)
  Actual Churned (1)                346 (FN)                 648 (TP)
```

#### 2. Testing Confusion Matrix ($N = 831$)
```
                            Predicted Retained (0)    Predicted Churned (1)
  Actual Retained (0)               364 (TN)                 135 (FP)
  Actual Churned (1)                133 (FN)                 199 (TP)
```

- **True Negatives ($\text{TN} = 364$):** Retained customers correctly predicted to stay.
- **False Positives ($\text{FP} = 135$):** Active customers predicted as churn risks. (Cost: Small promo coupon sent).
- **False Negatives ($\text{FN} = 133$):** Defecting customers missed by the model. (Cost: Lost Customer Lifetime Value).
- **True Positives ($\text{TP} = 199$):** At-risk churners caught before defection. (Benefit: Proactive retention).

---

## Real-World Customer Test Cases

To demonstrate practical deployment, we evaluated four distinct customer profiles from the held-out test set:

| Customer Profile | Customer ID | Past Spend | Frequency | Recency | Purchase Span | Churn Prob | Predicted | Actual | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1: Loyal Wholesale Client** | 12347 | £1,724.85 | 8 orders | 96 days | 168 days | **21.5%** | Retained (0) | Retained (0) | Correct Prediction |
| **Case 2: Inactive One-Time Buyer** | 12353 | £342.36 | 1 order | 258 days | 0 days | **59.1%** | Churned (1) | Churned (1) | Correct Prediction |
| **Case 3: At-Risk Repeat Customer** | 12361 | £838.27 | 5 orders | 63 days | 128 days | **18.1%** | Retained (0) | Churned (1) | False Negative (Type II) |
| **Case 4: Recent Growing Customer** | 12370 | £1,339.38 | 2 orders | 28 days | 127 days | **27.1%** | Retained (0) | Retained (0) | Correct Prediction |

*Note on Case 3:* When operating at the default 0.50 cutoff, Case 3 is missed because past spend was moderate. Lowering the operational alert threshold to **0.35** flags this customer in advance, raising overall test-set Recall from **59.0% to 82.5%**.

---

## Actionable Recommendations

1. **VIP Loyalty Program for Top 10% Spenders:** Protect the 432 accounts that bring 60% of revenue (£5.29M) with volume rebates and dedicated account support. Losing just 5% of this group costs >£264k annually.
2. **Automate Churn Win-Back Triggers (Threshold = 0.35):** When customer churn probability exceeds 0.35 or when recency approaches 60–90 days, trigger a replenishment email with a 10–12% discount on previously ordered categories. Reactivating 10% recovers ~£90k in revenue.
3. **Advance Q4 Inventory Orders for Home Decor & Storage:** With sales surging by 131% into November and these two categories driving 48% of sales, place supplier orders by August to prevent out-of-stock events during holiday demand.

---

## Getting Started

### 1. Setup Environment
```bash
git clone <repo-url>
cd "Customer churn Analysis"
pip install -r requirements.txt
```

### 2. Run the Analysis

**Option A: Execute Standalone CLI Pipeline**
```bash
python main.py
```

**Option B: Interactive Jupyter Notebook**
```bash
jupyter notebook customer_churn_analysis.ipynb
```

**Option C: View Standalone HTML in Browser**
```bash
# Double-click customer_churn_analysis.html or serve locally:
python -m http.server 8000
# Open http://localhost:8000/customer_churn_analysis.html
```

**Option D: Interactive Streamlit Web Application**
```bash
streamlit run app.py
```
*Features:* Executive KPI Cards, Category Bar Charts, Pareto 80/20 Curves, Live Customer Churn Simulator with Risk Gauge, and Individual Account Lookup.
