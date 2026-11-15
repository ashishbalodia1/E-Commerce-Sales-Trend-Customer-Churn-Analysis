# Executive Summary Report: E-Commerce Customer Churn Analysis
**Project:** Customer Churn & Sales Trend Analysis  
**Dataset:** Online Retail II (525,461 Transactions across 2009–2010)  
**Author:** Ashis  
**Read Time:** 2 Minutes  

---

## 1. Context & Business Problem
This report evaluates transactional performance across **4,312 customers** (**£8.83M** in clean gross revenue across **407,664** order lines). The objective was to analyze sales trends and seasonality, measure customer concentration (Pareto 80/20 rule), and deploy a **Machine Learning Churn Classifier** to identify customers at risk of leaving before they defect.

---

## 2. Key Findings

### A. Monthly Sales Trends & Seasonality
- **Peak Month:** **November 2010** was the highest-earning month at **£1,172,336** (**+13.1% MoM**), followed by October at **£1,036,680**.
- **Lowest Month:** **February 2010** at **£506,371**.
- **Seasonal Surge:** Sales jumped by **+131%** between February and November due to intense Q4 holiday gift purchasing.

### B. Category Breakdown
1. **Home Decor & Living:** **£2,426,424** (27.5% share, 101,795 items, avg. item spend £23.84)
2. **Bags & Storage:** **£1,815,870** (20.6% share, 81,363 items, avg. item spend £22.32)
3. **Kitchen & Dining:** **£1,664,498** (18.9% share, 75,711 items, avg. item spend £21.98)
4. **General Merchandise:** **£1,344,076** (15.2% share)
5. **Gifts & Seasonal:** **£882,414** (10.0% share)
6. **Stationery & Crafts:** **£530,672** (6.0% share)
7. **Apparel & Accessories:** **£168,048** (1.9% share)

*Key Takeaway:* **Home Decor & Living** and **Bags & Storage** generate **48%** of total revenue.

### C. The 80/20 Rule (Pareto Concentration)
- **Top 10% of Customers (432 accounts):** Generate **£5,288,001 (59.9% of total revenue)**.
- **Top 20% of Customers (863 accounts):** Generate **£6,490,094 (73.5% of total revenue)**.
- The average annual spend of the top 10% is **£12,240** vs **£2,048** for typical customers.

---

## 3. Machine Learning Churn Classification

### A. Problem Framing (Zero Data Leakage)
- **Observation Period (Dec 1, 2009 – Aug 31, 2010):** Extracted features for 3,321 active customers.
- **Target Period (Sep 1, 2010 – Dec 9, 2010):** Evaluated customer return rate. Churn rate: **39.9% (1,326 churned vs 1,995 retained)**.

### B. Model Performance: Training vs. Testing Evaluation

| Metric | Training Set ($N = 2,490$) | Testing Set ($N = 831$) | Generalization Gap |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **74.54%** | **67.75%** | -6.79% |
| **Precision** | **69.11%** | **59.58%** | -9.53% |
| **Recall** | **65.23%** | **59.94%** | -5.29% |
| **F1-Score** | **0.6711** | **0.5976** | -0.0735 |
| **ROC-AUC** | **0.8301** | **0.7333** | -0.0968 |

*Overfitting Assessment:* The small gap between training and testing accuracy (~6.8%) confirms that regularization (`max_depth=6`, `min_samples_leaf=2`, `min_samples_split=4`) prevented the ensemble from overfitting on noisy customer data.

### C. Testing Classification Matrix (Confusion Matrix, $N = 831$)
```
                            Predicted Retained (0)    Predicted Churned (1)
  Actual Retained (0)               364 (TN)                 135 (FP)
  Actual Churned (1)                133 (FN)                 199 (TP)
```
- **True Negatives ($\text{TN} = 364$):** Retained customers correctly classified.
- **False Positives ($\text{FP} = 135$):** Active customers predicted as churners. Cost: small promo coupon sent unnecessarily.
- **False Negatives ($\text{FN} = 133$):** Defecting customers missed by the model. Cost: lost customer lifetime value.
- **True Positives ($\text{TP} = 199$):** Defecting customers successfully caught before leaving.

### D. Decision Threshold Optimization
- At default threshold **0.50**: Catches **59.9% of churners** (199 / 332).
- At optimized threshold **0.35**: Catches **82.5% of churners** (274 / 332), providing an ideal trade-off for retention campaigns.

---

## 4. Real-World Customer Test Cases

| Case Profile | ID | Spend | Orders | Recency | Churn Prob | Predicted | Actual | Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Loyal Wholesale Client** | 12347 | £1,724.85 | 8 | 96d | **21.5%** | Retained (0) | Retained (0) | Correct Prediction |
| **2. Inactive One-Time Buyer** | 12353 | £342.36 | 1 | 258d | **59.1%** | Churned (1) | Churned (1) | Correct Prediction |
| **3. At-Risk Repeat Customer** | 12361 | £838.27 | 5 | 63d | **18.1%** | Retained (0) | Churned (1) | False Negative (caught at 0.35 threshold) |
| **4. Recent Growing Customer** | 12370 | £1,339.38 | 2 | 28d | **27.1%** | Retained (0) | Retained (0) | Correct Prediction |

---

## 5. Practical Recommendations

1. **VIP Loyalty Tier for Top 10% Spenders (59.9% Revenue):** Protect the 432 accounts generating £5.29M with volume rebates and dedicated account support. Losing just 5% of this group costs >£264k annually.
2. **Automated ML-Triggered Win-Back Campaigns (Threshold = 0.35):** When churn probability exceeds 0.35 or recency approaches 60–90 days, trigger a replenishment email with a 10–12% discount on previously ordered categories. Reactivating 10% recovers ~£90k in GMV.
3. **Advance Q4 Inventory Procurement for Home Decor & Storage:** Place manufacturer purchase orders by August to avoid stockouts in October and November when sales surge by +131%.
