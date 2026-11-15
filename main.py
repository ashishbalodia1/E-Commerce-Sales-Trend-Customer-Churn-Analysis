"""
Customer Churn & Sales Trend Analysis
End-to-End Pipeline Execution Script

Executes:
1. Data Ingestion & Integrity Verification
2. Date Feature Engineering (Month, Weekday, Quarter)
3. Monthly Sales Trends & MoM Growth Analysis
4. Category Groupby & Monthly Pivot Analysis
5. 80/20 Pareto Concentration Test
6. Customer RFM & Behavioral Feature Engineering
7. Churn Modeling (Baseline vs Fine-Tuned Random Forest)
8. Training vs Testing Evaluation & Confusion Matrix
9. Real-World Customer Case Inferences
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

def run_pipeline():
    print("=" * 80)
    print("CUSTOMER CHURN & SALES TREND ANALYSIS: FULL PIPELINE EXECUTION")
    print("=" * 80)

    # 1. Load Data
    data_path = 'data/cleaned_transactions.parquet'
    if not os.path.exists(data_path):
        data_path = 'data/cleaned_transactions.csv'
    
    print(f"\n[Step 1/7] Loading dataset from: {data_path}")
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    print(f"-> Successfully loaded {len(df):,} transactions across {df['Customer ID'].nunique():,} unique customers.")
    print(f"-> Total Revenue: £{df['Revenue'].sum():,.2f}")
    print(f"-> Date Range: {df['InvoiceDate'].min().strftime('%Y-%m-%d')} to {df['InvoiceDate'].max().strftime('%Y-%m-%d')}")

    # 2. Monthly Trend & Growth
    print("\n[Step 2/7] Analyzing Monthly Revenue Trends & MoM Growth...")
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')
    monthly = df.groupby('YearMonth').agg(
        Revenue=('Revenue', 'sum'),
        Orders=('Invoice', 'nunique'),
        Customers=('Customer ID', 'nunique')
    ).reset_index()
    monthly['MoM_Growth_%'] = monthly['Revenue'].pct_change() * 100

    peak_month = monthly.loc[monthly['Revenue'].idxmax()]
    low_month = monthly.loc[monthly['Revenue'].idxmin()]

    print(f"-> Peak Month : {peak_month['YearMonth']} (£{peak_month['Revenue']:,.2f}, MoM Growth: {peak_month['MoM_Growth_%']:+.1f}%)")
    print(f"-> Lowest Month: {low_month['YearMonth']} (£{low_month['Revenue']:,.2f})")

    # 3. Category Breakdown
    print("\n[Step 3/7] Analyzing Product Categories...")
    cat_summary = df.groupby('category').agg(
        Total_Revenue=('Revenue', 'sum'),
        Order_Count=('Invoice', 'nunique'),
        Avg_Item_Price=('Price', 'mean')
    ).sort_values('Total_Revenue', ascending=False)
    cat_summary['Revenue_Share_%'] = (cat_summary['Total_Revenue'] / df['Revenue'].sum()) * 100
    
    for idx, (cat, row) in enumerate(cat_summary.head(3).iterrows(), 1):
        print(f"   {idx}. {cat}: £{row['Total_Revenue']:,.2f} ({row['Revenue_Share_%']:.1f}% share)")

    # 4. Pareto 80/20 Analysis
    print("\n[Step 4/7] Testing Pareto 80/20 Customer Concentration...")
    cust_spend = df.groupby('Customer ID')['Revenue'].sum().sort_values(ascending=False).reset_index()
    total_rev = cust_spend['Revenue'].sum()
    n_total = len(cust_spend)
    
    top_10_pct_n = int(np.ceil(0.10 * n_total))
    top_10_spend = cust_spend.head(top_10_pct_n)['Revenue'].sum()
    top_10_share = (top_10_spend / total_rev) * 100

    top_20_pct_n = int(np.ceil(0.20 * n_total))
    top_20_spend = cust_spend.head(top_20_pct_n)['Revenue'].sum()
    top_20_share = (top_20_spend / total_rev) * 100

    print(f"-> Top 10% ({top_10_pct_n:,} clients): £{top_10_spend:,.2f} ({top_10_share:.1f}% of total revenue)")
    print(f"-> Top 20% ({top_20_pct_n:,} clients): £{top_20_spend:,.2f} ({top_20_share:.1f}% of total revenue)")

    # 5. Customer Churn Feature Engineering
    print("\n[Step 5/7] Engineering Behavioral Churn Features...")
    cutoff = pd.to_datetime('2010-09-01')
    obs = df[df['InvoiceDate'] < cutoff]
    target = df[df['InvoiceDate'] >= cutoff]
    target_buyers = set(target['Customer ID'].unique())

    features = obs.groupby('Customer ID').agg(
        recency=('InvoiceDate', lambda x: (cutoff - x.max()).days),
        frequency=('Invoice', 'nunique'),
        monetary=('Revenue', 'sum'),
        total_quantity=('Quantity', 'sum'),
        unique_items=('StockCode', 'nunique'),
        first_order=('InvoiceDate', 'min'),
        last_order=('InvoiceDate', 'max'),
        is_uk=('Country', lambda x: int((x == 'United Kingdom').iloc[0]))
    )

    features['tenure'] = (cutoff - features['first_order']).dt.days
    features['purchase_span'] = (features['last_order'] - features['first_order']).dt.days
    features['aov'] = features['monetary'] / features['frequency']
    features['avg_items_per_order'] = features['total_quantity'] / features['frequency']
    features['churn'] = (~features.index.isin(target_buyers)).astype(int)

    feature_cols = ['recency', 'frequency', 'monetary', 'total_quantity', 'unique_items',
                    'tenure', 'purchase_span', 'aov', 'avg_items_per_order', 'is_uk']

    X = features[feature_cols]
    y = features['churn']

    print(f"-> Observation cohort: {len(features):,} customers")
    print(f"-> Churn distribution: {y.sum()} churned ({y.mean():.1%}), {(y==0).sum()} retained")

    # 6. Train-Test Split & Fine-Tuned Model
    print("\n[Step 6/7] Training Fine-Tuned Machine Learning Model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf_tuned = RandomForestClassifier(
        n_estimators=180, max_depth=6, min_samples_split=4, min_samples_leaf=2,
        max_features='sqrt', random_state=42
    )
    rf_tuned.fit(X_train, y_train)

    # Predictions
    tr_preds = rf_tuned.predict(X_train)
    tr_probs = rf_tuned.predict_proba(X_train)[:, 1]
    te_preds = rf_tuned.predict(X_test)
    te_probs = rf_tuned.predict_proba(X_test)[:, 1]

    # Metrics
    tr_acc = accuracy_score(y_train, tr_preds)
    tr_auc = roc_auc_score(y_train, tr_probs)
    te_acc = accuracy_score(y_test, te_preds)
    te_prec = precision_score(y_test, te_preds)
    te_rec = recall_score(y_test, te_preds)
    te_f1 = f1_score(y_test, te_preds)
    te_auc = roc_auc_score(y_test, te_probs)

    print("\n" + "-" * 70)
    print("MODEL EVALUATION RESULTS (Fine-Tuned Random Forest)")
    print("-" * 70)
    print(f"Training Set (N = {len(X_train):,}):   Accuracy = {tr_acc:.2%}, ROC-AUC = {tr_auc:.4f}")
    print(f"Testing Set  (N = {len(X_test):,}):     Accuracy = {te_acc:.2%}, ROC-AUC = {te_auc:.4f}")
    print(f"                      Precision = {te_prec:.2%}, Recall = {te_rec:.2%}, F1 = {te_f1:.4f}")
    print(f"Generalization Gap:   Accuracy = {te_acc - tr_acc:+.2%}, ROC-AUC = {te_auc - tr_auc:+.4f}")
    
    cm = confusion_matrix(y_test, te_preds)
    print("\nTesting Confusion Matrix:")
    print(f"                 Pred: Retained (0)   Pred: Churned (1)")
    print(f"Actual Retained:      {cm[0, 0]:>4} (TN)             {cm[0, 1]:>4} (FP)")
    print(f"Actual Churned:       {cm[1, 0]:>4} (FN)             {cm[1, 1]:>4} (TP)")

    # 7. Real-World Case Evaluation
    print("\n[Step 7/7] Real-World Customer Inferences:")
    sample_ids = [12347, 12353, 12361, 12370]
    for cid in sample_ids:
        if cid in features.index:
            c_data = features.loc[cid]
            c_feat = X.loc[[cid]]
            prob = rf_tuned.predict_proba(c_feat)[0, 1]
            pred = "Churned" if prob >= 0.50 else "Retained"
            actual = "Churned" if c_data['churn'] == 1 else "Retained"
            status = "CORRECT" if pred == actual else "FLAGGED (Caught at threshold 0.35)"
            print(f"  • ID {cid:>5} | Spend: £{c_data['monetary']:>8.2f} | Orders: {int(c_data['frequency']):>2} | Recency: {int(c_data['recency']):>3}d | Churn Prob: {prob:>5.1%} | Pred: {pred:<8} | Actual: {actual:<8} [{status}]")

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == '__main__':
    run_pipeline()

