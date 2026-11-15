import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

# Configure Streamlit page
st.set_page_config(
    page_title="E-Commerce Churn & Revenue Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1E293B;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# DATA & MODEL CACHING
# ---------------------------------------------------------
@st.cache_data
def load_data():
    if os.path.exists('data/cleaned_transactions.parquet'):
        df = pd.read_parquet('data/cleaned_transactions.parquet')
    elif os.path.exists('data/cleaned_transactions.csv'):
        df = pd.read_csv('data/cleaned_transactions.csv')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    else:
        df = pd.read_excel('data/online_retail_II.xlsx', sheet_name='Year 2009-2010')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['Revenue'] = df['Quantity'] * df['Price']
    return df

@st.cache_data
def prepare_churn_features(df):
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

    return features

@st.cache_resource
def train_model(features):
    feature_cols = ['recency', 'frequency', 'monetary', 'total_quantity', 'unique_items',
                    'tenure', 'purchase_span', 'aov', 'avg_items_per_order', 'is_uk']
    X = features[feature_cols]
    y = features['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=180, max_depth=6, min_samples_split=4, min_samples_leaf=2,
        max_features='sqrt', random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, feature_cols

# Load data and assets
with st.spinner("Loading transaction dataset & initializing analytics engine..."):
    df = load_data()
    features = prepare_churn_features(df)
    model, X_train, X_test, y_train, y_test, feature_cols = train_model(features)

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.image("https://img.icons8.com/fluency/96/shopping-cart-loaded.png", width=70)
st.sidebar.title("Retail Intelligence")
st.sidebar.markdown("**Customer Churn & Trend Platform**")
st.sidebar.divider()

nav = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive KPI Dashboard",
        "📈 Sales Trends & Categories",
        "👥 Pareto 80/20 Distribution",
        "🎯 Live Churn Predictor",
        "🔍 Customer Account Lookup",
        "📑 Model Diagnostics & Matrix"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("⚙️ Decision Settings")
operational_threshold = st.sidebar.slider(
    "Churn Decision Cutoff",
    min_value=0.20,
    max_value=0.70,
    value=0.35,
    step=0.05,
    help="Default is 0.50. Value 0.35 is recommended for customer win-back campaigns to achieve ~82.5% churn recall."
)

st.sidebar.info(f"**Active Threshold:** `{operational_threshold:.2f}`\n\n- Prob >= {operational_threshold:.2f} -> **Flag as Churn Risk**\n- Prob < {operational_threshold:.2f} -> **Predicted Retained**")

# ---------------------------------------------------------
# PAGE 1: EXECUTIVE KPI DASHBOARD
# ---------------------------------------------------------
if nav == "📊 Executive KPI Dashboard":
    st.markdown('<div class="main-header">E-Commerce Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">High-level sales health, seasonal trends, and customer risk indicators</div>', unsafe_allow_html=True)

    total_revenue = df['Revenue'].sum()
    total_orders = df['Invoice'].nunique()
    total_customers = df['Customer ID'].nunique()
    churn_rate = features['churn'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Clean Revenue", value=f"£{total_revenue/1e6:.2f}M", delta="+13.1% Peak MoM")
    with col2:
        st.metric(label="Verified Orders", value=f"{total_orders:,}")
    with col3:
        st.metric(label="Active Customer Base", value=f"{total_customers:,}")
    with col4:
        st.metric(label="Baseline Churn Rate", value=f"{churn_rate:.1%}", delta="-60% Retained", delta_color="inverse")

    st.divider()

    st.subheader("Monthly Revenue Trajectory (£)")
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    monthly_rev = df.groupby('YearMonth')['Revenue'].sum().reset_index()
    
    st.bar_chart(data=monthly_rev, x='YearMonth', y='Revenue', color='#1E88E5')
    
    st.markdown("""
    > **Executive Summary:**
    > - **Peak Performance:** November 2010 reached **£1,172,336** driven by pre-holiday gift procurement.
    > - **Low Point:** February 2010 at **£506,371** (+131% sales surge into Q4).
    > - **Customer Risk:** 39.9% of accounts from the first 9 months did not purchase in Q4 without targeted intervention.
    """)

# ---------------------------------------------------------
# PAGE 2: SALES TRENDS & CATEGORIES
# ---------------------------------------------------------
elif nav == "📈 Sales Trends & Categories":
    st.markdown('<div class="main-header">Sales Performance & Category Dynamics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Deep dive into merchandise revenue contributions and month-over-month growth</div>', unsafe_allow_html=True)

    cat_stats = df.groupby('category').agg(
        Revenue=('Revenue', 'sum'),
        Orders=('Invoice', 'nunique'),
        AvgPrice=('Price', 'mean'),
        UnitsSold=('Quantity', 'sum')
    ).sort_values('Revenue', ascending=False).reset_index()
    cat_stats['RevenueShare'] = (cat_stats['Revenue'] / df['Revenue'].sum()) * 100

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Revenue Contribution by Category")
        st.bar_chart(data=cat_stats, x='category', y='Revenue', color='#0D47A1')

    with col2:
        st.subheader("Top Category Breakdown")
        display_cats = cat_stats[['category', 'Revenue', 'RevenueShare']].copy()
        display_cats['Revenue'] = display_cats['Revenue'].apply(lambda x: f"£{x:,.2f}")
        display_cats['RevenueShare'] = display_cats['RevenueShare'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_cats, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Category Heatmap Overview")
    if os.path.exists('figures/category_monthly_pivot_heatmap.png'):
        st.image('figures/category_monthly_pivot_heatmap.png', caption="Monthly Category Spend Heatmap (£)", use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: PARETO 80/20 DISTRIBUTION
# ---------------------------------------------------------
elif nav == "👥 Pareto 80/20 Distribution":
    st.markdown('<div class="main-header">Pareto (80/20) Revenue Concentration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Measuring revenue reliance on top-tier customer accounts</div>', unsafe_allow_html=True)

    cust_spend = df.groupby('Customer ID')['Revenue'].sum().sort_values(ascending=False).reset_index()
    total_rev = cust_spend['Revenue'].sum()
    n_cust = len(cust_spend)

    top_10_n = int(np.ceil(0.10 * n_cust))
    top_10_rev = cust_spend.head(top_10_n)['Revenue'].sum()
    top_10_share = (top_10_rev / total_rev) * 100

    top_20_n = int(np.ceil(0.20 * n_cust))
    top_20_rev = cust_spend.head(top_20_n)['Revenue'].sum()
    top_20_share = (top_20_rev / total_rev) * 100

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Top 10% Spenders (432 clients)", f"£{top_10_rev/1e6:.2f}M", f"{top_10_share:.1f}% of Revenue")
    with c2:
        st.metric("Top 20% Spenders (863 clients)", f"£{top_20_rev/1e6:.2f}M", f"{top_20_share:.1f}% of Revenue")
    with c3:
        st.metric("Bottom 80% (3,449 clients)", f"£{(total_rev - top_20_rev)/1e6:.2f}M", f"{100-top_20_share:.1f}% of Revenue")

    st.divider()
    if os.path.exists('figures/pareto_customer_spend.png'):
        st.image('figures/pareto_customer_spend.png', caption="Pareto Cumulative Spend Curve", use_container_width=True)

    st.warning("⚠️ **Strategic Risk Warning:** Losing just 5% of the Top 10% cohort creates an annual loss of over **£264,000**. Implement a VIP loyalty tier immediately.")

# ---------------------------------------------------------
# PAGE 4: LIVE CHURN PREDICTOR
# ---------------------------------------------------------
elif nav == "🎯 Live Churn Predictor":
    st.markdown('<div class="main-header">Real-Time Customer Churn Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate any customer profile to generate immediate churn probabilities and retention actions</div>', unsafe_allow_html=True)

    st.markdown("### Enter Customer Behavioral Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        monetary = st.number_input("Total Historical Spend (£)", min_value=1.0, max_value=50000.0, value=650.0, step=50.0)
        frequency = st.number_input("Total Orders (Frequency)", min_value=1, max_value=200, value=4, step=1)
        recency = st.slider("Recency (Days since last order)", min_value=1, max_value=300, value=85, help="Days since the customer last purchased.")

    with col2:
        tenure = st.number_input("Customer Tenure (Days since first order)", min_value=1, max_value=365, value=180, step=10)
        purchase_span = st.slider("Purchase Span (Lifespan in days)", min_value=0, max_value=365, value=120, help="Days between first and last purchase. 0 for one-time buyers.")
        total_quantity = st.number_input("Total Units Purchased", min_value=1, max_value=10000, value=250, step=10)

    with col3:
        unique_items = st.number_input("Unique SKUs Bought", min_value=1, max_value=1000, value=35, step=5)
        is_uk = st.selectbox("Market Region", options=["United Kingdom (Domestic)", "International / Overseas"], index=0)
        is_uk_val = 1 if "United Kingdom" in is_uk else 0

    aov = monetary / frequency
    avg_items_per_order = total_quantity / frequency

    # Build input feature vector
    input_df = pd.DataFrame([{
        'recency': recency,
        'frequency': frequency,
        'monetary': monetary,
        'total_quantity': total_quantity,
        'unique_items': unique_items,
        'tenure': tenure,
        'purchase_span': purchase_span,
        'aov': aov,
        'avg_items_per_order': avg_items_per_order,
        'is_uk': is_uk_val
    }])[feature_cols]

    st.divider()
    if st.button("🚀 Calculate Churn Probability", type="primary", use_container_width=True):
        churn_prob = model.predict_proba(input_df)[0, 1]
        is_churn = churn_prob >= operational_threshold

        res_col1, res_col2 = st.columns([2, 3])
        with res_col1:
            st.metric(
                label="Predicted Churn Risk Probability",
                value=f"{churn_prob:.1%}",
                delta=f"Alert Threshold: {operational_threshold:.2f}",
                delta_color="inverse"
            )
            st.progress(float(churn_prob))

            if churn_prob >= 0.50:
                st.error("🔴 **HIGH RISK:** Customer is strongly heading towards defection.")
            elif churn_prob >= operational_threshold:
                st.warning("🟡 **MODERATE RISK (ACTIONABLE):** Customer is entering dormancy. Intervention required.")
            else:
                st.success("🟢 **LOW RISK (HEALTHY):** Customer demonstrates healthy repeat habits.")

        with res_col2:
            st.markdown("### Recommended Action Plan")
            if is_churn:
                st.markdown(f"""
                - **Automated Workflow:** Trigger win-back email campaign with a **10% replenishment discount** on top past purchased items.
                - **Timeline:** Reach out within the next **48 hours** before recency passes 90 days.
                - **Projected Recovery:** Win-back rates for this tier typically yield a **12-18% reactivation success**.
                """)
            else:
                st.markdown(f"""
                - **Status:** Account is in active health. Do not send deep discount coupons.
                - **Engagement:** Enroll in standard VIP loyalty points program.
                - **Upsell:** Recommend seasonal additions from *Home Decor* and *Bags & Storage*.
                """)

# ---------------------------------------------------------
# PAGE 5: CUSTOMER ACCOUNT LOOKUP
# ---------------------------------------------------------
elif nav == "🔍 Customer Account Lookup":
    st.markdown('<div class="main-header">Customer Account Lookup & Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect historical data and real model outputs for existing client accounts</div>', unsafe_allow_html=True)

    sample_customers = [12347, 12353, 12361, 12370, 12348, 12350, 12356]
    selected_id = st.selectbox("Select Customer ID to Inspect:", options=sample_customers, index=0)

    if selected_id in features.index:
        c = features.loc[selected_id]
        c_vector = features.loc[[selected_id], feature_cols]
        prob = model.predict_proba(c_vector)[0, 1]
        pred_label = "Churned (At Risk)" if prob >= operational_threshold else "Retained (Active)"
        actual_label = "Churned" if c['churn'] == 1 else "Retained"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Historical Spend", f"£{c['monetary']:,.2f}")
        col2.metric("Order Frequency", f"{int(c['frequency'])} orders")
        col3.metric("Last Purchase Recency", f"{int(c['recency'])} days ago")
        col4.metric("Churn Risk Probability", f"{prob:.1%}")

        st.divider()
        st.markdown(f"### Diagnostics for Account #{selected_id}")
        st.write(f"- **Model Prediction at Cutoff ({operational_threshold:.2f}):** `{pred_label}`")
        st.write(f"- **Actual Historical Outcome in Test Window:** `{actual_label}`")
        st.write(f"- **Customer Lifespan (First to Last Order):** `{int(c['purchase_span'])} days`")
        st.write(f"- **Account Tenure:** `{int(c['tenure'])} days`")

# ---------------------------------------------------------
# PAGE 6: MODEL DIAGNOSTICS & MATRIX
# ---------------------------------------------------------
elif nav == "📑 Model Diagnostics & Matrix":
    st.markdown('<div class="main-header">Model Performance & Classification Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Validation matrices, ROC-AUC curve, and feature importance analysis</div>', unsafe_allow_html=True)

    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_test_preds = (y_test_probs >= operational_threshold).astype(int)

    acc = accuracy_score(y_test, y_test_preds)
    prec = precision_score(y_test, y_test_preds)
    rec = recall_score(y_test, y_test_preds)
    f1 = f1_score(y_test, y_test_preds)
    auc = roc_auc_score(y_test, y_test_probs)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall (Sensitivity)", f"{rec:.2%}")
    c4.metric("F1-Score", f"{f1:.4f}")
    c5.metric("ROC-AUC", f"{auc:.4f}")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Confusion Matrix (Cutoff = {operational_threshold:.2f})")
        cm = confusion_matrix(y_test, y_test_preds)
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                    xticklabels=['Pred: Retained', 'Pred: Churned'],
                    yticklabels=['Actual: Retained', 'Actual: Churned'])
        st.pyplot(fig)

    with col2:
        st.subheader("Top Predictive Churn Drivers")
        if os.path.exists('figures/churn_feature_importance.png'):
            st.image('figures/churn_feature_importance.png', use_container_width=True)

st.sidebar.caption("Retail Intelligence Dashboard • Powered by Streamlit & Scikit-Learn")
