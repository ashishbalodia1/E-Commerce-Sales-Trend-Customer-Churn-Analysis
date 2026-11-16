import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_file, send_from_directory

app = Flask(__name__)
application = app
handler = app

# Load trained Random Forest model if available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_model.joblib')
model_bundle = None

if os.path.exists(MODEL_PATH):
    try:
        model_bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Notice: Model load warning: {e}")

FEATURE_COLS = [
    'recency', 'frequency', 'monetary', 'total_quantity', 'unique_items',
    'tenure', 'purchase_span', 'aov', 'avg_items_per_order', 'is_uk'
]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-Commerce Churn & Revenue Analytics</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #F8FAFC; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; color: #1E293B; }
        .hero { background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%); color: white; padding: 2.5rem 1rem; border-radius: 0 0 1rem 1rem; margin-bottom: 2rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .kpi-card { background: white; border-radius: 0.75rem; padding: 1.25rem; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: transform 0.2s; border-left: 4px solid #2563EB; }
        .kpi-card:hover { transform: translateY(-2px); }
        .kpi-val { font-size: 1.75rem; font-weight: 700; color: #0F172A; }
        .kpi-label { font-size: 0.85rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }
        .card-custom { background: white; border-radius: 0.75rem; border: none; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 1.5rem; }
        .nav-tabs .nav-link { color: #475569; font-weight: 600; border: none; padding: 0.75rem 1.25rem; }
        .nav-tabs .nav-link.active { color: #2563EB; border-bottom: 3px solid #2563EB; background: transparent; }
        .btn-predict { background: #2563EB; color: white; font-weight: 600; padding: 0.75rem 1.5rem; border-radius: 0.5rem; border: none; width: 100%; transition: background 0.2s; }
        .btn-predict:hover { background: #1D4ED8; color: white; }
        .result-box { border-radius: 0.5rem; padding: 1.25rem; display: none; }
        .chart-img { width: 100%; border-radius: 0.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
    </style>
</head>
<body>

<div class="hero text-center">
    <div class="container">
        <h1 class="fw-bold mb-2"><i class="fa-solid fa-chart-line me-2"></i>E-Commerce Customer Churn Analytics</h1>
        <p class="lead mb-0 opacity-90">Revenue Trends, Pareto 80/20 Distribution & Machine Learning Churn Predictor</p>
    </div>
</div>

<div class="container">
    <!-- Top KPI Cards -->
    <div class="row g-3 mb-4">
        <div class="col-md-3 col-6">
            <div class="kpi-card">
                <div class="kpi-label">Gross Revenue</div>
                <div class="kpi-val">£8.83M</div>
                <small class="text-success"><i class="fa-solid fa-arrow-trend-up"></i> +13.1% Peak MoM</small>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="kpi-card" style="border-left-color: #10B981;">
                <div class="kpi-label">Verified Orders</div>
                <div class="kpi-val">407,664</div>
                <small class="text-muted">Cleaned line-items</small>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="kpi-card" style="border-left-color: #8B5CF6;">
                <div class="kpi-label">Customer Base</div>
                <div class="kpi-val">4,312</div>
                <small class="text-muted">Top 10% = 59.9% spend</small>
            </div>
        </div>
        <div class="col-md-3 col-6">
            <div class="kpi-card" style="border-left-color: #EF4444;">
                <div class="kpi-label">Baseline Churn</div>
                <div class="kpi-val">39.9%</div>
                <small class="text-danger"><i class="fa-solid fa-triangle-exclamation"></i> Actionable Defection</small>
            </div>
        </div>
    </div>

    <!-- Navigation Tabs -->
    <ul class="nav nav-tabs mb-4" id="mainTabs" role="tablist">
        <li class="nav-item">
            <button class="nav-link active" data-bs-toggle="tab" data-bs-target="#predictor"><i class="fa-solid fa-calculator me-1"></i>Live Churn Predictor</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#trends"><i class="fa-solid fa-chart-area me-1"></i>Monthly Sales Trends</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#categories"><i class="fa-solid fa-boxes-stacked me-1"></i>Category Dynamics</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#pareto"><i class="fa-solid fa-users-viewfinder me-1"></i>Pareto 80/20 Rule</button>
        </li>
        <li class="nav-item">
            <button class="nav-link" data-bs-toggle="tab" data-bs-target="#diagnostics"><i class="fa-solid fa-brain me-1"></i>Model Matrix & ROC</button>
        </li>
        <li class="nav-item">
            <a class="nav-link" href="/notebook" target="_blank"><i class="fa-solid fa-book-open me-1"></i>Full Notebook HTML <i class="fa-solid fa-arrow-up-right-from-square ms-1"></i></a>
        </li>
    </ul>

    <!-- Tab Contents -->
    <div class="tab-content">
        <!-- 1. Live Predictor -->
        <div class="tab-pane fade show active" id="predictor">
            <div class="row">
                <div class="col-lg-7">
                    <div class="card card-custom p-4">
                        <h5 class="fw-bold mb-3"><i class="fa-solid fa-user-tag me-2 text-primary"></i>Customer Behavioral Inputs</h5>
                        <form id="churnForm">
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Total Spend (£)</label>
                                    <input type="number" class="form-control" id="monetary" value="850" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Total Orders (Frequency)</label>
                                    <input type="number" class="form-control" id="frequency" value="5" min="1" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Recency (Days since last purchase)</label>
                                    <input type="number" class="form-control" id="recency" value="75" min="1" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Customer Tenure (Total days)</label>
                                    <input type="number" class="form-control" id="tenure" value="210" min="1" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Purchase Span (Lifespan in days)</label>
                                    <input type="number" class="form-control" id="purchase_span" value="135" min="0" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Total Units Bought</label>
                                    <input type="number" class="form-control" id="total_quantity" value="320" min="1" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Unique SKUs Purchased</label>
                                    <input type="number" class="form-control" id="unique_items" value="45" min="1" required>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label fw-semibold">Market Region</label>
                                    <select class="form-select" id="is_uk">
                                        <option value="1" selected>United Kingdom (Domestic)</option>
                                        <option value="0">International / Overseas</option>
                                    </select>
                                </div>
                                <div class="col-12 mt-3">
                                    <label class="form-label fw-semibold">Alert Cutoff Threshold: <span id="threshVal" class="text-primary fw-bold">0.35</span></label>
                                    <input type="range" class="form-range" id="threshold" min="0.2" max="0.7" step="0.05" value="0.35" oninput="document.getElementById('threshVal').innerText = this.value">
                                    <small class="text-muted">Recommended: 0.35 flags ~82.5% of churners for early retention workflows.</small>
                                </div>
                                <div class="col-12 mt-4">
                                    <button type="submit" class="btn btn-predict"><i class="fa-solid fa-bolt me-2"></i>Predict Customer Churn Probability</button>
                                </div>
                            </div>
                        </form>
                    </div>
                </div>

                <div class="col-lg-5">
                    <div class="card card-custom p-4">
                        <h5 class="fw-bold mb-3"><i class="fa-solid fa-square-poll-vertical me-2 text-primary"></i>Inference Output</h5>
                        <div id="placeholderResult" class="text-center py-5 text-muted">
                            <i class="fa-solid fa-chart-pie fa-3x mb-3 text-secondary opacity-50"></i>
                            <p>Fill the customer profile form and click <strong>Predict</strong> to calculate the real-time defection probability.</p>
                        </div>
                        <div id="resultBox" class="result-box">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <span class="badge px-3 py-2 fs-6" id="riskBadge">LOW RISK</span>
                                <span class="text-muted fw-semibold">Cutoff: <span id="resThreshold">0.35</span></span>
                            </div>
                            <div class="mb-3">
                                <div class="d-flex justify-content-between fw-bold mb-1">
                                    <span>Churn Risk:</span>
                                    <span id="probText">0.0%</span>
                                </div>
                                <div class="progress" style="height: 12px;">
                                    <div id="probBar" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="card bg-light p-3 border-0 mt-3">
                                <h6 class="fw-bold mb-1" id="actionTitle">Recommended Action</h6>
                                <p class="small text-muted mb-0" id="actionDesc">-</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 2. Sales Trends -->
        <div class="tab-pane fade" id="trends">
            <div class="card card-custom p-4">
                <h5 class="fw-bold mb-3">Monthly Revenue Trajectory & MoM Growth</h5>
                <img src="/figures/monthly_sales_trend.png" alt="Monthly Trend" class="chart-img mb-3">
                <div class="alert alert-info mb-0">
                    <strong>Key Seasonality Takeaways:</strong>
                    November was the highest revenue month (£1,172,336, +13.1% MoM), driven by Q4 holiday procurement. February was the lowest at £506,371. Sales expanded +131% between Feb and Nov.
                </div>
            </div>
        </div>

        <!-- 3. Categories -->
        <div class="tab-pane fade" id="categories">
            <div class="card card-custom p-4">
                <h5 class="fw-bold mb-3">Merchandise Performance by Product Category</h5>
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <img src="/figures/category_revenue_comparison.png" alt="Category Revenue" class="chart-img">
                    </div>
                    <div class="col-md-6 mb-3">
                        <img src="/figures/category_monthly_pivot_heatmap.png" alt="Category Heatmap" class="chart-img">
                    </div>
                </div>
                <p class="text-muted mb-0"><strong>Home Decor & Living (£2.43M)</strong> and <strong>Bags & Storage (£1.82M)</strong> represent <strong>48.1%</strong> of total company sales.</p>
            </div>
        </div>

        <!-- 4. Pareto -->
        <div class="tab-pane fade" id="pareto">
            <div class="card card-custom p-4">
                <h5 class="fw-bold mb-3">Pareto (80/20) Customer Spend Concentration</h5>
                <img src="/figures/pareto_customer_spend.png" alt="Pareto Spend" class="chart-img mb-3">
                <div class="row text-center g-3">
                    <div class="col-md-4">
                        <div class="p-3 bg-light rounded">
                            <h6 class="fw-bold text-primary mb-1">Top 10% Spenders (432 clients)</h6>
                            <h4 class="fw-bold mb-0">£5.29M (59.9%)</h4>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-3 bg-light rounded">
                            <h6 class="fw-bold text-success mb-1">Top 20% Spenders (863 clients)</h6>
                            <h4 class="fw-bold mb-0">£6.49M (73.5%)</h4>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="p-3 bg-light rounded">
                            <h6 class="fw-bold text-secondary mb-1">Bottom 80% (3,449 clients)</h6>
                            <h4 class="fw-bold mb-0">£2.34M (26.5%)</h4>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 5. Model Diagnostics -->
        <div class="tab-pane fade" id="diagnostics">
            <div class="card card-custom p-4">
                <h5 class="fw-bold mb-3">Model Validation Matrices & Feature Importances</h5>
                <div class="row">
                    <div class="col-md-4 mb-3">
                        <img src="/figures/churn_confusion_matrix.png" alt="Confusion Matrix" class="chart-img">
                    </div>
                    <div class="col-md-4 mb-3">
                        <img src="/figures/churn_roc_curve.png" alt="ROC Curve" class="chart-img">
                    </div>
                    <div class="col-md-4 mb-3">
                        <img src="/figures/churn_feature_importance.png" alt="Feature Importance" class="chart-img">
                    </div>
                </div>
                <div class="alert alert-secondary mb-0">
                    <strong>Model Architecture:</strong> Fine-Tuned Random Forest (180 trees, max_depth=6). Test Accuracy: 67.75% | ROC-AUC: 0.7333 | Recall at 0.35 cutoff: 82.53%.
                </div>
            </div>
        </div>
    </div>
</div>

<footer class="text-center py-4 text-muted small mt-5 border-top">
    E-Commerce Customer Churn Analysis Platform • Deployed on Vercel • Author: Ashis Balodia
</footer>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
document.getElementById('churnForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const payload = {
        monetary: parseFloat(document.getElementById('monetary').value) || 0,
        frequency: parseInt(document.getElementById('frequency').value) || 1,
        recency: parseInt(document.getElementById('recency').value) || 0,
        tenure: parseInt(document.getElementById('tenure').value) || 1,
        purchase_span: parseInt(document.getElementById('purchase_span').value) || 0,
        total_quantity: parseInt(document.getElementById('total_quantity').value) || 1,
        unique_items: parseInt(document.getElementById('unique_items').value) || 1,
        is_uk: parseInt(document.getElementById('is_uk').value) || 1,
        threshold: parseFloat(document.getElementById('threshold').value) || 0.35
    };

    try {
        // Try /api/predict first, fallback to /predict or current origin
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const contentType = res.headers.get("content-type");
        if (!contentType || !contentType.includes("application/json")) {
            const rawText = await res.text();
            throw new Error("Server returned non-JSON response (status: " + res.status + ")");
        }

        const data = await res.json();

        document.getElementById('placeholderResult').style.display = 'none';
        const box = document.getElementById('resultBox');
        box.style.display = 'block';

        const probPct = (data.churn_probability * 100).toFixed(1) + '%';
        document.getElementById('probText').innerText = probPct;
        document.getElementById('resThreshold').innerText = data.threshold.toFixed(2);

        const bar = document.getElementById('probBar');
        bar.style.width = probPct;

        const badge = document.getElementById('riskBadge');
        if (data.risk_level === 'High') {
            badge.className = 'badge px-3 py-2 fs-6 bg-danger';
            badge.innerText = 'HIGH CHURN RISK';
            bar.className = 'progress-bar bg-danger';
            box.style.border = '2px solid #EF4444';
        } else if (data.risk_level === 'Moderate') {
            badge.className = 'badge px-3 py-2 fs-6 bg-warning text-dark';
            badge.innerText = 'MODERATE AT-RISK';
            bar.className = 'progress-bar bg-warning';
            box.style.border = '2px solid #F59E0B';
        } else {
            badge.className = 'badge px-3 py-2 fs-6 bg-success';
            badge.innerText = 'HEALTHY ACCOUNT';
            bar.className = 'progress-bar bg-success';
            box.style.border = '2px solid #10B981';
        }

        document.getElementById('actionTitle').innerText = data.prediction === 'Churned' ? 'Retention Workflow Triggered' : 'Healthy Engagement Plan';
        document.getElementById('actionDesc').innerText = data.recommendation;

    } catch (err) {
        alert('Prediction request failed: ' + err.message);
    }
});
</script>
</body>
</html>
"""

def serve_home():
    return render_template_string(HTML_TEMPLATE)

def serve_notebook():
    nb_path = os.path.join(BASE_DIR, 'customer_churn_analysis.html')
    if os.path.exists(nb_path):
        return send_file(nb_path)
    return "<h3>customer_churn_analysis.html not found.</h3>", 404

def serve_figure(filename):
    figures_dir = os.path.join(BASE_DIR, 'figures')
    return send_from_directory(figures_dir, filename)

def handle_predict():
    data = request.get_json(force=True, silent=True) or {}
    
    try:
        recency = float(data.get('recency', 60))
        frequency = max(1.0, float(data.get('frequency', 3)))
        monetary = float(data.get('monetary', 500))
        total_quantity = max(1.0, float(data.get('total_quantity', 100)))
        unique_items = max(1.0, float(data.get('unique_items', 15)))
        tenure = max(1.0, float(data.get('tenure', 180)))
        purchase_span = max(0.0, float(data.get('purchase_span', 90)))
        is_uk = int(data.get('is_uk', 1))
        threshold = float(data.get('threshold', 0.35))
    except Exception:
        recency, frequency, monetary, total_quantity, unique_items = 60.0, 3.0, 500.0, 100.0, 15.0
        tenure, purchase_span, is_uk, threshold = 180.0, 90.0, 1, 0.35

    aov = monetary / frequency
    avg_items_per_order = total_quantity / frequency

    features_vector = np.array([[
        recency, frequency, monetary, total_quantity, unique_items,
        tenure, purchase_span, aov, avg_items_per_order, is_uk
    ]])

    if model_bundle and 'model' in model_bundle:
        prob = float(model_bundle['model'].predict_proba(features_vector)[0, 1])
    else:
        dormancy_score = min(1.0, recency / 150.0)
        span_hazard = 1.0 if purchase_span == 0 else max(0.0, 1.0 - (purchase_span / tenure))
        prob = float(np.clip(0.45 * dormancy_score + 0.35 * span_hazard + 0.20 * (1.0 / (frequency + 1)), 0.05, 0.95))

    is_churn = prob >= threshold
    if prob >= 0.50:
        risk_level = "High"
        recommendation = "Immediate account manager outreach within 48h. Provide a 15% VIP retention rebate on previous ordered items."
    elif prob >= threshold:
        risk_level = "Moderate"
        recommendation = "Automated win-back email workflow with a 10% replenishment coupon on top past categories."
    else:
        risk_level = "Low"
        recommendation = "Account is healthy and active. Enroll in standard VIP loyalty points and cross-sell Home Decor additions."

    return jsonify({
        "churn_probability": round(prob, 4),
        "prediction": "Churned" if is_churn else "Retained",
        "risk_level": risk_level,
        "threshold": threshold,
        "recommendation": recommendation
    })

# -------------------------------------------------------------
# EXPLICIT FLASK ROUTES (Standard Matchers)
# -------------------------------------------------------------
@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def api_predict_route():
    return handle_predict()

@app.route('/notebook', methods=['GET'])
@app.route('/customer_churn_analysis.html', methods=['GET'])
def notebook_route():
    return serve_notebook()

@app.route('/figures/<path:filename>', methods=['GET'])
def figures_route(filename):
    return serve_figure(filename)

# -------------------------------------------------------------
# CATCH-ALL ROUTER (Handles rewrites & edge paths)
# -------------------------------------------------------------
@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def universal_handler(path):
    # Rule 1: ANY POST request in this application is ALWAYS a prediction call!
    if request.method == 'POST':
        return handle_predict()

    p_clean = path.strip('/').lower()

    # Rule 2: Figures
    if p_clean.startswith('figures/'):
        fname = path.split('/')[-1]
        return serve_figure(fname)

    # Rule 3: Notebook
    if 'notebook' in p_clean or 'customer_churn_analysis.html' in p_clean:
        return serve_notebook()

    # Rule 4: Everything else is Home
    return serve_home()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
