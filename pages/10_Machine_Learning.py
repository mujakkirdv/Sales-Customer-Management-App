"""
pages/10_Machine_Learning.py — ML on your sales data.
  • Sales forecasting (Linear Regression + Random Forest)
  • Customer segmentation (KMeans)
  • Outstanding prediction
  • Churn probability (classification)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score

from styles import inject_theme, hero, toast_success
from utils import get_df

st.set_page_config(page_title="Machine Learning", page_icon="🤖", layout="wide")
inject_theme()
hero("Machine Learning", "Forecast sales, segment customers, predict churn.", icon="🤖")

df = get_df()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Sales Forecast", "👥 Segmentation", "💰 Outstanding Prediction", "⚠️ Churn Risk"]
)

# =====================================================================
# TAB 1 — SALES FORECAST
# =====================================================================
with tab1:
    st.markdown("### 📈 Monthly Sales Forecast")

    monthly = (df.set_index('Date')['Sales Amount']
                 .resample('M').sum().reset_index())
    monthly['t'] = np.arange(len(monthly))
    monthly['month'] = monthly['Date'].dt.month
    monthly['year']  = monthly['Date'].dt.year

    if len(monthly) < 4:
        st.warning("Need at least 4 months of data for forecasting.")
    else:
        # Features
        X = monthly[['t', 'month', 'year']].values
        y = monthly['Sales Amount'].values

        # Split
        split = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        r2  = r2_score(y_te, y_pred) if len(y_te) > 1 else float('nan')
        mae = mean_absolute_error(y_te, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score",   f"{r2:.3f}")
        c2.metric("MAE",        f"৳ {mae:,.0f}")
        c3.metric("Test Months", len(y_te))

        # Future forecast
        n_future = st.slider("Months to forecast", 1, 12, 6)
        last_t   = monthly['t'].max()
        last_date = monthly['Date'].max()
        future = []
        for i in range(1, n_future + 1):
            fd = last_date + pd.DateOffset(months=i)
            future.append([last_t + i, fd.month, fd.year])
        future = np.array(future)
        y_future = model.predict(future)
        future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, n_future + 1)]

        # ---- Chart ----
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly['Date'], y=monthly['Sales Amount'],
                                 mode='lines+markers', name='Actual',
                                 line=dict(color='#3b82f6', width=3)))
        fig.add_trace(go.Scatter(x=monthly['Date'].iloc[split:], y=y_pred,
                                 mode='lines+markers', name='Predicted (test)',
                                 line=dict(color='#f59e0b', width=3, dash='dot')))
        fig.add_trace(go.Scatter(x=future_dates, y=y_future,
                                 mode='lines+markers', name='Forecast',
                                 line=dict(color='#10b981', width=3, dash='dash')))
        fig.update_layout(title="Sales Forecast", hovermode='x unified', height=450)
        st.plotly_chart(fig, use_container_width=True)

        # ---- Feature importance ----
        imp = pd.DataFrame({
            'Feature': ['Time index', 'Month', 'Year'],
            'Importance': model.feature_importances_,
        }).sort_values('Importance', ascending=False)
        st.plotly_chart(px.bar(imp, x='Importance', y='Feature',
                               orientation='h', title="Feature Importance",
                               color='Importance', color_continuous_scale='viridis'),
                        use_container_width=True)

        with st.expander("🔮 Forecast table"):
            st.dataframe(pd.DataFrame({
                'Month': [d.strftime('%b %Y') for d in future_dates],
                'Forecast (৳)': y_future.round(0).astype(int),
            }), use_container_width=True)

# =====================================================================
# TAB 2 — CUSTOMER SEGMENTATION
# =====================================================================
with tab2:
    st.markdown("### 👥 Customer Segmentation (K-Means)")

    cust = (df.groupby('Customer Name')
              .agg(Sales=('Sales Amount', 'sum'),
                   Credited=('Credited Amount', 'sum'),
                   Outstanding=('Outstanding', 'sum'),
                   Returns=('Sales Return', 'sum'),
                   Invoices=('Invoice No', 'count'),
                   Avg_Invoice=('Invoice Value', 'mean'))
              .reset_index())

    # RFM-ish features
    features = ['Sales', 'Credited', 'Outstanding', 'Invoices', 'Avg_Invoice']
    X_seg = cust[features].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seg)

    k = st.slider("Number of segments (k)", 2, 8, 4)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cust['Segment'] = km.fit_predict(X_scaled)

    # Segment profiles
    prof = cust.groupby('Segment')[features].mean().round(0)
    prof['Customers'] = cust.groupby('Segment').size()
    st.markdown("#### Segment profiles")
    st.dataframe(prof, use_container_width=True)

    # 3D scatter
    fig = px.scatter_3d(cust.sample(min(2000, len(cust)), random_state=1),
                        x='Sales', y='Credited', z='Outstanding',
                        color=cust['Segment'].astype(str),
                        hover_name='Customer Name',
                        title="Customer Segments (3D)",
                        opacity=0.7)
    st.plotly_chart(fig, use_container_width=True)

    # Segment sizes
    st.plotly_chart(px.pie(cust, names='Segment', title="Segment Distribution",
                           hole=0.4), use_container_width=True)

    with st.expander("📋 Full customer → segment mapping"):
        st.dataframe(cust.sort_values('Segment'), use_container_width=True)

# =====================================================================
# TAB 3 — OUTSTANDING PREDICTION
# =====================================================================
with tab3:
    st.markdown("### 💰 Predicting Outstanding from transaction attributes")

    feats = ['Invoice Value', 'Discount', 'Sales Amount', 'Sales VAT',
             'Sales Return', 'Credited Amount']
    data = df[feats + ['Outstanding']].dropna()

    X = data[feats].values
    y = data['Outstanding'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2_score(y_te, pred):.3f}")
    c2.metric("MAE",      f"৳ {mean_absolute_error(y_te, pred):,.0f}")
    c3.metric("Samples",  len(y_te))

    # Predicted vs actual
    fig = px.scatter(x=y_te, y=pred, opacity=0.4,
                     labels={'x': 'Actual Outstanding', 'y': 'Predicted Outstanding'},
                     title="Predicted vs Actual")
    lim = max(abs(y_te.max()), abs(pred.max()))
    fig.add_shape(type='line', x0=-lim, y0=-lim, x1=lim, y1=lim,
                  line=dict(color='red', dash='dash'))
    st.plotly_chart(fig, use_container_width=True)

    imp = pd.DataFrame({'Feature': feats,
                        'Importance': model.feature_importances_}
                       ).sort_values('Importance', ascending=False)
    st.plotly_chart(px.bar(imp, x='Importance', y='Feature', orientation='h',
                           title="Feature Importance", color='Importance',
                           color_continuous_scale='greens'),
                    use_container_width=True)

# =====================================================================
# TAB 4 — CHURN RISK
# =====================================================================
with tab4:
    st.markdown("### ⚠️ Churn Risk Prediction")
    st.caption("A customer is labelled **churned** if their last transaction "
               "is older than the cutoff below.")

    cutoff_days = st.slider("Churn cutoff (days since last purchase)", 30, 365, 90)
    ref_date = df['Date'].max()
    cutoff = ref_date - pd.Timedelta(days=cutoff_days)

    cust = (df.groupby('Customer Name')
              .agg(Sales=('Sales Amount', 'sum'),
                   Credited=('Credited Amount', 'sum'),
                   Outstanding=('Outstanding', 'sum'),
                   Invoices=('Invoice No', 'count'),
                   Avg_Invoice=('Invoice Value', 'mean'),
                   Last_Date=('Date', 'max'),
                   First_Date=('Date', 'min'))
              .reset_index())
    cust['Churned'] = (cust['Last_Date'] < cutoff).astype(int)
    cust['Tenure_Days'] = (cust['Last_Date'] - cust['First_Date']).dt.days

    st.info(f"Churned: **{cust['Churned'].sum()}** / {len(cust)} customers "
            f"({cust['Churned'].mean()*100:.1f}%)")

    feats = ['Sales', 'Credited', 'Outstanding', 'Invoices', 'Avg_Invoice', 'Tenure_Days']
    X = cust[feats].fillna(0).values
    y = cust['Churned'].values

    if len(np.unique(y)) < 2:
        st.warning("All customers fall in the same class — adjust the cutoff.")
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                  random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy_score(y_te, pred):.2%}")
        c2.metric("Churn class",  f"{(y.mean()*100):.1f}%")
        c3.metric("Test size",    len(y_te))

        cust['Churn_Prob'] = clf.predict_proba(cust[feats].fillna(0))[:, 1]
        top_risk = cust.nlargest(15, 'Churn_Prob')[
            ['Customer Name', 'Churn_Prob', 'Sales', 'Last_Date']
        ]
        st.markdown("#### 🚨 Top 15 churn-risk customers")
        st.dataframe(top_risk.style.format({'Churn_Prob': '{:.1%}'}),
                     use_container_width=True)

        st.plotly_chart(px.histogram(cust, x='Churn_Prob', nbins=25,
                                     title="Churn Probability Distribution",
                                     color_discrete_sequence=['#ef4444']),
                        use_container_width=True)

        st.download_button(
            "📥 Download full risk report (CSV)",
            cust.to_csv(index=False).encode('utf-8'),
            file_name="churn_risk.csv", mime="text/csv",
        )
        toast_success("ML models trained successfully")