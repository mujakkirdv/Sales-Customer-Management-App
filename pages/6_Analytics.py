"""
pages/6_Analytics.py — Advanced Analytics
Trends · Rankings · Correlation · Zone & Channel · Native charts
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from styles import inject_theme, hero, chips, toast_success
from utils import get_df

# =====================================================================
# PAGE CONFIG & THEME
# =====================================================================
st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")
inject_theme()
hero("Advanced Analytics",
     "Trends, rankings, correlations, and channel performance across your sales data.",
     icon="📈")

df = get_df()

# =====================================================================
# SIDEBAR FILTERS
# =====================================================================
with st.sidebar:
    st.markdown("### 🎛️ Analytics Filters")

    with st.expander("📅 Date & Entities", expanded=True):
        date_range = st.date_input(
            "Date Range",
            [df['Date'].min().date(), df['Date'].max().date()],
        )
        execs = st.multiselect("Executives",
                               df['Executive'].unique(),
                               df['Executive'].unique())
        zones = st.multiselect("Sales Zone",
                               df['Sales Zone'].unique(),
                               df['Sales Zone'].unique())

    with st.expander("🏷️ Segments"):
        ctypes = st.multiselect("Customer Type",
                                df['Customer Type'].unique(),
                                df['Customer Type'].unique())
        channels = st.multiselect("Sales Channel",
                                  df['Sales Channel'].unique(),
                                  df['Sales Channel'].unique())
        banks = st.multiselect("Bank",
                               df['Bank Name'].unique(),
                               df['Bank Name'].unique())

    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

# =====================================================================
# APPLY FILTERS
# =====================================================================
fdf = df[
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Executive'].isin(execs)) &
    (df['Sales Zone'].isin(zones)) &
    (df['Customer Type'].isin(ctypes)) &
    (df['Sales Channel'].isin(channels)) &
    (df['Bank Name'].isin(banks))
]

if fdf.empty:
    st.warning("⚠️ No records match the current filters.")
    st.stop()

# =====================================================================
# HEADER CHIPS
# =====================================================================
chips([
    (f"📦 {len(fdf):,} records", ""),
    (f"📅 {date_range[0]:%d/%m/%y} → {date_range[1]:%d/%m/%y}", ""),
    (f"📈 {(len(fdf)/len(df)*100):.1f}% of dataset",
     "green" if len(fdf) > len(df) * 0.5 else "amber"),
])

# =====================================================================
# TOP KPI STRIP
# =====================================================================
total_sales    = fdf['Sales Amount'].sum()
total_credited = fdf['Credited Amount'].sum()
total_out      = fdf['Outstanding'].sum()
total_profit   = fdf['Profit'].sum()
total_returns  = fdf['Sales Return'].sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Total Sales",     f"৳ {total_sales:,.0f}")
k2.metric("💳 Credited",        f"৳ {total_credited:,.0f}")
k3.metric("📊 Outstanding",     f"৳ {total_out:,.0f}")
k4.metric("📈 Profit",          f"৳ {total_profit:,.0f}")
k5.metric("↩️ Returns",         f"৳ {total_returns:,.0f}")

# =====================================================================
# TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Trend Analysis",
    "🏆 Rankings",
    "🧩 Correlation",
    "🌍 Zone & Channel",
    "⚡ Native Charts",
])

# =====================================================================
# TAB 1 — TREND ANALYSIS
# =====================================================================
with tab1:
    st.markdown("### 📊 Monthly Sales / Credited / Outstanding")

    monthly = (fdf.groupby('Month Name')
               .agg(Sales=('Sales Amount', 'sum'),
                    Credited=('Credited Amount', 'sum'),
                    Outstanding=('Outstanding', 'sum'),
                    Returns=('Sales Return', 'sum'),
                    Invoices=('Invoice No', 'count'))
               .reset_index())

    # Sort by actual date to keep chronological order
    month_order = (fdf.drop_duplicates('Month Name')
                   .sort_values('Date')['Month Name'].tolist())
    monthly['Month Name'] = pd.Categorical(monthly['Month Name'],
                                           categories=month_order, ordered=True)
    monthly = monthly.sort_values('Month Name')

    # Combined bar + line chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=monthly['Month Name'], y=monthly['Sales'],
                         name='Sales', marker_color='#3b82f6',
                         hovertemplate='%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Bar(x=monthly['Month Name'], y=monthly['Credited'],
                         name='Credited', marker_color='#10b981',
                         hovertemplate='%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=monthly['Month Name'], y=monthly['Outstanding'],
                             name='Outstanding', mode='lines+markers',
                             line=dict(color='#ef4444', width=3),
                             marker=dict(size=10),
                             hovertemplate='%{y:,.0f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=monthly['Month Name'], y=monthly['Returns'],
                             name='Returns', mode='lines+markers',
                             line=dict(color='#f59e0b', width=2, dash='dot'),
                             hovertemplate='%{y:,.0f}<extra></extra>'))
    fig.update_layout(
        title="Monthly Financial Overview",
        barmode='group',
        hovermode='x unified',
        height=460,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Monthly summary table", expanded=False):
        st.dataframe(
            monthly.style.format({
                'Sales': '৳{:,.0f}', 'Credited': '৳{:,.0f}',
                'Outstanding': '৳{:,.0f}', 'Returns': '৳{:,.0f}',
            }),
            use_container_width=True,
        )
        st.download_button(
            "📥 Download monthly trend (CSV)",
            monthly.to_csv(index=False).encode('utf-8'),
            file_name="monthly_trend.csv", mime="text/csv",
        )

    # ---------- Growth rate ----------
    st.markdown("#### 📈 Month-over-Month Growth")
    if len(monthly) >= 2:
        monthly['MoM %'] = monthly['Sales'].pct_change() * 100
        growth = monthly.dropna(subset=['MoM %'])

        fig_growth = px.bar(
            growth, x='Month Name', y='MoM %',
            color='MoM %',
            color_continuous_scale=['#ef4444', '#94a3b8', '#10b981'],
            color_continuous_midpoint=0,
            title="Month-over-Month Sales Growth (%)",
        )
        fig_growth.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_growth, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Best month",
                  f"{growth.loc[growth['MoM %'].idxmax(), 'Month Name']}",
                  f"+{growth['MoM %'].max():.1f}%")
        c2.metric("Worst month",
                  f"{growth.loc[growth['MoM %'].idxmin(), 'Month Name']}",
                  f"{growth['MoM %'].min():.1f}%")
        c3.metric("Avg MoM growth", f"{growth['MoM %'].mean():+.1f}%")
    else:
        st.info("Need at least 2 months to compute growth.")

    # ---------- Rolling average ----------
    with st.expander("📉 Rolling 3-month average", expanded=False):
        daily = (fdf.set_index('Date')['Sales Amount']
                 .resample('D').sum().reset_index())
        daily['Rolling_7'] = daily['Sales Amount'].rolling(7, min_periods=1).mean()
        daily['Rolling_30'] = daily['Sales Amount'].rolling(30, min_periods=1).mean()

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(x=daily['Date'], y=daily['Sales Amount'],
                                   name='Daily', mode='lines',
                                   line=dict(color='#cbd5e1', width=1)))
        fig_r.add_trace(go.Scatter(x=daily['Date'], y=daily['Rolling_7'],
                                   name='7-day avg', mode='lines',
                                   line=dict(color='#3b82f6', width=3)))
        fig_r.add_trace(go.Scatter(x=daily['Date'], y=daily['Rolling_30'],
                                   name='30-day avg', mode='lines',
                                   line=dict(color='#ef4444', width=3)))
        fig_r.update_layout(title="Daily sales with rolling averages",
                            hovermode='x unified', height=400)
        st.plotly_chart(fig_r, use_container_width=True)

# =====================================================================
# TAB 2 — RANKINGS
# =====================================================================
with tab2:
    st.markdown("### 🏆 Top Performers")

    n_top = st.slider("Show top N", 5, 30, 10, key="rank_n")

    c1, c2 = st.columns(2)

    with c1:
        top_exec = (fdf.groupby('Executive')
                    .agg(Sales=('Sales Amount', 'sum'),
                         Credited=('Credited Amount', 'sum'),
                         Outstanding=('Outstanding', 'sum'),
                         Customers=('Customer Name', 'nunique'))
                    .reset_index()
                    .sort_values('Sales', ascending=False)
                    .head(n_top))

        fig = px.bar(top_exec, x='Sales', y='Executive', orientation='h',
                     title=f"Top {n_top} Executives by Sales",
                     color='Sales', color_continuous_scale='plasma',
                     hover_data=['Credited', 'Outstanding', 'Customers'])
        fig.update_layout(height=480, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_cust = (fdf.groupby('Customer Name')
                    .agg(Sales=('Sales Amount', 'sum'),
                         Credited=('Credited Amount', 'sum'),
                         Outstanding=('Outstanding', 'sum'),
                         Invoices=('Invoice No', 'count'))
                    .reset_index()
                    .sort_values('Sales', ascending=False)
                    .head(n_top))

        fig = px.bar(top_cust, x='Sales', y='Customer Name', orientation='h',
                     title=f"Top {n_top} Customers by Sales",
                     color='Sales', color_continuous_scale='viridis',
                     hover_data=['Credited', 'Outstanding', 'Invoices'])
        fig.update_layout(height=480, coloraxis_showscale=False,
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    # ---------- Bar race style: top 3 stacked ----------
    with st.expander("🥇 Top performers comparison table", expanded=False):
        combined = top_exec.merge(
            top_cust, how='outer', left_on='Executive', right_on='Customer Name'
        )
        st.dataframe(
            top_exec.style.format({
                'Sales': '৳{:,.0f}', 'Credited': '৳{:,.0f}',
                'Outstanding': '৳{:,.0f}',
            }).background_gradient(subset=['Sales'], cmap='Blues'),
            use_container_width=True,
        )
        st.dataframe(
            top_cust.style.format({
                'Sales': '৳{:,.0f}', 'Credited': '৳{:,.0f}',
                'Outstanding': '৳{:,.0f}',
            }).background_gradient(subset=['Sales'], cmap='Greens'),
            use_container_width=True,
        )

    # ---------- Executive collection rate ranking ----------
    st.markdown("#### 🎯 Collection Rate by Executive")
    exec_rate = (fdf.groupby('Executive')
                 .agg(Sales=('Sales Amount', 'sum'),
                      Credited=('Credited Amount', 'sum'))
                 .reset_index())
    exec_rate['Collection %'] = np.where(
        exec_rate['Sales'] > 0,
        exec_rate['Credited'] / exec_rate['Sales'] * 100, 0
    )
    exec_rate = exec_rate.sort_values('Collection %', ascending=False)

    fig_rate = px.bar(exec_rate, x='Executive', y='Collection %',
                      title="Collection Rate (%)",
                      color='Collection %',
                      color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
                      text='Collection %')
    fig_rate.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig_rate.update_layout(height=400, coloraxis_showscale=False,
                           xaxis_tickangle=-45)
    st.plotly_chart(fig_rate, use_container_width=True)

# =====================================================================
# TAB 3 — CORRELATION
# =====================================================================
with tab3:
    st.markdown("### 🧩 Correlation & Relationships")

    num_cols = ['Invoice Value', 'Discount', 'Sales Amount', 'Sales VAT',
                'Sales Return', 'Credited Amount', 'Outstanding', 'Profit']

    corr = fdf[num_cols].corr().round(3)

    # ---------- Heatmap ----------
    fig = px.imshow(corr,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix (Pearson)")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Strongest correlations ----------
    corr_pairs = (corr.where(~np.eye(len(corr), dtype=bool))
                  .stack().reset_index())
    corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    corr_pairs['abs'] = corr_pairs['Correlation'].abs()
    # remove duplicate pairs
    corr_pairs = corr_pairs.drop_duplicates(subset=['abs'], keep='first')
    corr_pairs = corr_pairs.sort_values('abs', ascending=False).head(10)

    st.markdown("#### 🔗 Top 10 strongest correlations")
    st.dataframe(
        corr_pairs[['Feature 1', 'Feature 2', 'Correlation']]
        .style.format({'Correlation': '{:+.3f}'})
        .background_gradient(subset=['Correlation'], cmap='RdBu_r'),
        use_container_width=True,
    )

    # ---------- Scatter matrix ----------
    with st.expander("🔬 Scatter matrix (sampled)", expanded=False):
        sample = fdf.sample(min(2000, len(fdf)), random_state=1)
        dims = ['Sales Amount', 'Credited Amount', 'Outstanding',
                'Discount', 'Profit']
        fig_sm = px.scatter_matrix(
            sample, dimensions=dims,
            color='Customer Type',
            title="Scatter Matrix (sampled)",
            opacity=0.5,
        )
        fig_sm.update_traces(diagonal_visible=False, showupperhalf=False)
        fig_sm.update_layout(height=700)
        st.plotly_chart(fig_sm, use_container_width=True)

    # ---------- Discount vs Sales bubble ----------
    with st.expander("💬 Discount vs Sales Amount", expanded=False):
        sample = fdf.sample(min(3000, len(fdf)), random_state=2)
        fig_b = px.scatter(
            sample,
            x='Discount', y='Sales Amount',
            size='Invoice Value', color='Customer Type',
            hover_name='Customer Name',
            title="Discount vs Sales Amount (bubble size = Invoice Value)",
            opacity=0.7,
        )
        fig_b.update_layout(height=520)
        st.plotly_chart(fig_b, use_container_width=True)

# =====================================================================
# TAB 4 — ZONE & CHANNEL
# =====================================================================
with tab4:
    st.markdown("### 🌍 Geographic & Channel Performance")

    c1, c2 = st.columns(2)

    with c1:
        z = (fdf.groupby('Sales Zone')
             .agg(Sales=('Sales Amount', 'sum'),
                  Credited=('Credited Amount', 'sum'),
                  Outstanding=('Outstanding', 'sum'),
                  Customers=('Customer Name', 'nunique'))
             .reset_index())

        fig_z = px.bar(z, x='Sales Zone',
                       y=['Sales', 'Credited', 'Outstanding'],
                       barmode='group',
                       title="Zone Performance",
                       color_discrete_sequence=['#3b82f6', '#10b981', '#ef4444'])
        fig_z.update_layout(height=420, legend_title=None)
        st.plotly_chart(fig_z, use_container_width=True)

    with c2:
        ch = (fdf.groupby('Sales Channel')
              .agg(Sales=('Sales Amount', 'sum'),
                   Credited=('Credited Amount', 'sum'),
                   Outstanding=('Outstanding', 'sum'))
              .reset_index())

        fig_ch = px.bar(ch, x='Sales Channel',
                        y=['Sales', 'Credited', 'Outstanding'],
                        barmode='group',
                        title="Channel Performance",
                        color_discrete_sequence=['#6366f1', '#06b6d4', '#f59e0b'])
        fig_ch.update_layout(height=420, legend_title=None)
        st.plotly_chart(fig_ch, use_container_width=True)

    # ---------- Treemap Zone → Channel ----------
    with st.expander("🌳 Zone × Channel treemap", expanded=True):
        tc = (fdf.groupby(['Sales Zone', 'Sales Channel'])['Sales Amount']
              .sum().reset_index())
        fig_tree = px.treemap(
            tc, path=['Sales Zone', 'Sales Channel'],
            values='Sales Amount',
            title="Sales Volume: Zone → Channel",
            color='Sales Amount',
            color_continuous_scale='Blues',
        )
        fig_tree.update_layout(height=520)
        st.plotly_chart(fig_tree, use_container_width=True)

    # ---------- Sunburst Zone → Customer Type ----------
    with st.expander("☀️ Zone × Customer Type sunburst"):
        sb = (fdf.groupby(['Sales Zone', 'Customer Type'])['Sales Amount']
              .sum().reset_index())
        fig_sun = px.sunburst(
            sb, path=['Sales Zone', 'Customer Type'],
            values='Sales Amount',
            title="Sales by Zone & Customer Type",
            color='Sales Amount',
            color_continuous_scale='Viridis',
        )
        fig_sun.update_layout(height=560)
        st.plotly_chart(fig_sun, use_container_width=True)

    # ---------- Payment method breakdown ----------
    with st.expander("💳 Payment method mix"):
        pm = (fdf.groupby('Payment Method')['Credited Amount']
              .sum().reset_index())
        fig_pm = px.pie(pm, values='Credited Amount', names='Payment Method',
                        hole=0.45, title="Credited Amount by Payment Method",
                        color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig_pm.update_traces(textposition='inside', textinfo='percent+label')
        fig_pm.update_layout(height=460)
        st.plotly_chart(fig_pm, use_container_width=True)

# =====================================================================
# TAB 5 — NATIVE CHARTS (Streamlit primitives)
# =====================================================================
with tab5:
    st.markdown("### ⚡ Streamlit Native Charts")
    st.caption("Quick, lightweight charts rendered directly by Streamlit.")

    # ---------- Frequency ----------
    with st.expander("🎛️ Controls", expanded=True):
        metric = st.selectbox(
            "Metric",
            ['Sales Amount', 'Credited Amount', 'Outstanding', 'Profit'],
            key="native_metric",
        )
        freq = st.selectbox(
            "Frequency", ['D', 'W', 'M'], index=2,
            format_func=lambda x: {'D': 'Daily', 'W': 'Weekly', 'M': 'Monthly'}[x],
            key="native_freq",
        )

    ts = (fdf.set_index('Date')[metric]
          .resample(freq).sum()
          .reset_index()
          .rename(columns={metric: 'value'})
          .dropna())

    # ---------- Area chart ----------
    st.markdown(f"#### 🟦 `st.area_chart` — {metric} over time")
    st.area_chart(ts.set_index('Date'), color="#3b82f6", height=300)

    # ---------- Line chart ----------
    st.markdown("#### 🟩 `st.line_chart` — multiple metrics")
    multi = (fdf.set_index('Date')
             .resample(freq)[['Sales Amount', 'Credited Amount', 'Outstanding']]
             .sum().reset_index().set_index('Date'))
    st.line_chart(multi, height=300,
                  color=['#3b82f6', '#10b981', '#ef4444'])

    # ---------- Scatter chart ----------
    st.markdown("#### 🟨 `st.scatter_chart` — Discount vs Sales")
    sample = fdf.sample(min(2000, len(fdf)), random_state=1)
    st.scatter_chart(
        sample, x='Discount', y='Sales Amount',
        color='Customer Type', size='Invoice Value', height=340,
    )

    # ---------- Bar chart ----------
    st.markdown("#### 🟧 `st.bar_chart` — Sales by zone")
    z_bar = fdf.groupby('Sales Zone')['Sales Amount'].sum().reset_index()
    st.bar_chart(z_bar.set_index('Sales Zone'), height=300, color="#6366f1")

# =====================================================================
# EXPORT FULL ANALYTICS
# =====================================================================
st.markdown("---")
with st.expander("📥 Export all analytics as CSV bundle", expanded=False):
    monthly_export = (fdf.groupby('Month Name')
                      .agg(Sales=('Sales Amount', 'sum'),
                           Credited=('Credited Amount', 'sum'),
                           Outstanding=('Outstanding', 'sum'),
                           Returns=('Sales Return', 'sum'))
                      .reset_index())
    exec_export = (fdf.groupby('Executive')
                   .agg(Sales=('Sales Amount', 'sum'),
                        Credited=('Credited Amount', 'sum'),
                        Outstanding=('Outstanding', 'sum'))
                   .reset_index())
    zone_export = (fdf.groupby('Sales Zone')
                   .agg(Sales=('Sales Amount', 'sum'),
                        Credited=('Credited Amount', 'sum'),
                        Outstanding=('Outstanding', 'sum'))
                   .reset_index())

    c1, c2, c3 = st.columns(3)
    c1.download_button("📥 Monthly trend",
                       monthly_export.to_csv(index=False).encode('utf-8'),
                       file_name="analytics_monthly.csv", mime="text/csv",
                       use_container_width=True)
    c2.download_button("📥 Executive summary",
                       exec_export.to_csv(index=False).encode('utf-8'),
                       file_name="analytics_executives.csv", mime="text/csv",
                       use_container_width=True)
    c3.download_button("📥 Zone summary",
                       zone_export.to_csv(index=False).encode('utf-8'),
                       file_name="analytics_zones.csv", mime="text/csv",
                       use_container_width=True)