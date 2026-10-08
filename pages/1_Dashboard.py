import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from styles import inject_theme, hero, chips, toast_success
from utils import get_df

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
inject_theme()

hero("Dashboard", "Interactive overview of sales, collections, and outstanding balances.", icon="📊")

df = get_df()

# =====================================================================
# SIDEBAR FILTERS  (grouped inside expanders)
# =====================================================================
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    with st.expander("📅 Date & Entities", expanded=True):
        date_range = st.date_input(
            "Date Range",
            [df['Date'].min().date(), df['Date'].max().date()]
        )
        selected_exec     = st.multiselect("Executives",      df['Executive'].unique(),      df['Executive'].unique())
        selected_customer = st.multiselect("Customers",       df['Customer Name'].unique(),  df['Customer Name'].unique())

    with st.expander("🏷️ Segments"):
        selected_ctype    = st.multiselect("Customer Type",   df['Customer Type'].unique(),  df['Customer Type'].unique())
        selected_zone     = st.multiselect("Sales Zone",      df['Sales Zone'].unique(),     df['Sales Zone'].unique())
        selected_channel  = st.multiselect("Sales Channel",   df['Sales Channel'].unique(),  df['Sales Channel'].unique())

    with st.expander("💳 Payments"):
        selected_pm       = st.multiselect("Payment Method",  df['Payment Method'].unique(), df['Payment Method'].unique())
        selected_banks    = st.multiselect("Bank",            df['Bank Name'].unique(),      df['Bank Name'].unique())

    with st.expander("⚡ Amount Ranges"):
        smin, smax = int(df['Sales Amount'].min()), int(df['Sales Amount'].max())
        omin, omax = int(df['Outstanding'].min()),  int(df['Outstanding'].max())
        c1, c2 = st.columns(2)
        min_sales = c1.number_input("Min Sales", smin, smax, smin)
        max_sales = c2.number_input("Max Sales", smin, smax, smax)
        c3, c4 = st.columns(2)
        min_out = c3.number_input("Min Outstanding", omin, omax, omin)
        max_out = c4.number_input("Max Outstanding", omin, omax, omax)

    if st.button("🔄 Reset Filters", use_container_width=True):
        st.rerun()

# =====================================================================
# APPLY FILTERS
# =====================================================================
filtered_df = df[
    (df['Executive'].isin(selected_exec)) &
    (df['Customer Name'].isin(selected_customer)) &
    (df['Customer Type'].isin(selected_ctype)) &
    (df['Sales Zone'].isin(selected_zone)) &
    (df['Sales Channel'].isin(selected_channel)) &
    (df['Payment Method'].isin(selected_pm)) &
    (df['Bank Name'].isin(selected_banks)) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1])) &
    (df['Sales Amount'].between(min_sales, max_sales)) &
    (df['Outstanding'].between(min_out, max_out))
]

# =====================================================================
# STATUS CHIPS
# =====================================================================
chips([
    (f"📦 {len(filtered_df):,} records", ""),
    (f"📉 {(len(filtered_df)/len(df)*100):.1f}% of total", "green" if len(filtered_df) > len(df)*0.5 else "amber"),
    (f"📅 {date_range[0]:%d/%m/%y} → {date_range[1]:%d/%m/%y}", ""),
])

# =====================================================================
# PRIMARY KPIs
# =====================================================================
st.markdown("#### 🎯 Key Performance Indicators")
k1, k2, k3, k4, k5 = st.columns(5)
total_sales = filtered_df['Sales Amount'].sum()
total_paid  = filtered_df['Credited Amount'].sum()

k1.metric("💰 Total Sales",       f"৳ {total_sales:,.0f}")
k2.metric("💳 Total Credited",    f"৳ {total_paid:,.0f}")
k3.metric("📊 Outstanding",       f"৳ {filtered_df['Outstanding'].sum():,.0f}")
k4.metric("👨‍💼 Commission",     f"৳ {filtered_df['Commission'].sum():,.0f}")
k5.metric("📈 Profit",            f"৳ {filtered_df['Profit'].sum():,.0f}")

# =====================================================================
# SECONDARY METRICS
# =====================================================================
st.markdown("#### 📊 Performance Metrics")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📦 Avg Sale", f"৳ {filtered_df['Sales Amount'].mean():,.0f}")
m2.metric("🎯 Collection Rate",
          f"{(total_paid/total_sales*100) if total_sales else 0:.1f}%")
m3.metric("👥 Customers",   filtered_df['Customer Name'].nunique())
m4.metric("👨‍💼 Executives", filtered_df['Executive'].nunique())
m5.metric("⚡ Avg Outstanding", f"৳ {filtered_df['Outstanding'].mean():,.0f}")

# =====================================================================
# CHARTS — in expanders
# =====================================================================
st.markdown("---")
st.markdown("#### 📊 Performance Analytics")

if filtered_df.empty:
    st.info("No data for current filters.")
else:
    with st.expander("🏆 Executive & Bank", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            sales_exec = (filtered_df.groupby('Executive')
                          .agg({'Sales Amount':'sum','Credited Amount':'sum','Profit':'sum'})
                          .reset_index().sort_values('Sales Amount', ascending=False))
            fig = px.bar(sales_exec, x='Executive', y='Sales Amount',
                         title="Sales by Executive", color='Sales Amount',
                         color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            bank_pay = (filtered_df.groupby('Bank Name')['Credited Amount']
                        .sum().reset_index().sort_values('Credited Amount', ascending=False))
            fig = px.pie(bank_pay, values='Credited Amount', names='Bank Name',
                         title="Payment by Bank", hole=0.4,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📉 Outstanding & Daily Trend", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            cust_out = (filtered_df.groupby('Customer Name')['Outstanding']
                        .sum().reset_index().nlargest(15, 'Outstanding'))
            fig = px.bar(cust_out, x='Customer Name', y='Outstanding',
                         title="Top 15 Customers by Outstanding",
                         color='Outstanding', color_continuous_scale='reds')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            daily = (filtered_df.groupby('Date')
                     .agg({'Sales Amount':'sum','Credited Amount':'sum','Outstanding':'sum'})
                     .reset_index())
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Sales Amount'],
                                     mode='lines+markers', name='Sales',
                                     line=dict(color='#3b82f6', width=3)))
            fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Credited Amount'],
                                     mode='lines+markers', name='Credited',
                                     line=dict(color='#10b981', width=3)))
            fig.add_trace(go.Scatter(x=daily['Date'], y=daily['Outstanding'],
                                     mode='lines+markers', name='Outstanding',
                                     line=dict(color='#ef4444', width=2)))
            fig.update_layout(title="Daily Trends", hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("🛒 Channel & Customer Type"):
        c1, c2 = st.columns(2)
        with c1:
            ch = filtered_df.groupby('Sales Channel')['Sales Amount'].sum().reset_index()
            st.plotly_chart(px.bar(ch, x='Sales Channel', y='Sales Amount',
                                   title="Sales by Channel", color='Sales Channel'),
                            use_container_width=True)
        with c2:
            ct = filtered_df.groupby('Customer Type')['Sales Amount'].sum().reset_index()
            st.plotly_chart(px.pie(ct, values='Sales Amount', names='Customer Type',
                                   title="Sales by Customer Type", hole=0.3),
                            use_container_width=True)

# =====================================================================
# QUICK ACTIONS
# =====================================================================
st.markdown("---")
a1, a2, a3 = st.columns(3)
with a1:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export Filtered CSV", csv,
                       file_name=f"dashboard_{datetime.now():%Y%m%d_%H%M}.csv",
                       mime="text/csv", use_container_width=True)
with a2:
    if st.button("📊 Show Detailed Table", use_container_width=True):
        with st.expander("📋 Filtered Data", expanded=True):
            st.dataframe(filtered_df, use_container_width=True, height=400)
        toast_success("Table displayed below.")
with a3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()