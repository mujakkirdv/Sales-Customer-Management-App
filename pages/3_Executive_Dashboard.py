import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils import get_df


from styles import inject_theme, hero
# ...
st.set_page_config(page_title="Executive Dashboard", page_icon="👨‍💼", layout="wide")
inject_theme()
hero("Executive Dashboard", "Monitor executive performance and sales metrics", icon="👨‍💼")

st.set_page_config(page_title="Executive Dashboard", page_icon="👨‍💼", layout="wide")
st.subheader("👨‍💼 Executive Performance Dashboard")
df = get_df()

# ---------- Aggregate ----------
exec_summary = df.groupby('Executive').agg({
    'Outstanding':'sum',
    'Credited Amount':'sum',
    'Sales Amount':'sum',
    'Profit':'sum',
    'Commission':'sum',
    'Customer Name':'nunique',
    'Invoice No':'count'
}).reset_index()
exec_summary.columns = ['Executive','Outstanding','Collected','Sales',
                        'Profit','Commission','Customers','Invoices']
exec_summary['Collection %'] = np.where(
    exec_summary['Sales'] > 0,
    exec_summary['Collected'] / exec_summary['Sales'] * 100, 0)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Executives",       len(exec_summary))
k2.metric("Total Sales",      f"৳{exec_summary['Sales'].sum():,.0f}")
k3.metric("Outstanding",      f"৳{exec_summary['Outstanding'].sum():,.0f}")
k4.metric("Commission Paid",  f"৳{exec_summary['Commission'].sum():,.0f}")

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.bar(exec_summary, x='Executive', y='Sales',
                           title="Sales by Executive", color='Sales'),
                    use_container_width=True)
with c2:
    st.plotly_chart(px.bar(exec_summary, x='Executive', y='Collection %',
                           title="Collection Rate (%)", color='Collection %'),
                    use_container_width=True)

st.markdown("---")
st.subheader("📋 Executive Details")
sel = st.selectbox("Select Executive", exec_summary['Executive'].unique())
row = exec_summary[exec_summary['Executive'] == sel].iloc[0]
edf = df[df['Executive'] == sel]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Sales",       f"৳{row['Sales']:,.0f}")
k2.metric("Collected",   f"৳{row['Collected']:,.0f}")
k3.metric("Outstanding", f"৳{row['Outstanding']:,.0f}")
k4.metric("Commission",  f"৳{row['Commission']:,.0f}")

c1, c2 = st.columns(2)
with c1:
    zone = edf.groupby('Sales Zone')['Sales Amount'].sum().reset_index()
    st.plotly_chart(px.bar(zone, x='Sales Zone', y='Sales Amount',
                           title="Sales by Zone"), use_container_width=True)
with c2:
    ch = edf.groupby('Sales Channel')['Sales Amount'].sum().reset_index()
    st.plotly_chart(px.pie(ch, values='Sales Amount', names='Sales Channel',
                           title="Sales by Channel"), use_container_width=True)

st.subheader(f"👥 Customers managed by {sel}")
cust = (edf.groupby('Customer Name')
        .agg({'Outstanding':'sum','Credited Amount':'sum',
              'Sales Amount':'sum','Date':'max'})
        .reset_index().sort_values('Outstanding', ascending=False))
st.dataframe(cust, use_container_width=True)