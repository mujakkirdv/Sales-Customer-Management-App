import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import get_df

from styles import inject_theme, hero
# ...
st.set_page_config(page_title="Customer Management", page_icon="👥", layout="wide")
inject_theme()
hero("Customer Management Dashboard", "Manage and analyze customer data", icon="👥")

st.set_page_config(page_title="Customer Management", page_icon="👥", layout="wide")
st.subheader("👥 Customer Management")
df = get_df()

# ---------- Filters ----------
c1, c2, c3 = st.columns(3)
customer_filter = c1.selectbox("Customer", ['All'] + list(df['Customer Name'].unique()))
date_range_c    = c2.date_input("Date Range",
                                [df['Date'].min().date(), df['Date'].max().date()])
show_dash       = c3.button("📊 Customer Dashboard")

fdf = df[(df['Date'] >= pd.to_datetime(date_range_c[0])) &
         (df['Date'] <= pd.to_datetime(date_range_c[1]))]
if customer_filter != 'All':
    fdf = fdf[fdf['Customer Name'] == customer_filter]

# ---------- Summary table ----------
summary = fdf.groupby('Customer Name').agg({
    'Customer Type':   lambda x: x.mode()[0] if len(x.mode()) else '-',
    'Outstanding':     'sum',
    'Credited Amount': 'sum',
    'Sales Amount':    'sum',
    'Sales Return':    'sum',
    'Date':            'max',
    'Bank Name':       lambda x: x.mode()[0] if len(x.mode()) else '-'
}).reset_index()
summary.columns = ['Customer Name', 'Type', 'Outstanding', 'Total Credited',
                   'Total Sales', 'Total Returns', 'Last Transaction', 'Preferred Bank']

# ---------- Individual dashboard ----------
if show_dash and customer_filter != 'All':
    st.markdown(f"### 📊 Customer: **{customer_filter}**")
    row = summary[summary['Customer Name'] == customer_filter].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Sales",   f"৳{row['Total Sales']:,.0f}")
    k2.metric("Credited",      f"৳{row['Total Credited']:,.0f}")
    k3.metric("Outstanding",   f"৳{row['Outstanding']:,.0f}")
    k4.metric("Returns",       f"৳{row['Total Returns']:,.0f}")

    cc1, cc2 = st.columns(2)
    cdf = fdf[fdf['Customer Name'] == customer_filter]
    with cc1:
        pm = cdf.groupby('Payment Method')['Credited Amount'].sum().reset_index()
        st.plotly_chart(px.pie(pm, values='Credited Amount', names='Payment Method',
                               title="Payment Methods"), use_container_width=True)
    with cc2:
        tr = cdf.groupby('Date').agg({'Sales Amount':'sum',
                                      'Credited Amount':'sum'}).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tr['Date'], y=tr['Sales Amount'],
                                 mode='lines', name='Sales'))
        fig.add_trace(go.Scatter(x=tr['Date'], y=tr['Credited Amount'],
                                 mode='lines', name='Credited'))
        fig.update_layout(title="Sales vs Credited Trend")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔄 Last 10 Transactions")
    st.dataframe(cdf.sort_values('Date', ascending=False)
                 [['Date','Invoice No','Sales Amount','Credited Amount',
                   'Outstanding','Payment Method','Bank Name']].head(10),
                 use_container_width=True)

# ---------- All-customers view ----------
if customer_filter == 'All':
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers",   len(summary))
    k2.metric("Total Sales",       f"৳{summary['Total Sales'].sum():,.0f}")
    k3.metric("Total Credited",    f"৳{summary['Total Credited'].sum():,.0f}")
    k4.metric("Total Outstanding", f"৳{summary['Outstanding'].sum():,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        top = summary.nlargest(10, 'Total Sales')
        st.plotly_chart(px.bar(top, x='Customer Name', y='Total Sales',
                               title="Top 10 Customers by Sales"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(summary, x='Outstanding',
                                     title="Outstanding Distribution"),
                        use_container_width=True)

st.subheader("📋 Customer Details")
st.dataframe(summary.sort_values('Outstanding', ascending=False),
             use_container_width=True)

if customer_filter != 'All':
    st.subheader(f"📊 Full History – {customer_filter}")
    st.dataframe(fdf.sort_values('Date', ascending=False), use_container_width=True)