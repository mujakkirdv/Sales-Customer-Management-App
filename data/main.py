import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
import os

st.set_page_config(page_title="Sales & Customer App", layout="wide")
st.title("🚀 Sales & Customer Management App")

'''
# Data set updated column 
Index(['Date', 'Transaction ID', 'Invoice No', 'Customer Name',
       'Customer Type', 'Executive', 'Sales Zone', 'Sales Channel',
       'Invoice Value', 'Discount', 'Sales Amount', 'Sales VAT',
       'Sales Return', 'Credited Amount', 'Payment Method', 'Bank Name',
       'Remarks', 'Month Name'],
      dtype='object')

    build a final project based on updateded data column and write code page by page with all the features and functionalities of the app.
'''

# -------------------------------
# Load or Create Sample Data with Bank Name
# -------------------------------
@st.cache_data
def load_data(n=100000):
    np.random.seed(42)
    
    # Bank names for payment method
    banks = ['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 'MBL WB CC', 'MBL WB CD',
             'Cash', 'BRAC BANK', 'Credit Card', 'Debit Card', 'Cheque To MBL CD']
    
    executives = ["ATM Nur Hossain Rumel",
                  "Mynuddin Hasan hridoy",
                  "Sujoy Kumar Biswas",
                  "Sajib Kumar Biswas",
                  "Sanjoy Hore",
                  "Mohammad Sumon",
                  "Al - Amin",]
    
    # Try to load from Excel file if exists, otherwise create sample data
    try:
        if os.path.exists("sales_data.xlsx"):
            df = pd.read_excel("sales_data.xlsx")
            # Ensure all required columns exist
            required_columns = ['Date', 'Order no', 'Customer Name', 'Executive Name', 'Area code', 
                              'Opening balance', 'Sales Value', 'Sales return', 'Sales transfer', 
                              'Paid Amount', 'Cashback']
            
            for col in required_columns:
                if col not in df.columns:
                    if col == 'Date':
                        df[col] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
                    elif col == 'Order no':
                        df[col] = [f'ORD{1000+i}' for i in range(len(df))]
                    elif col == 'Customer Name':
                        df[col] = [f'Customer {i%100}' for i in range(len(df))]
                    elif col == 'Executive Name':
                        df[col] = np.random.choice(executives, len(df))
                    elif col == 'Area code':
                        df[col] = np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1'], len(df))
                    else:
                        df[col] = np.random.rand(len(df)) * 100
            
            # Add Bank Name if not exists
            if 'Bank Name' not in df.columns:
                df['Bank Name'] = np.random.choice(banks, len(df))
                
        else:
            # Create sample data if file doesn't exist
            df = pd.DataFrame({
                'Date': pd.date_range(start='2023-01-01', periods=n, freq='D'),
                'Order no': [f'ORD{1000+i}' for i in range(n)],
                'Customer Name': [f'Customer {i%100}' for i in range(n)],
                'Executive Name': np.random.choice(executives, n),
                'Area code': np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1'], n),
                'Opening balance': np.random.rand(n)*1000,
                'Sales Value': np.random.rand(n)*500,
                'Sales return': np.random.rand(n)*50,
                'Sales transfer': np.random.rand(n)*30,
                'Paid Amount': np.random.rand(n)*400,
                'Cashback': np.random.rand(n)*20,
                'Bank Name': np.random.choice(banks, n)
            })
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Fallback to sample data
        df = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=n, freq='D'),
            'Order no': [f'ORD{1000+i}' for i in range(n)],
            'Customer Name': [f'Customer {i%100}' for i in range(n)],
            'Executive Name': np.random.choice(executives, n),
            'Area code': np.random.choice(['A1', 'A2', 'B1', 'B2', 'C1'], n),
            'Opening balance': np.random.rand(n)*1000,
            'Sales Value': np.random.rand(n)*500,
            'Sales return': np.random.rand(n)*50,
            'Sales transfer': np.random.rand(n)*30,
            'Paid Amount': np.random.rand(n)*400,
            'Cashback': np.random.rand(n)*20,
            'Bank Name': np.random.choice(banks, n)
        })
    
    # Calculate outstanding using regular Python (no CUDA)
    df['Outstanding'] = df['Opening balance'] + df['Sales Value'] + df['Sales transfer'] - df['Sales return'] - df['Paid Amount'] - df['Cashback']
    df['Commission'] = df['Paid Amount'] * 0.01
    df['Profit'] = df['Paid Amount'] * 0.25
    
    return df

df = load_data()

# -------------------------------
# Sidebar Navigation
# -------------------------------
menu = ["Dashboard", 
        "Customer Management", 
        "Executive Dashboard", 
        "Sales Entry", 
        "Reports", 
        "Analytics",
        "Data Management",
        "Edit Data"]
choice = st.sidebar.radio("📑 Navigation", menu)

# -------------------------------
# Enhanced Dashboard
# -------------------------------

if choice == "Dashboard":
    st.subheader("📊 Dashboard")
    
    # Create a filter button in the sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("🎛️ Dashboard Filters")
        
        # Date Range Filter
        st.markdown("**📅 Date Range**")
        date_range = st.date_input(
            "Select Date Range", 
            [df['Date'].min().date(), df['Date'].max().date()],
            label_visibility="collapsed"
        )
        
        # Executive Filter with Search
        st.markdown("**👨‍💼 Executive Selection**")
        selected_exec = st.multiselect(
            "Select Executives",
            options=df['Executive Name'].unique(),
            default=df['Executive Name'].unique(),
            label_visibility="collapsed"
        )
        
        # Customer Filter with Search
        st.markdown("**👥 Customer Selection**")
        selected_customer = st.multiselect(
            "Select Customers",
            options=df['Customer Name'].unique(),
            default=df['Customer Name'].unique(),
            label_visibility="collapsed"
        )
        
        # Enhanced Sales and Outstanding Filters
        st.markdown("**⚡ Sales & Outstanding Filters**")
        
        # Calculate min/max values for better filter defaults
        sales_min_val = int(df['Sales Value'].min())
        sales_max_val = int(df['Sales Value'].max())
        outstanding_min_val = int(df['Outstanding'].min())
        outstanding_max_val = int(df['Outstanding'].max())
        
        col1, col2 = st.columns(2)
        with col1:
            min_sales = st.number_input(
                "Min Sales (৳)", 
                min_value=sales_min_val, 
                max_value=sales_max_val,
                value=sales_min_val,
                help="Filter by minimum sales amount"
            )
        with col2:
            max_sales = st.number_input(
                "Max Sales (৳)", 
                min_value=sales_min_val, 
                max_value=sales_max_val,
                value=sales_max_val,
                help="Filter by maximum sales amount"
            )
        
        col3, col4 = st.columns(2)
        with col3:
            min_outstanding = st.number_input(
                "Min Outstanding (৳)", 
                min_value=outstanding_min_val, 
                max_value=outstanding_max_val,
                value=outstanding_min_val,
                help="Filter by minimum outstanding amount"
            )
        with col4:
            max_outstanding = st.number_input(
                "Max Outstanding (৳)", 
                min_value=outstanding_min_val, 
                max_value=outstanding_max_val,
                value=outstanding_max_val,
                help="Filter by maximum outstanding amount"
            )
        
        # Additional Filters
        st.markdown("**🎯 Additional Filters**")
        
        # Bank Filter
        selected_banks = st.multiselect(
            "Select Payment Methods",
            options=df['Bank Name'].unique(),
            default=df['Bank Name'].unique(),
            help="Filter by payment methods"
        )
        
        # Apply Filters Button
        apply_filters = st.button("🚀 Apply Filters", use_container_width=True)
        clear_filters = st.button("🔄 Reset Filters", use_container_width=True)
        
        if clear_filters:
            selected_exec = df['Executive Name'].unique()
            selected_customer = df['Customer Name'].unique()
            selected_banks = df['Bank Name'].unique()
            date_range = [df['Date'].min().date(), df['Date'].max().date()]
            min_sales = sales_min_val
            max_sales = sales_max_val
            min_outstanding = outstanding_min_val
            max_outstanding = outstanding_max_val
            st.rerun()
    
    # Apply filters
    filtered_df = df[
        (df['Executive Name'].isin(selected_exec)) &
        (df['Customer Name'].isin(selected_customer)) &
        (df['Bank Name'].isin(selected_banks)) &
        (df['Date'] >= pd.to_datetime(date_range[0])) &
        (df['Date'] <= pd.to_datetime(date_range[1])) &
        (df['Sales Value'] >= min_sales) &
        (df['Sales Value'] <= max_sales) &
        (df['Outstanding'] >= min_outstanding) &
        (df['Outstanding'] <= max_outstanding)
    ]
    
    # Dashboard Header with Summary
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(f"### 📈 Performance Overview")
    with col2:
        st.metric("Records Found", len(filtered_df))
    with col3:
        st.metric("Filtered %", f"{(len(filtered_df)/len(df)*100):.1f}%")
    with col4:
        st.metric("Date Range", f"{date_range[0].strftime('%d/%m/%y')} to {date_range[1].strftime('%d/%m/%y')}")
    
    # KPIs with better styling and icons - FIXED SALES CALCULATION
    st.markdown("---")
    st.subheader("🎯 Key Performance Indicators")
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        # Calculate total sales from Sales Value column (not Price)
        total_sales = filtered_df['Sales Value'].sum()
        original_total_sales = df['Sales Value'].sum()
        sales_delta = total_sales - original_total_sales
        sales_delta_pct = (sales_delta / original_total_sales * 100) if original_total_sales > 0 else 0
        
        st.metric(
            label="💰 Total Sales",
            value=f"৳ {total_sales:,.0f}",
            delta=f"{sales_delta_pct:+.1f}%",
            delta_color="normal" if sales_delta >= 0 else "inverse"
        )
    
    with kpi2:
        total_paid = filtered_df['Paid Amount'].sum()
        original_paid = df['Paid Amount'].sum()
        paid_delta = total_paid - original_paid
        paid_delta_pct = (paid_delta / original_paid * 100) if original_paid > 0 else 0
        
        st.metric(
            label="💳 Total Paid",
            value=f"৳ {total_paid:,.0f}",
            delta=f"{paid_delta_pct:+.1f}%",
            delta_color="normal" if paid_delta >= 0 else "inverse"
        )
    
    with kpi3:
        total_outstanding = filtered_df['Outstanding'].sum()
        original_outstanding = df['Outstanding'].sum()
        outstanding_delta = total_outstanding - original_outstanding
        outstanding_color = "inverse" if total_outstanding > 100000000 else "normal"
        
        st.metric(
            label="📊 Total Outstanding",
            value=f"৳ {total_outstanding:,.0f}",
            delta=f"৳ {outstanding_delta:,.0f}",
            delta_color=outstanding_color
        )
    
    with kpi4:
        total_commission = filtered_df['Commission'].sum()
        original_commission = df['Commission'].sum()
        commission_delta = total_commission - original_commission
        commission_delta_pct = (commission_delta / original_commission * 100) if original_commission > 0 else 0
        
        st.metric(
            label="👨‍💼 Total Commission",
            value=f"৳ {total_commission:,.0f}",
            delta=f"{commission_delta_pct:+.1f}%",
            delta_color="normal" if commission_delta >= 0 else "inverse"
        )
    
    with kpi5:
        total_profit = filtered_df['Profit'].sum()
        original_profit = df['Profit'].sum()
        profit_delta = total_profit - original_profit
        profit_delta_pct = (profit_delta / original_profit * 100) if original_profit > 0 else 0
        
        st.metric(
            label="📈 Total Profit",
            value=f"৳ {total_profit:,.0f}",
            delta=f"{profit_delta_pct:+.1f}%",
            delta_color="normal" if profit_delta >= 0 else "inverse"
        )
    
    # Additional Metrics Row
    st.markdown("---")
    st.subheader("📊 Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_sale = filtered_df['Sales Value'].mean()
        st.metric("📦 Avg Sale", f"৳ {avg_sale:,.0f}")
    
    with col2:
        collection_rate = (total_paid / total_sales * 100) if total_sales > 0 else 0
        original_collection_rate = (df['Paid Amount'].sum() / df['Sales Value'].sum() * 100) if df['Sales Value'].sum() > 0 else 0
        collection_delta = collection_rate - original_collection_rate
        
        st.metric(
            "🎯 Collection Rate", 
            f"{collection_rate:.1f}%",
            delta=f"{collection_delta:+.1f}%",
            delta_color="normal" if collection_delta >= 0 else "inverse"
        )
    
    with col3:
        total_customers = filtered_df['Customer Name'].nunique()
        original_customers = df['Customer Name'].nunique()
        customer_delta = total_customers - original_customers
        
        st.metric(
            "👥 Customers", 
            total_customers,
            delta=customer_delta,
            delta_color="normal" if customer_delta >= 0 else "inverse"
        )
    
    with col4:
        total_executives = filtered_df['Executive Name'].nunique()
        original_executives = df['Executive Name'].nunique()
        executive_delta = total_executives - original_executives
        
        st.metric(
            "👨‍💼 Executives", 
            total_executives,
            delta=executive_delta,
            delta_color="normal" if executive_delta >= 0 else "inverse"
        )
    
    with col5:
        avg_outstanding = filtered_df['Outstanding'].mean()
        original_avg_outstanding = df['Outstanding'].mean()
        outstanding_avg_delta = avg_outstanding - original_avg_outstanding
        
        st.metric(
            "⚡ Avg Outstanding", 
            f"৳ {avg_outstanding:,.0f}",
            delta=f"৳ {outstanding_avg_delta:,.0f}",
            delta_color="normal" if outstanding_avg_delta <= 0 else "inverse"  # Lower is better
        )
    
    # Main Charts Section
    st.markdown("---")
    st.subheader("📊 Performance Analytics")
    
    # First Row of Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Sales by Executive - Enhanced chart
        if not filtered_df.empty:
            sales_exec = filtered_df.groupby('Executive Name').agg({
                'Sales Value': 'sum',
                'Paid Amount': 'sum',
                'Profit': 'sum'
            }).reset_index().sort_values('Sales Value', ascending=False)
            
            fig_sales = px.bar(sales_exec, x='Executive Name', y='Sales Value', 
                              title="🏆 Sales Performance by Executive",
                              color='Sales Value',
                              color_continuous_scale='viridis',
                              labels={'Sales Value': 'Sales Amount (৳)', 'Executive Name': 'Executive'})
            fig_sales.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_sales, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    with col2:
        # Payment Methods Distribution
        if not filtered_df.empty:
            bank_payments = filtered_df.groupby('Bank Name').agg({
                'Paid Amount': 'sum',
                'Sales Value': 'sum'
            }).reset_index().sort_values('Paid Amount', ascending=False)
            
            # Show top 8 banks, group others as "Other"
            if len(bank_payments) > 8:
                top_banks = bank_payments.head(8)
                other_sum = bank_payments['Paid Amount'].iloc[8:].sum()
                other_row = pd.DataFrame({'Bank Name': ['Other'], 'Paid Amount': [other_sum], 'Sales Value': [bank_payments['Sales Value'].iloc[8:].sum()]})
                bank_payments = pd.concat([top_banks, other_row])
            
            fig_pie = px.pie(bank_payments, values='Paid Amount', names='Bank Name', 
                            title="💳 Payment Method Distribution",
                            hole=0.4,
                            color_discrete_sequence=px.colors.sequential.RdBu)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    # Second Row of Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Outstanding by Customer (Top 15)
        if not filtered_df.empty:
            out_customer = filtered_df.groupby('Customer Name')['Outstanding'].sum().reset_index()
            top_outstanding = out_customer.nlargest(15, 'Outstanding')
            
            fig_out = px.bar(top_outstanding, x='Customer Name', y='Outstanding',
                            title="📊 Top 15 Customers by Outstanding Amount",
                            color='Outstanding',
                            color_continuous_scale='reds',
                            labels={'Outstanding': 'Outstanding Amount (৳)', 'Customer Name': 'Customer'})
            fig_out.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_out, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    with col2:
        # Daily Trends
        if not filtered_df.empty:
            daily_metrics = filtered_df.groupby('Date').agg({
                'Sales Value': 'sum',
                'Paid Amount': 'sum',
                'Outstanding': 'sum'
            }).reset_index()
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Sales Value'], 
                                         mode='lines+markers', name='Sales', line=dict(color='#1f77b4', width=3)))
            fig_trend.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Paid Amount'], 
                                         mode='lines+markers', name='Collections', line=dict(color='#2ca02c', width=3)))
            fig_trend.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Outstanding'], 
                                         mode='lines+markers', name='Outstanding', line=dict(color='#d62728', width=2)))
            fig_trend.update_layout(title="📈 Daily Performance Trends",
                                  xaxis_title="Date",
                                  yaxis_title="Amount (৳)",
                                  hovermode='x unified')
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    # Quick Actions Section
    st.markdown("---")
    st.subheader("⚡ Quick Actions & Export")
    
    action_col1, action_col2, action_col3, action_col4 = st.columns(4)
    
    with action_col1:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export CSV",
            data=csv,
            file_name=f"dashboard_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with action_col2:
        if st.button("📊 View Detailed Table", use_container_width=True):
            st.subheader("🗂 Detailed Data Table")
            st.dataframe(
                filtered_df.style.format({
                    'Sales Value': '৳{:,.2f}',
                    'Paid Amount': '৳{:,.2f}',
                    'Outstanding': '৳{:,.2f}',
                    'Profit': '৳{:,.2f}',
                    'Commission': '৳{:,.2f}'
                }).background_gradient(subset=['Sales Value', 'Profit'], cmap='Blues'),
                use_container_width=True,
                height=400
            )
    
    with action_col3:
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()
    
    with action_col4:
        if st.button("📈 Show Summary", use_container_width=True):
            # Performance Summary Cards
            st.markdown("---")
            st.subheader("🎯 Performance Summary")
            
            if not filtered_df.empty:
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    top_exec = sales_exec.loc[sales_exec['Sales Value'].idxmax()]
                    st.info(f"**🏆 Top Executive**  \n{top_exec['Executive Name']}  \nSales: ৳{top_exec['Sales Value']:,.0f}  \nProfit: ৳{top_exec['Profit']:,.0f}")
                
                with summary_col2:
                    best_collection = filtered_df.groupby('Executive Name').apply(
                        lambda x: (x['Paid Amount'].sum() / x['Sales Value'].sum() * 100) if x['Sales Value'].sum() > 0 else 0
                    ).idxmax()
                    best_rate = filtered_df.groupby('Executive Name').apply(
                        lambda x: (x['Paid Amount'].sum() / x['Sales Value'].sum() * 100) if x['Sales Value'].sum() > 0 else 0
                    ).max()
                    st.success(f"**🎯 Best Collection Rate**  \n{best_collection}  \nRate: {best_rate:.1f}%")
                
                with summary_col3:
                    highest_outstanding = top_outstanding.loc[top_outstanding['Outstanding'].idxmax()]
                    st.warning(f"**⚠️ Highest Outstanding**  \n{highest_outstanding['Customer Name']}  \nAmount: ৳{highest_outstanding['Outstanding']:,.0f}")
                
                with summary_col4:
                    popular_payment = bank_payments.loc[bank_payments['Paid Amount'].idxmax()]
                    st.error(f"**💳 Most Popular Payment**  \n{popular_payment['Bank Name']}  \nAmount: ৳{popular_payment['Paid Amount']:,.0f}  \nUsage: {popular_payment['Paid Amount']/bank_payments['Paid Amount'].sum()*100:.1f}%")

# -------------------------------
# Enhanced Customer Management with Dashboard
# -------------------------------
elif choice == "Customer Management":
    st.subheader("👥 Customer Management Dashboard")
    
    # Customer filters
    col1, col2, col3 = st.columns(3)
    with col1:
        customer_filter = st.selectbox("Select Customer", options=['All'] + list(df['Customer Name'].unique()))
    with col2:
        date_range_customer = st.date_input("Date Range", [df['Date'].min().date(), df['Date'].max().date()], key="customer_date_range")
    with col3:
        show_dashboard = st.button("📊 Customer Dashboard")
    
    # Apply date filter
    filtered_customer_df = df[
        (df['Date'] >= pd.to_datetime(date_range_customer[0])) &
        (df['Date'] <= pd.to_datetime(date_range_customer[1]))
    ]
    
    if customer_filter != 'All':
        filtered_customer_df = filtered_customer_df[filtered_customer_df['Customer Name'] == customer_filter]
    
    # Customer summary with date range
    customer_summary = filtered_customer_df.groupby('Customer Name').agg({
        'Outstanding': 'sum',
        'Paid Amount': 'sum',
        'Sales Value': 'sum',
        'Cashback': 'sum',
        'Date': 'max',
        'Bank Name': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).reset_index()
    
    customer_summary.columns = ['Customer Name', 'Total Outstanding', 'Total Paid', 
                               'Total Sales', 'Total Cashback', 'Last Payment Date', 'Preferred Payment Method']
    
    # Display Customer Dashboard when button is clicked
    if show_dashboard and customer_filter != 'All':
        st.subheader(f"📊 Customer Dashboard: {customer_filter}")
        
        # Customer Metrics
        customer_metrics = customer_summary[customer_summary['Customer Name'] == customer_filter].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sales", f"৳{customer_metrics['Total Sales']:,.2f}")
        with col2:
            st.metric("Total Paid", f"৳{customer_metrics['Total Paid']:,.2f}")
        with col3:
            st.metric("Outstanding", f"৳{customer_metrics['Total Outstanding']:,.2f}")
        with col4:
            st.metric("Cashback", f"৳{customer_metrics['Total Cashback']:,.2f}")
        
        # Customer Charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Payment method distribution for this customer
            customer_payments = filtered_customer_df[filtered_customer_df['Customer Name'] == customer_filter]
            payment_summary = customer_payments.groupby('Bank Name')['Paid Amount'].sum().reset_index()
            fig_payment = px.pie(payment_summary, values='Paid Amount', names='Bank Name',
                                title=f"Payment Methods - {customer_filter}")
            st.plotly_chart(fig_payment, use_container_width=True)
        
        with col2:
            # Sales trend for this customer
            customer_trend = customer_payments.groupby('Date').agg({
                'Sales Value': 'sum',
                'Paid Amount': 'sum'
            }).reset_index()
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=customer_trend['Date'], y=customer_trend['Sales Value'],
                                         mode='lines', name='Sales', line=dict(color='blue')))
            fig_trend.add_trace(go.Scatter(x=customer_trend['Date'], y=customer_trend['Paid Amount'],
                                         mode='lines', name='Payments', line=dict(color='green')))
            fig_trend.update_layout(title=f"Sales & Payment Trend - {customer_filter}")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Last 5 transactions
        st.subheader("🔄 Last 5 Transactions")
        last_transactions = customer_payments.sort_values('Date', ascending=False).head(5)
        st.dataframe(last_transactions[['Date', 'Order no', 'Sales Value', 'Paid Amount', 'Bank Name', 'Outstanding']])
    
    # Display metrics for all customers
    if customer_filter == 'All':
        total_customers = len(customer_summary)
        total_outstanding = customer_summary['Total Outstanding'].sum()
        total_sales = customer_summary['Total Sales'].sum()
        total_paid = customer_summary['Total Paid'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", total_customers)
        col2.metric("Total Sales", f"৳{total_sales:,.2f}")
        col3.metric("Total Paid", f"৳{total_paid:,.2f}")
        col4.metric("Total Outstanding", f"৳{total_outstanding:,.2f}")
        
        # Visualizations for all customers
        st.subheader("📊 Customer Analytics")
        col1, col2 = st.columns(2)
        
        with col1:
            # Top customers by sales
            top_customers = customer_summary.nlargest(10, 'Total Sales')
            fig_top_sales = px.bar(top_customers, x='Customer Name', y='Total Sales',
                                  title="Top 10 Customers by Sales")
            st.plotly_chart(fig_top_sales, use_container_width=True)
        
        with col2:
            # Outstanding distribution
            fig_outstanding = px.histogram(customer_summary, x='Total Outstanding',
                                         title="Outstanding Distribution Across Customers")
            st.plotly_chart(fig_outstanding, use_container_width=True)
    
    # Customer details table
    st.subheader("📋 Customer Details")
    st.dataframe(customer_summary.sort_values('Total Outstanding', ascending=False))
    
    # Customer drill-down
    if customer_filter != 'All':
        st.subheader(f"📊 Transaction History for {customer_filter}")
        customer_transactions = filtered_customer_df[filtered_customer_df['Customer Name'] == customer_filter].sort_values('Date', ascending=False)
        st.dataframe(customer_transactions)


# -------------------------------
# Executive Dashboard
# -------------------------------
elif choice == "Executive Dashboard":
    st.subheader("👨‍💼 Executive Performance Dashboard")
    
    # Executive summary
    executive_summary = df.groupby('Executive Name').agg({
        'Outstanding': 'sum',
        'Paid Amount': 'sum',
        'Sales Value': 'sum',
        'Customer Name': 'nunique',
        'Commission': 'sum',
        'Profit': 'sum'
    }).reset_index()
    
    executive_summary.columns = ['Executive Name', 'Total Outstanding', 'Total Collected', 
                                'Total Sales', 'Number of Customers', 'Total Commission', 'Total Profit']
    
    # Executive metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Executives", len(executive_summary))
    col2.metric("Total Executive Sales", f"৳{executive_summary['Total Sales'].sum():,.0f}")
    col3.metric("Total Outstanding", f"৳{executive_summary['Total Outstanding'].sum():,.0f}")
    col4.metric("Total Commission Paid", f"৳{executive_summary['Total Commission'].sum():,.0f}")
    
    # Executive comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Sales Performance")
        fig_sales = px.bar(executive_summary, x='Executive Name', y='Total Sales',
                          title="Sales by Executive", color='Total Sales')
        st.plotly_chart(fig_sales, use_container_width=True)
    
    with col2:
        st.subheader("💰 Collection Efficiency")
        executive_summary['Collection_Rate'] = (executive_summary['Total Collected'] / 
                                              executive_summary['Total Sales'] * 100)
        fig_collection = px.bar(executive_summary, x='Executive Name', y='Collection_Rate',
                               title="Collection Rate (%) by Executive", color='Collection_Rate')
        st.plotly_chart(fig_collection, use_container_width=True)
    
    # Detailed executive view
    st.subheader("📋 Executive Details")
    selected_exec = st.selectbox("Select Executive for Details", executive_summary['Executive Name'].unique())
    
    exec_data = executive_summary[executive_summary['Executive Name'] == selected_exec].iloc[0]
    exec_transactions = df[df['Executive Name'] == selected_exec]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sales", f"৳{exec_data['Total Sales']:,.0f}")
    col2.metric("Amount Collected", f"৳{exec_data['Total Collected']:,.0f}")
    col3.metric("Outstanding", f"৳{exec_data['Total Outstanding']:,.0f}")
    col4.metric("Commission Earned", f"৳{exec_data['Total Commission']:,.0f}")
    
    # Executive's customers outstanding
    st.subheader(f"👥 Customers Managed by {selected_exec}")
    exec_customers = exec_transactions.groupby('Customer Name').agg({
        'Outstanding': 'sum',
        'Paid Amount': 'sum',
        'Sales Value': 'sum',
        'Date': 'max'
    }).reset_index()
    
    st.dataframe(exec_customers.sort_values('Outstanding', ascending=False))



# -------------------------------
# Enhanced Sales Entry with Bank Name
# -------------------------------
elif choice == "Sales Entry":
    st.subheader("📝 Add New Sale Entry")
    
    with st.form("add_sale", clear_on_submit=True):
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.now())
            order_no = st.text_input("Order No", value=f"ORD{len(df)+1000}")
            customer = st.text_input("Customer Name")
            executive = st.selectbox("Executive Name", options=df['Executive Name'].unique())
            area = st.text_input("Area Code")
        
        with col2:
            opening = st.number_input("Opening Balance", min_value=0.0, value=0.0)
            sales_val = st.number_input("Sales Value", min_value=0.0, value=0.0)
            sales_ret = st.number_input("Sales Return", min_value=0.0, value=0.0)
            sales_transfer = st.number_input("Sales Transfer", min_value=0.0, value=0.0)
            paid = st.number_input("Paid Amount", min_value=0.0, value=0.0)
            cashback = st.number_input("Cashback", min_value=0.0, value=0.0)
            bank_name = st.selectbox("Payment Method/Bank", 
                                   options=['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 
                                           'MBL WB CC', 'MBL WB CD', 'Cash', 'BRAC BANK', 
                                           'Credit Card', 'Debit Card', 'Cheque'])
        
        submit = st.form_submit_button("Add Sale Entry")
        
        if submit:
            # Calculate outstanding for new entry
            new_outstanding = opening + sales_val + sales_transfer - sales_ret - paid - cashback
            new_commission = paid * 0.01
            new_profit = paid * 0.25
            
            new_entry = {
                'Date': date,
                'Order no': order_no,
                'Customer Name': customer,
                'Executive Name': executive,
                'Area code': area,
                'Opening balance': opening,
                'Sales Value': sales_val,
                'Sales return': sales_ret,
                'Sales transfer': sales_transfer,
                'Paid Amount': paid,
                'Cashback': cashback,
                'Bank Name': bank_name,
                'Outstanding': new_outstanding,
                'Commission': new_commission,
                'Profit': new_profit
            }
            
            # Add to dataframe
            df.loc[len(df)] = new_entry
            st.success("✅ Sale Entry Added Successfully!")
            st.balloons()




# -------------------------------
# Enhanced Reports with Date Range and Visualizations
# -------------------------------
elif choice == "Reports":
    st.subheader("📑 Comprehensive Reports")
    
    # Report filters
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        report_type = st.selectbox("Select Report Type", [
            "Outstanding Report", 
            "Sales Report", 
            "Profit Report",
            "Payment Method Report",
            "Executive Performance Report",
            "Customer Summary Report",
            "Detailed Transaction Report",
            "Collection Efficiency Report"
        ])
    with col2:
        export_format = st.radio("Export Format", ["CSV", "Excel"])
    with col3:
        report_date_range = st.date_input("Date Range", [df['Date'].min().date(), df['Date'].max().date()], key="report_date_range")
    with col4:
        selected_exec_report = st.multiselect("Select Executive", options=['All'] + list(df['Executive Name'].unique()), default=['All'])
    with col5:
        selected_customer_report = st.multiselect("Select Customer", options=['All'] + list(df['Customer Name'].unique()), default=['All'])    
    
    # Apply all filters
    report_df_filtered = df[
        (df['Date'] >= pd.to_datetime(report_date_range[0])) &
        (df['Date'] <= pd.to_datetime(report_date_range[1]))
    ]
    
    # Apply executive filter
    if 'All' not in selected_exec_report:
        report_df_filtered = report_df_filtered[report_df_filtered['Executive Name'].isin(selected_exec_report)]
    
    # Apply customer filter
    if 'All' not in selected_customer_report:
        report_df_filtered = report_df_filtered[report_df_filtered['Customer Name'].isin(selected_customer_report)]
    
    # Generate reports based on selection
    if report_type == "Outstanding Report":
        report_df = report_df_filtered[['Date', 'Customer Name', 'Executive Name', 'Bank Name', 'Outstanding', 'Sales Value', 'Paid Amount']]
        st.subheader("📋 Outstanding Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Outstanding", f"৳{report_df['Outstanding'].sum():,.2f}")
        col2.metric("Average Outstanding", f"৳{report_df['Outstanding'].mean():,.2f}")
        col3.metric("Customers with Outstanding", len(report_df['Customer Name'].unique()))
        col4.metric("Highest Outstanding", f"৳{report_df['Outstanding'].max():,.2f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            # Top 20 Customers by Outstanding
            outstanding_by_customer = report_df.groupby('Customer Name')['Outstanding'].sum().nlargest(20)
            fig_outstanding = px.bar(outstanding_by_customer, 
                                   x=outstanding_by_customer.index, 
                                   y=outstanding_by_customer.values,
                                   title="Top 20 Customers by Outstanding",
                                   labels={'x': 'Customer Name', 'y': 'Outstanding Amount ($)'})
            st.plotly_chart(fig_outstanding, use_container_width=True)
        
        with col2:
            # Outstanding by Executive
            outstanding_by_exec = report_df.groupby('Executive Name')['Outstanding'].sum().reset_index()
            fig_exec_outstanding = px.pie(outstanding_by_exec, 
                                        values='Outstanding', 
                                        names='Executive Name',
                                        title="Outstanding Distribution by Executive")
            st.plotly_chart(fig_exec_outstanding, use_container_width=True)
        
        # Outstanding Trend
        st.subheader("📈 Outstanding Trend Analysis")
        outstanding_trend = report_df.groupby('Date')['Outstanding'].sum().reset_index()
        fig_trend = px.area(outstanding_trend, x='Date', y='Outstanding', 
                           title="Outstanding Amount Trend Over Time")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    elif report_type == "Sales Report":
        report_df = report_df_filtered[['Date', 'Customer Name', 'Executive Name', 'Bank Name', 'Sales Value', 'Paid Amount', 'Outstanding', 'Profit']]
        st.subheader("📊 Sales Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        total_sales = report_df['Sales Value'].sum()
        total_collections = report_df['Paid Amount'].sum()
        collection_rate = (total_collections / total_sales * 100) if total_sales > 0 else 0
        
        col1.metric("Total Sales", f"৳{total_sales:,.2f}")
        col2.metric("Total Collections", f"৳{total_collections:,.2f}")
        col3.metric("Collection Rate", f"{collection_rate:.1f}%")
        col4.metric("Average Sale Value", f"৳{report_df['Sales Value'].mean():,.2f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            # Daily Sales Trend
            daily_sales = report_df.groupby('Date').agg({
                'Sales Value': 'sum',
                'Paid Amount': 'sum'
            }).reset_index()
            fig_sales_trend = px.line(daily_sales, x='Date', y=['Sales Value', 'Paid Amount'],
                                    title="Daily Sales vs Collections Trend",
                                    labels={'value': 'Amount ($)', 'variable': 'Metric'})
            st.plotly_chart(fig_sales_trend, use_container_width=True)
        
        with col2:
            # Sales by Executive
            exec_sales = report_df.groupby('Executive Name').agg({
                'Sales Value': 'sum',
                'Paid Amount': 'sum'
            }).reset_index()
            fig_exec_sales = px.bar(exec_sales, x='Executive Name', y=['Sales Value', 'Paid Amount'],
                                  title="Sales & Collections by Executive",
                                  barmode='group')
            st.plotly_chart(fig_exec_sales, use_container_width=True)
        
        # Additional Analysis
        st.subheader("📈 Sales Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Customers by Sales
            top_customers_sales = report_df.groupby('Customer Name')['Sales Value'].sum().nlargest(10)
            fig_top_customers = px.bar(top_customers_sales, 
                                     x=top_customers_sales.index, 
                                     y=top_customers_sales.values,
                                     title="Top 10 Customers by Sales")
            st.plotly_chart(fig_top_customers, use_container_width=True)
        
        with col2:
            # Monthly Sales Breakdown
            report_df['Month'] = report_df['Date'].dt.to_period('M').astype(str)
            monthly_sales = report_df.groupby('Month')['Sales Value'].sum().reset_index()
            fig_monthly = px.line(monthly_sales, x='Month', y='Sales Value',
                                title="Monthly Sales Trend")
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    elif report_type == "Profit Report":
        report_df = report_df_filtered[['Date', 'Customer Name', 'Executive Name', 'Bank Name', 'Profit', 'Commission', 'Sales Value', 'Paid Amount']]
        st.subheader("💰 Profit Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        total_profit = report_df['Profit'].sum()
        total_commission = report_df['Commission'].sum()
        profit_margin = (total_profit / report_df['Sales Value'].sum() * 100) if report_df['Sales Value'].sum() > 0 else 0
        
        col1.metric("Total Profit", f"৳{total_profit:,.2f}")
        col2.metric("Total Commission", f"৳{total_commission:,.2f}")
        col3.metric("Profit Margin", f"{profit_margin:.1f}%")
        col4.metric("Profit per Sale", f"৳{report_df['Profit'].mean():,.2f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            # Profit Trend
            profit_trend = report_df.groupby('Date')['Profit'].sum().reset_index()
            fig_profit = px.area(profit_trend, x='Date', y='Profit', 
                               title="Daily Profit Trend")
            st.plotly_chart(fig_profit, use_container_width=True)
        
        with col2:
            # Profit by Executive
            profit_by_exec = report_df.groupby('Executive Name')['Profit'].sum().reset_index()
            fig_profit_exec = px.bar(profit_by_exec, x='Executive Name', y='Profit',
                                   title="Profit by Executive",
                                   color='Profit')
            st.plotly_chart(fig_profit_exec, use_container_width=True)
    
    elif report_type == "Payment Method Report":
        report_df = report_df_filtered.groupby('Bank Name').agg({
            'Paid Amount': 'sum',
            'Sales Value': 'sum',
            'Customer Name': 'nunique',
            'Profit': 'sum',
            'Date': 'count'
        }).reset_index()
        report_df.columns = ['Payment Method', 'Total Collected', 'Total Sales', 'Unique Customers', 'Total Profit', 'Transaction Count']
        report_df['Collection_Rate'] = (report_df['Total Collected'] / report_df['Total Sales'] * 100)
        
        st.subheader("💳 Payment Method Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Payment Methods", len(report_df))
        col2.metric("Most Used Method", report_df.loc[report_df['Transaction Count'].idxmax(), 'Payment Method'])
        col3.metric("Highest Collection Method", report_df.loc[report_df['Total Collected'].idxmax(), 'Payment Method'])
        col4.metric("Average Collection Rate", f"{report_df['Collection_Rate'].mean():.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig_payment = px.pie(report_df, values='Total Collected', names='Payment Method',
                               title="Payment Distribution by Amount Collected")
            st.plotly_chart(fig_payment, use_container_width=True)
        
        with col2:
            fig_sales_payment = px.bar(report_df, x='Payment Method', y=['Total Sales', 'Total Collected'],
                                     title="Sales vs Collections by Payment Method",
                                     barmode='group')
            st.plotly_chart(fig_sales_payment, use_container_width=True)
        
        # Additional Analysis
        st.subheader("📊 Payment Method Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_collection_rate = px.bar(report_df, x='Payment Method', y='Collection_Rate',
                                       title="Collection Rate by Payment Method (%)",
                                       color='Collection_Rate')
            st.plotly_chart(fig_collection_rate, use_container_width=True)
        
        with col2:
            fig_customers_payment = px.scatter(report_df, x='Unique Customers', y='Total Collected',
                                             size='Transaction Count', color='Payment Method',
                                             title="Customer Reach vs Amount Collected")
            st.plotly_chart(fig_customers_payment, use_container_width=True)
    
    elif report_type == "Executive Performance Report":
        report_df = report_df_filtered.groupby('Executive Name').agg({
            'Sales Value': 'sum',
            'Paid Amount': 'sum',
            'Outstanding': 'sum',
            'Commission': 'sum',
            'Profit': 'sum',
            'Customer Name': 'nunique',
            'Date': 'count'
        }).reset_index()
        report_df.columns = ['Executive Name', 'Total Sales', 'Total Collected', 'Total Outstanding', 
                           'Total Commission', 'Total Profit', 'Unique Customers', 'Transaction Count']
        report_df['Collection_Rate'] = (report_df['Total Collected'] / report_df['Total Sales'] * 100)
        report_df['Profit_Margin'] = (report_df['Total Profit'] / report_df['Total Sales'] * 100)
        
        st.subheader("👨‍💼 Executive Performance Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        top_exec = report_df.loc[report_df['Total Sales'].idxmax()]
        
        col1.metric("Top Executive", top_exec['Executive Name'])
        col2.metric("Highest Sales", f"৳{top_exec['Total Sales']:,.0f}")
        col3.metric("Best Collection Rate", f"{report_df['Collection_Rate'].max():.1f}%")
        col4.metric("Total Executives", len(report_df))
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig_sales = px.bar(report_df, x='Executive Name', y='Total Sales', 
                             title="Total Sales by Executive",
                             color='Total Sales')
            st.plotly_chart(fig_sales, use_container_width=True)
        
        with col2:
            fig_collection = px.bar(report_df, x='Executive Name', y='Collection_Rate', 
                                  title="Collection Rate by Executive (%)",
                                  color='Collection_Rate')
            st.plotly_chart(fig_collection, use_container_width=True)
        
        # Additional Metrics
        st.subheader("📈 Executive Efficiency Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_efficiency = px.scatter(report_df, x='Total Sales', y='Collection_Rate',
                                      size='Unique Customers', color='Executive Name',
                                      title="Sales vs Collection Rate Efficiency")
            st.plotly_chart(fig_efficiency, use_container_width=True)
        
        with col2:
            fig_profit_margin = px.bar(report_df, x='Executive Name', y='Profit_Margin',
                                     title="Profit Margin by Executive (%)",
                                     color='Profit_Margin')
            st.plotly_chart(fig_profit_margin, use_container_width=True)
    
    elif report_type == "Customer Summary Report":
        report_df = report_df_filtered.groupby('Customer Name').agg({
            'Sales Value': 'sum',
            'Paid Amount': 'sum',
            'Outstanding': 'sum',
            'Profit': 'sum',
            'Date': ['max', 'count'],
            'Bank Name': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
            'Executive Name': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        report_df.columns = ['Customer Name', 'Total Sales', 'Total Paid', 'Total Outstanding', 
                           'Total Profit', 'Last Transaction Date', 'Transaction Count', 
                           'Preferred Payment Method', 'Primary Executive']
        
        st.subheader("👥 Customer Summary Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(report_df))
        col2.metric("Active Customers", len(report_df[report_df['Transaction Count'] > 0]))
        col3.metric("Customers with Outstanding", len(report_df[report_df['Total Outstanding'] > 0]))
        col4.metric("Average Transaction per Customer", f"{report_df['Transaction Count'].mean():.1f}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            top_customers = report_df.nlargest(15, 'Total Sales')
            fig_customers = px.bar(top_customers, x='Customer Name', y='Total Sales', 
                                 title="Top 15 Customers by Sales",
                                 color='Total Sales')
            st.plotly_chart(fig_customers, use_container_width=True)
        
        with col2:
            fig_customer_value = px.scatter(report_df, x='Total Sales', y='Total Outstanding',
                                          size='Transaction Count', color='Total Paid',
                                          title="Customer Value Analysis",
                                          hover_name='Customer Name')
            st.plotly_chart(fig_customer_value, use_container_width=True)
    
    elif report_type == "Detailed Transaction Report":
        report_df = report_df_filtered[[
            'Date', 'Order no', 'Customer Name', 'Executive Name', 'Bank Name',
            'Sales Value', 'Paid Amount', 'Outstanding', 'Profit', 'Commission', 'Cashback'
        ]]
        st.subheader("📋 Detailed Transaction Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", len(report_df))
        col2.metric("Unique Customers", report_df['Customer Name'].nunique())
        col3.metric("Unique Executives", report_df['Executive Name'].nunique())
        col4.metric("Average Transaction Value", f"৳{report_df['Sales Value'].mean():,.2f}")
    
    elif report_type == "Collection Efficiency Report":
        report_df = report_df_filtered.groupby(['Executive Name', 'Customer Name']).agg({
            'Sales Value': 'sum',
            'Paid Amount': 'sum',
            'Outstanding': 'sum',
            'Date': 'max'
        }).reset_index()
        report_df['Collection_Rate'] = (report_df['Paid Amount'] / report_df['Sales Value'] * 100)
        report_df['Days_Since_Last_Transaction'] = (pd.to_datetime('today') - report_df['Date']).dt.days
        
        st.subheader("⚡ Collection Efficiency Report")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Collection Rate", f"{report_df['Collection_Rate'].mean():.1f}%")
        col2.metric("Best Collection Rate", f"{report_df['Collection_Rate'].max():.1f}%")
        col3.metric("Customers >90% Collection", len(report_df[report_df['Collection_Rate'] > 90]))
        col4.metric("Customers <50% Collection", len(report_df[report_df['Collection_Rate'] < 50]))
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig_efficiency_dist = px.histogram(report_df, x='Collection_Rate',
                                             title="Collection Rate Distribution",
                                             nbins=20)
            st.plotly_chart(fig_efficiency_dist, use_container_width=True)
        
        with col2:
            fig_aging = px.scatter(report_df, x='Days_Since_Last_Transaction', y='Outstanding',
                                 color='Collection_Rate', size='Sales Value',
                                 title="Outstanding Aging Analysis",
                                 hover_name='Customer Name')
            st.plotly_chart(fig_aging, use_container_width=True)

    # Display report data with pagination
    st.subheader("📊 Report Data")
    
    # Show basic statistics
    st.write(f"**Report Summary:** {len(report_df)} records found")
    
    # Add search and filter for the data table
    if len(report_df) > 1000:
        st.info(f"Showing first 1000 records out of {len(report_df)}. Use filters to reduce data size.")
        display_df = report_df.head(1000)
    else:
        display_df = report_df
    
    st.dataframe(display_df, use_container_width=True)
    
    # Enhanced Export functionality
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if export_format == "CSV":
            csv = report_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Report", 
                             data=csv, 
                             file_name=f"{report_type.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv", 
                             mime="text/csv")
        else:
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    report_df.to_excel(writer, index=False, sheet_name='Report')
                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Records', 'Report Type', 'Date Range', 'Generated On'],
                        'Value': [len(report_df), report_type, 
                                f"{report_date_range[0]} to {report_date_range[1]}", 
                                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, index=False, sheet_name='Summary')
                excel_data = output.getvalue()
                st.download_button(
                    label="📥 Download Excel Report",
                    data=excel_data,
                    file_name=f"{report_type.replace(' ', '_')}_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            except:
                st.info("Excel export requires openpyxl. Install with: pip install openpyxl")
    
    with col2:
        # Export summary statistics
        if export_format == "CSV":
            summary_stats = report_df.describe().round(2)
            csv_summary = summary_stats.to_csv().encode('utf-8')
            st.download_button("📊 Download Summary Stats", 
                             data=csv_summary,
                             file_name=f"{report_type.replace(' ', '_')}_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                             mime="text/csv")
    
    with col3:
        # Quick insights export
        insights = {
            'Report Type': [report_type],
            'Total Records': [len(report_df)],
            'Date Range': [f"{report_date_range[0]} to {report_date_range[1]}"],
            'Generated On': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')]
        }
        insights_df = pd.DataFrame(insights)
        csv_insights = insights_df.to_csv(index=False).encode('utf-8')
        st.download_button("📋 Download Report Info", 
                         data=csv_insights,
                         file_name=f"{report_type.replace(' ', '_')}_info_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                         mime="text/csv")
    
    # Report Insights Section
    st.subheader("💡 Quick Insights")
    
    if len(report_df) > 0:
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.info("**Data Overview**")
            st.write(f"• Total records: {len(report_df)}")
            st.write(f"• Date range: {report_date_range[0]} to {report_date_range[1]}")
            if 'Customer Name' in report_df.columns:
                st.write(f"• Unique customers: {report_df['Customer Name'].nunique()}")
            if 'Executive Name' in report_df.columns:
                st.write(f"• Unique executives: {report_df['Executive Name'].nunique()}")
        
        with insights_col2:
            st.info("**Performance Highlights**")
            numeric_cols = report_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:3]:  # Show first 3 numeric columns
                    if col in report_df.columns:
                        st.write(f"• Total {col}: ${report_df[col].sum():,.2f}")


# -------------------------------
# Analytics Dashboard
# -------------------------------
elif choice == "Analytics":
    st.subheader("📈 Advanced Analytics Dashboard")
    
    # Analytics Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        analytics_date_range = st.date_input("Date Range", 
                                           [df['Date'].min().date(), df['Date'].max().date()], 
                                           key="analytics_date_range")
    with col2:
        selected_exec_analytics = st.multiselect("Select Executive", 
                                               options=['All'] + list(df['Executive Name'].unique()),
                                               default=['All'])
    with col3:
        metric_choice = st.selectbox("Primary Metric", 
                                   ['Sales Value', 'Paid Amount', 'Outstanding', 'Profit', 'Commission'])
    
    # Filter data for analytics
    analytics_df = df[
        (df['Date'] >= pd.to_datetime(analytics_date_range[0])) &
        (df['Date'] <= pd.to_datetime(analytics_date_range[1]))
    ]
    
    if 'All' not in selected_exec_analytics:
        analytics_df = analytics_df[analytics_df['Executive Name'].isin(selected_exec_analytics)]
    
    # Key Metrics Summary
    st.subheader("📊 Performance Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Sales", f"৳{analytics_df['Sales Value'].sum():,.0f}")
    with col2:
        st.metric("Total Collected", f"৳{analytics_df['Paid Amount'].sum():,.0f}")
    with col3:
        st.metric("Outstanding", f"৳{analytics_df['Outstanding'].sum():,.0f}")
    with col4:
        st.metric("Total Profit", f"৳{analytics_df['Profit'].sum():,.0f}")
    with col5:
        collection_rate = (analytics_df['Paid Amount'].sum() / analytics_df['Sales Value'].sum() * 100) if analytics_df['Sales Value'].sum() > 0 else 0
        st.metric("Collection Rate", f"{collection_rate:.1f}%")
    
    # Sales Trends Over Time
    st.subheader("📈 Trends Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Daily Trends
        daily_metrics = analytics_df.groupby('Date').agg({
            'Sales Value': 'sum',
            'Paid Amount': 'sum',
            'Outstanding': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        fig_daily_trend = go.Figure()
        fig_daily_trend.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Sales Value'], 
                                           mode='lines', name='Sales', line=dict(color='#1f77b4')))
        fig_daily_trend.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Paid Amount'], 
                                           mode='lines', name='Collections', line=dict(color='#2ca02c')))
        fig_daily_trend.add_trace(go.Scatter(x=daily_metrics['Date'], y=daily_metrics['Outstanding'], 
                                           mode='lines', name='Outstanding', line=dict(color='#d62728')))
        fig_daily_trend.update_layout(title="Daily Sales vs Collections vs Outstanding Trend",
                                    xaxis_title="Date", yaxis_title="Amount (৳)")
        st.plotly_chart(fig_daily_trend, use_container_width=True)
    
    with col2:
        # Monthly Trends
        analytics_df['Month'] = analytics_df['Date'].dt.to_period('M').astype(str)
        monthly_metrics = analytics_df.groupby('Month').agg({
            'Sales Value': 'sum',
            'Paid Amount': 'sum',
            'Profit': 'sum'
        }).reset_index()
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(x=monthly_metrics['Month'], y=monthly_metrics['Sales Value'],
                                   name='Sales', marker_color='#1f77b4'))
        fig_monthly.add_trace(go.Bar(x=monthly_metrics['Month'], y=monthly_metrics['Paid Amount'],
                                   name='Collections', marker_color='#2ca02c'))
        fig_monthly.add_trace(go.Scatter(x=monthly_metrics['Month'], y=monthly_metrics['Profit'],
                                       mode='lines+markers', name='Profit', line=dict(color='#ff7f0e')))
        fig_monthly.update_layout(title="Monthly Performance", barmode='group',
                                xaxis_title="Month", yaxis_title="Amount (৳)")
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Executive Performance Analysis
    st.subheader("👨‍💼 Executive Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Executive Comparison
        exec_performance = analytics_df.groupby('Executive Name').agg({
            'Sales Value': 'sum',
            'Paid Amount': 'sum',
            'Outstanding': 'sum',
            'Profit': 'sum',
            'Customer Name': 'nunique'
        }).reset_index()
        
        exec_performance['Collection_Rate'] = (exec_performance['Paid Amount'] / exec_performance['Sales Value'] * 100)
        
        fig_exec_sales = px.bar(exec_performance, x='Executive Name', y=metric_choice,
                              title=f"{metric_choice} by Executive",
                              color=metric_choice,
                              color_continuous_scale='viridis')
        st.plotly_chart(fig_exec_sales, use_container_width=True)
    
    with col2:
        # Executive Efficiency
        fig_efficiency = px.scatter(exec_performance, x='Sales Value', y='Collection_Rate',
                                  size='Customer Name', color='Executive Name',
                                  title="Executive Efficiency: Sales vs Collection Rate",
                                  labels={'Sales Value': 'Total Sales (৳)', 'Collection_Rate': 'Collection Rate (%)'})
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Payment Method Analysis
    st.subheader("💳 Payment Method Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Payment Method Performance
        payment_analysis = analytics_df.groupby('Bank Name').agg({
            'Paid Amount': 'sum',
            'Sales Value': 'sum',
            'Customer Name': 'nunique',
            'Profit': 'sum'
        }).reset_index()
        
        payment_analysis['Collection_Rate'] = (payment_analysis['Paid Amount'] / payment_analysis['Sales Value'] * 100)
        
        fig_payment_bar = px.bar(payment_analysis, x='Bank Name', y='Collection_Rate',
                               title="Collection Rate by Payment Method",
                               color='Paid Amount',
                               color_continuous_scale='thermal')
        st.plotly_chart(fig_payment_bar, use_container_width=True)
    
    with col2:
        # Payment Distribution
        fig_payment_pie = px.pie(payment_analysis, values='Paid Amount', names='Bank Name',
                               title="Payment Distribution by Method",
                               hole=0.4)
        fig_payment_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_payment_pie, use_container_width=True)
    
    # Customer Analysis
    st.subheader("👥 Customer Behavior Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Customers
        top_customers = analytics_df.groupby('Customer Name').agg({
            'Sales Value': 'sum',
            'Paid Amount': 'sum',
            'Outstanding': 'sum',
            'Date': 'count'
        }).reset_index()
        top_customers.columns = ['Customer Name', 'Total Sales', 'Total Paid', 'Total Outstanding', 'Transaction Count']
        
        fig_top_customers = px.bar(top_customers.nlargest(10, 'Total Sales'), 
                                 x='Customer Name', y='Total Sales',
                                 title="Top 10 Customers by Sales Volume",
                                 color='Total Sales',
                                 color_continuous_scale='sunset')
        st.plotly_chart(fig_top_customers, use_container_width=True)
    
    with col2:
        # Customer Value vs Outstanding
        fig_customer_scatter = px.scatter(top_customers, x='Total Sales', y='Total Outstanding',
                                        size='Transaction Count', color='Total Paid',
                                        title="Customer Value Analysis: Sales vs Outstanding",
                                        hover_name='Customer Name',
                                        labels={'Total Sales': 'Total Sales ($)', 'Total Outstanding': 'Outstanding ($)'})
        st.plotly_chart(fig_customer_scatter, use_container_width=True)
    
    # Profitability Analysis
    st.subheader("💰 Profitability Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Profit Trend
        profit_trend = analytics_df.groupby('Date').agg({
            'Profit': 'sum',
            'Commission': 'sum'
        }).reset_index()
        
        fig_profit_trend = go.Figure()
        fig_profit_trend.add_trace(go.Scatter(x=profit_trend['Date'], y=profit_trend['Profit'],
                                            mode='lines', name='Profit', line=dict(color='#ff7f0e')))
        fig_profit_trend.add_trace(go.Scatter(x=profit_trend['Date'], y=profit_trend['Commission'],
                                            mode='lines', name='Commission', line=dict(color='#8c564b')))
        fig_profit_trend.update_layout(title="Daily Profit & Commission Trend",
                                     xaxis_title="Date", yaxis_title="Amount (৳)")
        st.plotly_chart(fig_profit_trend, use_container_width=True)
    
    with col2:
        # Profit by Executive
        profit_exec = analytics_df.groupby('Executive Name').agg({
            'Profit': 'sum',
            'Commission': 'sum',
            'Sales Value': 'sum'
        }).reset_index()
        profit_exec['Profit_Margin'] = (profit_exec['Profit'] / profit_exec['Sales Value'] * 100)
        
        fig_profit_margin = px.bar(profit_exec, x='Executive Name', y='Profit_Margin',
                                 title="Profit Margin by Executive (%)",
                                 color='Profit',
                                 color_continuous_scale='rainbow')
        st.plotly_chart(fig_profit_margin, use_container_width=True)
    
    # Advanced Analytics: Correlation Matrix
    st.subheader("🔍 Advanced Correlation Analysis")
    
    # Prepare correlation data
    numeric_cols = ['Sales Value', 'Paid Amount', 'Outstanding', 'Profit', 'Commission', 'Cashback']
    correlation_data = analytics_df[numeric_cols].corr()
    
    fig_corr = px.imshow(correlation_data,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title="Correlation Matrix: Key Metrics Relationship")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Key Insights Section
    st.subheader("💡 Key Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
        st.info("**Top Performing Executive**")
        top_exec = exec_performance.loc[exec_performance['Sales Value'].idxmax()]
        st.write(f"🏆 {top_exec['Executive Name']}")
        st.write(f"Sales: ৳{top_exec['Sales Value']:,.0f}")
        st.write(f"Collection Rate: {top_exec['Collection_Rate']:.1f}%")
        
        st.info("**Most Popular Payment Method**")
        top_payment = payment_analysis.loc[payment_analysis['Paid Amount'].idxmax()]
        st.write(f"💳 {top_payment['Bank Name']}")
        st.write(f"Amount: ${top_payment['Paid Amount']:,.0f}")
        st.write(f"Usage: {top_payment['Customer Name']} customers")
    
    with insights_col2:
        st.info("**Best Customer**")
        best_customer = top_customers.loc[top_customers['Total Sales'].idxmax()]
        st.write(f"👑 {best_customer['Customer Name']}")
        st.write(f"Lifetime Value: ${best_customer['Total Sales']:,.0f}")
        st.write(f"Transactions: {best_customer['Transaction Count']}")
        
        st.info("**Performance Summary**")
        st.write(f"📈 Total Sales: ৳{analytics_df['Sales Value'].sum():,.0f}")
        st.write(f"💰 Total Profit: ৳{analytics_df['Profit'].sum():,.0f}")
        st.write(f"🎯 Collection Rate: {collection_rate:.1f}%")
    
    # Download Analytics Report
    st.subheader("📥 Export Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary statistics
        summary_stats = analytics_df[numeric_cols].describe()
        csv_summary = summary_stats.to_csv().encode('utf-8')
        st.download_button("Download Summary Statistics", 
                         data=csv_summary,
                         file_name="analytics_summary.csv",
                         mime="text/csv")
    
    with col2:
        # Executive performance export
        exec_export = exec_performance[['Executive Name', 'Sales Value', 'Paid Amount', 'Collection_Rate', 'Profit']]
        csv_exec = exec_export.to_csv(index=False).encode('utf-8')
        st.download_button("Download Executive Performance", 
                         data=csv_exec,
                         file_name="executive_performance.csv",
                         mime="text/csv")
        
 
elif choice == "Data Management":
    st.subheader("✏️ Data Management Center")
    
    # Create tabs for different editing functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["📋 View & Edit Records", "➕ Add New Records", "🗑️ Delete Records", "🔄 Bulk Operations"])
    
    with tab1:
        st.subheader("📋 View and Edit Existing Records")
        
        # Search and Filter Options
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search_term = st.text_input("🔍 Search by Customer Name", placeholder="Enter customer name...")
        with col2:
            exec_filter = st.selectbox("Filter by Executive", options=['All'] + list(df['Executive Name'].unique()))
        with col3:
            date_filter = st.date_input("Filter by Date", value=None)
        with col4:
            records_per_page = st.selectbox("Records per page", [10, 25, 50, 100], index=1)
        
        # Apply filters
        filtered_edit_df = df.copy()
        if search_term:
            filtered_edit_df = filtered_edit_df[filtered_edit_df['Customer Name'].str.contains(search_term, case=False, na=False)]
        if exec_filter != 'All':
            filtered_edit_df = filtered_edit_df[filtered_edit_df['Executive Name'] == exec_filter]
        if date_filter:
            filtered_edit_df = filtered_edit_df[filtered_edit_df['Date'] == pd.to_datetime(date_filter)]
        
        # Pagination
        total_records = len(filtered_edit_df)
        total_pages = (total_records + records_per_page - 1) // records_per_page
        current_page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        
        start_idx = (current_page - 1) * records_per_page
        end_idx = min(start_idx + records_per_page, total_records)
        
        st.info(f"Showing records {start_idx + 1} to {end_idx} of {total_records} total records")
        
        if total_records > 0:
            # Display current page records
            current_records = filtered_edit_df.iloc[start_idx:end_idx].reset_index(drop=True)
            
            # Create editable dataframe
            st.subheader("✏️ Edit Records (Double-click to edit)")
            
            # Create a copy for editing
            edited_df = current_records.copy()
            
            # Display editable dataframe
            for idx, row in current_records.iterrows():
                with st.expander(f"📝 Edit: {row['Customer Name']} - {row['Order no']} - {row['Date'].strftime('%Y-%m-%d')}", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        new_date = st.date_input("Date", value=row['Date'].date(), key=f"date_{idx}")
                        new_order_no = st.text_input("Order No", value=row['Order no'], key=f"order_{idx}")
                        new_customer = st.text_input("Customer Name", value=row['Customer Name'], key=f"customer_{idx}")
                        new_executive = st.selectbox("Executive", options=df['Executive Name'].unique(), 
                                                   index=list(df['Executive Name'].unique()).index(row['Executive Name']) 
                                                   if row['Executive Name'] in df['Executive Name'].unique() else 0, 
                                                   key=f"exec_{idx}")
                    
                    with col2:
                        new_area = st.text_input("Area Code", value=row['Area code'], key=f"area_{idx}")
                        new_opening = st.number_input("Opening Balance", value=float(row['Opening balance']), key=f"open_{idx}")
                        new_sales = st.number_input("Sales Value", value=float(row['Sales Value']), key=f"sales_{idx}")
                        new_sales_ret = st.number_input("Sales Return", value=float(row['Sales return']), key=f"ret_{idx}")
                    
                    with col3:
                        new_sales_transfer = st.number_input("Sales Transfer", value=float(row['Sales transfer']), key=f"trans_{idx}")
                        new_paid = st.number_input("Paid Amount", value=float(row['Paid Amount']), key=f"paid_{idx}")
                        new_cashback = st.number_input("Cashback", value=float(row['Cashback']), key=f"cash_{idx}")
                        new_bank = st.selectbox("Bank Name", 
                                              options=['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 'MBL WB CC', 'MBL WB CD',
                                                       'Cash', 'BRAC BANK', 'Credit Card', 'Debit Card', 'Cheque To MBL CD'],
                                              index=['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 'MBL WB CC', 'MBL WB CD',
                                                     'Cash', 'BRAC BANK', 'Credit Card', 'Debit Card', 'Cheque To MBL CD'].index(row['Bank Name'])
                                              if row['Bank Name'] in ['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 'MBL WB CC', 'MBL WB CD',
                                                                     'Cash', 'BRAC BANK', 'Credit Card', 'Debit Card', 'Cheque To MBL CD'] else 0,
                                              key=f"bank_{idx}")
                    
                    # Calculate new outstanding
                    new_outstanding = new_opening + new_sales + new_sales_transfer - new_sales_ret - new_paid - new_cashback
                    new_commission = new_paid * 0.01
                    new_profit = new_paid * 0.25
                    
                    st.info(f"Calculated Outstanding: ${new_outstanding:,.2f} | Commission: ${new_commission:,.2f} | Profit: ${new_profit:,.2f}")
                    
                    col1, col2, col3 = st.columns(3)
                    with col2:
                        if st.button("💾 Save Changes", key=f"save_{idx}", use_container_width=True):
                            # Update the record in the main dataframe
                            original_index = df[df['Order no'] == row['Order no']].index[0]
                            
                            df.at[original_index, 'Date'] = pd.to_datetime(new_date)
                            df.at[original_index, 'Order no'] = new_order_no
                            df.at[original_index, 'Customer Name'] = new_customer
                            df.at[original_index, 'Executive Name'] = new_executive
                            df.at[original_index, 'Area code'] = new_area
                            df.at[original_index, 'Opening balance'] = new_opening
                            df.at[original_index, 'Sales Value'] = new_sales
                            df.at[original_index, 'Sales return'] = new_sales_ret
                            df.at[original_index, 'Sales transfer'] = new_sales_transfer
                            df.at[original_index, 'Paid Amount'] = new_paid
                            df.at[original_index, 'Cashback'] = new_cashback
                            df.at[original_index, 'Bank Name'] = new_bank
                            df.at[original_index, 'Outstanding'] = new_outstanding
                            df.at[original_index, 'Commission'] = new_commission
                            df.at[original_index, 'Profit'] = new_profit
                            
                            st.success(f"✅ Record updated successfully for Order: {new_order_no}")
                            st.rerun()
        
        else:
            st.warning("No records found matching your filters.")
    
    with tab2:
        st.subheader("➕ Add New Sales Records")
        
        with st.form("add_new_record_form", clear_on_submit=True):
            st.markdown("### Enter New Sale Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                new_date = st.date_input("Date", datetime.now())
                new_order_no = st.text_input("Order No*", value=f"ORD{len(df) + 1000}")
                new_customer = st.text_input("Customer Name*", placeholder="Enter customer name")
                new_executive = st.selectbox("Executive Name*", options=df['Executive Name'].unique())
                new_area = st.text_input("Area Code", placeholder="Enter area code")
            
            with col2:
                new_opening = st.number_input("Opening Balance", min_value=0.0, value=0.0, step=100.0)
                new_sales_val = st.number_input("Sales Value*", min_value=0.0, value=0.0, step=100.0)
                new_sales_ret = st.number_input("Sales Return", min_value=0.0, value=0.0, step=10.0)
                new_sales_transfer = st.number_input("Sales Transfer", min_value=0.0, value=0.0, step=10.0)
                new_paid = st.number_input("Paid Amount*", min_value=0.0, value=0.0, step=100.0)
                new_cashback = st.number_input("Cashback", min_value=0.0, value=0.0, step=10.0)
                new_bank_name = st.selectbox("Payment Method/Bank*", 
                                           options=['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 'MBL WB CC', 'MBL WB CD',
                                                    'Cash', 'BRAC BANK', 'Credit Card', 'Debit Card', 'Cheque To MBL CD'])
            
            # Calculate derived fields
            new_outstanding = new_opening + new_sales_val + new_sales_transfer - new_sales_ret - new_paid - new_cashback
            new_commission = new_paid * 0.01
            new_profit = new_paid * 0.25
            
            st.markdown("### Calculated Values")
            col1, col2, col3 = st.columns(3)
            col1.metric("Outstanding", f"${new_outstanding:,.2f}")
            col2.metric("Commission", f"${new_commission:,.2f}")
            col3.metric("Profit", f"${new_profit:,.2f}")
            
            submitted = st.form_submit_button("🚀 Add New Record", use_container_width=True)
            
            if submitted:
                if not new_customer or not new_order_no or new_sales_val == 0:
                    st.error("Please fill in all required fields (*)")
                else:
                    new_entry = {
                        'Date': new_date,
                        'Order no': new_order_no,
                        'Customer Name': new_customer,
                        'Executive Name': new_executive,
                        'Area code': new_area,
                        'Opening balance': new_opening,
                        'Sales Value': new_sales_val,
                        'Sales return': new_sales_ret,
                        'Sales transfer': new_sales_transfer,
                        'Paid Amount': new_paid,
                        'Cashback': new_cashback,
                        'Bank Name': new_bank_name,
                        'Outstanding': new_outstanding,
                        'Commission': new_commission,
                        'Profit': new_profit
                    }
                    
                    df.loc[len(df)] = new_entry
                    st.success("🎉 New record added successfully!")
                    st.balloons()
    
    with tab3:
        st.subheader("🗑️ Delete Records")
        st.warning("⚠️ This action cannot be undone. Please be cautious.")
        
        col1, col2 = st.columns(2)
        with col1:
            delete_search = st.text_input("Search record to delete", placeholder="Enter Order No or Customer Name")
        with col2:
            delete_method = st.radio("Delete by", ["Order Number", "Customer Name"])
        
        if delete_search:
            if delete_method == "Order Number":
                records_to_delete = df[df['Order no'].str.contains(delete_search, case=False, na=False)]
            else:
                records_to_delete = df[df['Customer Name'].str.contains(delete_search, case=False, na=False)]
            
            if len(records_to_delete) > 0:
                st.subheader("📋 Records Found for Deletion")
                st.dataframe(records_to_delete[['Order no', 'Customer Name', 'Date', 'Sales Value', 'Paid Amount']])
                
                st.error(f"Found {len(records_to_delete)} record(s) matching your search.")
                
                if st.button("🗑️ Delete All Matching Records", type="secondary"):
                    df.drop(records_to_delete.index, inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    st.success(f"✅ {len(records_to_delete)} record(s) deleted successfully!")
                    st.rerun()
                
                # Single record deletion
                if len(records_to_delete) == 1:
                    record = records_to_delete.iloc[0]
                    if st.button(f"Delete only: {record['Order no']} - {record['Customer Name']}", type="primary"):
                        df.drop(records_to_delete.index, inplace=True)
                        df.reset_index(drop=True, inplace=True)
                        st.success("✅ Record deleted successfully!")
                        st.rerun()
            else:
                st.info("No records found matching your search.")
    
    with tab4:
        st.subheader("🔄 Bulk Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📥 Import Data")
            uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        new_data = pd.read_csv(uploaded_file)
                    else:
                        new_data = pd.read_excel(uploaded_file)
                    
                    st.success(f"✅ File loaded successfully! {len(new_data)} records found.")
                    st.dataframe(new_data.head())
                    
                    # Check required columns
                    required_cols = ['Date', 'Order no', 'Customer Name', 'Executive Name', 'Sales Value', 'Paid Amount']
                    missing_cols = [col for col in required_cols if col not in new_data.columns]
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(missing_cols)}")
                    else:
                        if st.button("📤 Append to Existing Data", use_container_width=True):
                            # Add missing columns if necessary
                            for col in df.columns:
                                if col not in new_data.columns:
                                    if col in ['Outstanding', 'Commission', 'Profit']:
                                        # Calculate these columns
                                        if 'Opening balance' not in new_data.columns:
                                            new_data['Opening balance'] = 0
                                        if 'Sales return' not in new_data.columns:
                                            new_data['Sales return'] = 0
                                        if 'Sales transfer' not in new_data.columns:
                                            new_data['Sales transfer'] = 0
                                        if 'Cashback' not in new_data.columns:
                                            new_data['Cashback'] = 0
                                        
                                        new_data['Outstanding'] = (new_data['Opening balance'] + 
                                                                 new_data['Sales Value'] + 
                                                                 new_data['Sales transfer'] - 
                                                                 new_data['Sales return'] - 
                                                                 new_data['Paid Amount'] - 
                                                                 new_data['Cashback'])
                                        new_data['Commission'] = new_data['Paid Amount'] * 0.01
                                        new_data['Profit'] = new_data['Paid Amount'] * 0.25
                                    else:
                                        new_data[col] = None
                            
                            # Append to main dataframe
                            df = pd.concat([df, new_data], ignore_index=True)
                            st.success(f"✅ {len(new_data)} records added successfully! Total records: {len(df)}")
                
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            st.markdown("### 📊 Data Management")
            
            # Data summary
            st.info(f"**Current Data Summary:**")
            st.write(f"• Total Records: {len(df)}")
            st.write(f"• Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
            st.write(f"• Unique Customers: {df['Customer Name'].nunique()}")
            st.write(f"• Unique Executives: {df['Executive Name'].nunique()}")
            
            # Data cleanup options
            st.markdown("### 🧹 Data Cleanup")
            
            if st.button("🔄 Remove Duplicate Orders", use_container_width=True):
                duplicates = df[df.duplicated('Order no', keep=False)]
                if len(duplicates) > 0:
                    st.warning(f"Found {len(duplicates)} duplicate orders")
                    st.dataframe(duplicates[['Order no', 'Customer Name', 'Date']])
                    if st.button("✅ Confirm Remove Duplicates"):
                        df = df.drop_duplicates('Order no', keep='first')
                        st.success("✅ Duplicates removed successfully!")
                else:
                    st.success("✅ No duplicate orders found!")
            
            if st.button("📝 Recalculate All Values", use_container_width=True):
                df['Outstanding'] = (df['Opening balance'] + df['Sales Value'] + 
                                   df['Sales transfer'] - df['Sales return'] - 
                                   df['Paid Amount'] - df['Cashback'])
                df['Commission'] = df['Paid Amount'] * 0.01
                df['Profit'] = df['Paid Amount'] * 0.25
                st.success("✅ All values recalculated successfully!")
    
    # Sidebar with quick stats
    with st.sidebar:
        st.markdown("---")
        st.subheader("📈 Data Statistics")
        st.metric("Total Records", len(df))
        st.metric("Unique Customers", df['Customer Name'].nunique())
        st.metric("Unique Executives", df['Executive Name'].nunique())
        st.metric("Data Size", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        st.markdown("---")
        st.subheader("💡 Quick Actions")
        if st.button("🔄 Refresh Data View", use_container_width=True):
            st.rerun()
        if st.button("📥 Download Current Data", use_container_width=True):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"sales_data_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )

elif choice == "Edit Data":
    st.subheader("✏️ Excel-Style Data Editor")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Spreadsheet View", "⚙️ Advanced Editing"])
    
    with tab1:
        st.subheader("📊 Excel-Style Data Grid")
        st.info("💡 **Tip**: Click on any cell to edit. Changes are saved automatically when you move to another cell.")
        
        # Display data statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Customers", df['Customer Name'].nunique())
        with col3:
            st.metric("Total Executives", df['Executive Name'].nunique())
        with col4:
            st.metric("Data Size", f"{(df.memory_usage(deep=True).sum() / 1024 / 1024):.1f} MB")
        
        # Search and filter options
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            search_term = st.text_input("🔍 Search across all columns", placeholder="Search...")
        with col2:
            show_columns = st.multiselect(
                "Columns to show",
                options=df.columns.tolist(),
                default=df.columns.tolist()
            )
        with col3:
            rows_per_page = st.selectbox("Rows per page", [10, 25, 50, 100, 200], index=2)
        with col4:
            st.write("")  # Spacer
            if st.button("🔄 Refresh View"):
                st.rerun()
        
        # Apply search filter
        filtered_df = df.copy()
        if search_term:
            mask = pd.DataFrame(columns=df.columns)
            for col in df.columns:
                try:
                    mask[col] = df[col].astype(str).str.contains(search_term, case=False, na=False)
                except:
                    continue
            filtered_df = df[mask.any(axis=1)]
        
        # Select only the columns to show
        filtered_df = filtered_df[show_columns]
        
        # Display the editable dataframe
        st.subheader("📝 Editable Data Grid")
        
        # Use st.data_editor for Excel-like editing
        edited_df = st.data_editor(
            filtered_df,
            key="data_editor",
            num_rows="dynamic",
            use_container_width=True,
            height=600,
            column_config={
                "Date": st.column_config.DateColumn(
                    "Date",
                    format="YYYY-MM-DD",
                    required=True,
                ),
                "Order no": st.column_config.TextColumn(
                    "Order No",
                    required=True,
                    max_chars=50,
                ),
                "Customer Name": st.column_config.TextColumn(
                    "Customer Name",
                    required=True,
                    max_chars=100,
                ),
                "Executive Name": st.column_config.SelectboxColumn(
                    "Executive Name",
                    options=df['Executive Name'].unique().tolist(),
                    required=True,
                ),
                "Bank Name": st.column_config.SelectboxColumn(
                    "Payment Method",
                    options=['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 'MBL WB CC', 'MBL WB CD',
                            'Cash', 'BRAC BANK', 'Credit Card', 'Debit Card', 'Cheque To MBL CD'],
                    required=True,
                ),
                "Sales Value": st.column_config.NumberColumn(
                    "Sales Value",
                    format="$%.2f",
                    min_value=0.0,
                ),
                "Paid Amount": st.column_config.NumberColumn(
                    "Paid Amount",
                    format="$%.2f",
                    min_value=0.0,
                ),
                "Opening balance": st.column_config.NumberColumn(
                    "Opening Balance",
                    format="$%.2f",
                ),
                "Sales return": st.column_config.NumberColumn(
                    "Sales Return",
                    format="$%.2f",
                    min_value=0.0,
                ),
                "Sales transfer": st.column_config.NumberColumn(
                    "Sales Transfer",
                    format="$%.2f",
                    min_value=0.0,
                ),
                "Cashback": st.column_config.NumberColumn(
                    "Cashback",
                    format="$%.2f",
                    min_value=0.0,
                ),
                "Outstanding": st.column_config.NumberColumn(
                    "Outstanding",
                    format="$%.2f",
                ),
                "Commission": st.column_config.NumberColumn(
                    "Commission",
                    format="$%.2f",
                    min_value=0.0,
                ),
                "Profit": st.column_config.NumberColumn(
                    "Profit",
                    format="$%.2f",
                ),
                "Area code": st.column_config.TextColumn(
                    "Area Code",
                    max_chars=20,
                )
            },
            disabled=["Outstanding", "Commission", "Profit"]  # These are calculated automatically
        )
        
        # Action buttons row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            if st.button("💾 Save Changes", use_container_width=True):
                # Update the main dataframe with changes
                for idx, row in edited_df.iterrows():
                    original_idx = df.index[idx]  # Get the original index
                    df.loc[original_idx] = row
                
                # Recalculate derived columns
                df['Outstanding'] = (df['Opening balance'] + df['Sales Value'] + 
                                   df['Sales transfer'] - df['Sales return'] - 
                                   df['Paid Amount'] - df['Cashback'])
                df['Commission'] = df['Paid Amount'] * 0.01
                df['Profit'] = df['Paid Amount'] * 0.25
                
                st.success("✅ All changes saved successfully!")
                st.balloons()
        
        with col2:
            if st.button("🔄 Recalculate All", use_container_width=True):
                df['Outstanding'] = (df['Opening balance'] + df['Sales Value'] + 
                                   df['Sales transfer'] - df['Sales return'] - 
                                   df['Paid Amount'] - df['Cashback'])
                df['Commission'] = df['Paid Amount'] * 0.01
                df['Profit'] = df['Paid Amount'] * 0.25
                st.success("✅ All values recalculated!")
                st.rerun()
        
        with col3:
            if st.button("📥 Export Current View", use_container_width=True):
                csv = edited_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"data_export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col4:
            if st.button("➕ Add New Row", use_container_width=True):
                # Add a new empty row
                new_row = {
                    'Date': pd.Timestamp.now(),
                    'Order no': f"NEW_{len(df) + 1}",
                    'Customer Name': "New Customer",
                    'Executive Name': df['Executive Name'].iloc[0] if len(df) > 0 else "Executive",
                    'Area code': "",
                    'Opening balance': 0.0,
                    'Sales Value': 0.0,
                    'Sales return': 0.0,
                    'Sales transfer': 0.0,
                    'Paid Amount': 0.0,
                    'Cashback': 0.0,
                    'Bank Name': 'Cash',
                    'Outstanding': 0.0,
                    'Commission': 0.0,
                    'Profit': 0.0
                }
                df.loc[len(df)] = new_row
                st.success("✅ New row added! Scroll down to see it.")
                st.rerun()
        
        with col5:
            if st.button("🗑️ Delete Selected", use_container_width=True):
                st.warning("⚠️ Select rows by checking the checkbox column, then click delete.")
                # This would require additional implementation for row selection
        
        # Display changes summary
        st.subheader("📋 Changes Summary")
        if not filtered_df.equals(df[show_columns].iloc[:len(filtered_df)]):
            st.info("🔄 You have unsaved changes. Click 'Save Changes' to apply them.")
        else:
            st.success("✅ All changes are saved.")
    
    with tab2:
        st.subheader("⚙️ Advanced Editing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Bulk Operations")
            
            # Bulk update by condition
            st.subheader("Bulk Update Values")
            update_column = st.selectbox("Select column to update", df.select_dtypes(include=[np.number]).columns.tolist())
            update_value = st.number_input("New value", value=0.0)
            
            update_condition_col = st.selectbox("Where column", df.columns.tolist())
            update_condition_val = st.text_input("Equals value")
            
            if st.button("Apply Bulk Update", use_container_width=True):
                if update_condition_val:
                    mask = df[update_condition_col].astype(str) == str(update_condition_val)
                    df.loc[mask, update_column] = update_value
                    st.success(f"✅ Updated {mask.sum()} records")
                else:
                    df[update_column] = update_value
                    st.success(f"✅ Updated all records in {update_column}")
            
            # Data validation
            st.subheader("🔍 Data Validation")
            if st.button("Check Data Quality", use_container_width=True):
                issues = []
                
                # Check for missing values
                missing_data = df.isnull().sum()
                if missing_data.sum() > 0:
                    issues.append(f"Missing values found in {missing_data[missing_data > 0].to_dict()}")
                
                # Check for negative sales
                negative_sales = (df['Sales Value'] < 0).sum()
                if negative_sales > 0:
                    issues.append(f"Negative sales values: {negative_sales} records")
                
                # Check for duplicate order numbers
                duplicates = df[df.duplicated('Order no', keep=False)]
                if len(duplicates) > 0:
                    issues.append(f"Duplicate order numbers: {len(duplicates)} records")
                
                if issues:
                    st.error("❌ Data quality issues found:")
                    for issue in issues:
                        st.write(f"- {issue}")
                else:
                    st.success("✅ No data quality issues found!")
        
        with col2:
            st.markdown("### 📊 Data Management")
            
            # Import new data
            st.subheader("📥 Import Data")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(new_data)} records")
                    
                    # Show preview
                    st.dataframe(new_data.head())
                    
                    if st.button("Merge with Existing Data", use_container_width=True):
                        # Ensure required columns exist
                        required_cols = ['Date', 'Order no', 'Customer Name', 'Sales Value', 'Paid Amount']
                        missing_cols = [col for col in required_cols if col not in new_data.columns]
                        
                        if missing_cols:
                            st.error(f"Missing required columns: {', '.join(missing_cols)}")
                        else:
                            # Calculate derived columns if missing
                            if 'Outstanding' not in new_data.columns:
                                new_data['Outstanding'] = (new_data.get('Opening balance', 0) + 
                                                         new_data['Sales Value'] + 
                                                         new_data.get('Sales transfer', 0) - 
                                                         new_data.get('Sales return', 0) - 
                                                         new_data['Paid Amount'] - 
                                                         new_data.get('Cashback', 0))
                            if 'Commission' not in new_data.columns:
                                new_data['Commission'] = new_data['Paid Amount'] * 0.01
                            if 'Profit' not in new_data.columns:
                                new_data['Profit'] = new_data['Paid Amount'] * 0.25
                            
                            df = pd.concat([df, new_data], ignore_index=True)
                            st.success(f"✅ Data merged successfully! Total records: {len(df)}")
                            
                except Exception as e:
                    st.error(f"Error reading file: {e}")
            
            # Data cleanup
            st.subheader("🧹 Data Cleanup")
            if st.button("Remove Duplicate Orders", use_container_width=True):
                initial_count = len(df)
                df = df.drop_duplicates('Order no', keep='first')
                removed = initial_count - len(df)
                if removed > 0:
                    st.success(f"✅ Removed {removed} duplicate orders")
                else:
                    st.info("✅ No duplicate orders found")
            
            if st.button("Fix Data Types", use_container_width=True):
                # Ensure proper data types
                numeric_cols = ['Opening balance', 'Sales Value', 'Sales return', 'Sales transfer', 
                              'Paid Amount', 'Cashback', 'Outstanding', 'Commission', 'Profit']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                
                st.success("✅ Data types fixed!")
    
    # Sidebar with quick actions and stats
    with st.sidebar:
        st.markdown("---")
        st.subheader("📈 Quick Stats")
        st.write(f"**Total Records:** {len(df)}")
        st.write(f"**Date Range:** {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"**Total Sales:** ${df['Sales Value'].sum():,.2f}")
        st.write(f"**Total Outstanding:** ${df['Outstanding'].sum():,.2f}")
        
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        if st.button("Download Full Dataset", use_container_width=True):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"full_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        if st.button("Create Backup", use_container_width=True):
            # Create a backup in session state
            st.session_state.data_backup = df.copy()
            st.success("✅ Backup created!")
        
        if st.button("Restore Backup", use_container_width=True) and hasattr(st.session_state, 'data_backup'):
            df = st.session_state.data_backup.copy()
            st.success("✅ Backup restored!")
            st.rerun()
             
        

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.info("💡 **Tips:** Use filters to analyze specific data segments. Export reports for external analysis.")

# Add a footer fixed at the bottom
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #16aadb;
        color: #e4e4ed;
        text-align: center;
        padding: 10px 0;
        font-size: 16px;
        box-shadow: 0 -1px 5px rgba(0,0,0,0.1);
    }
    </style>
    <div class="footer">
        © 2025 Mujakkir Ahmad | All Rights Reserved
    </div>
    """,
    unsafe_allow_html=True
)
