import streamlit as st
import pandas as pd
from datetime import datetime

from styles import inject_theme, hero, bubble, toast_success, toast_error
from utils import (get_df, persist, VAT_RATE,
                   CUSTOMER_TYPES, SALES_ZONES, SALES_CHANNELS,
                   PAYMENT_METHODS, BANKS, EXECUTIVES)

st.set_page_config(page_title="Sales Entry", page_icon="📝", layout="wide")
inject_theme()

hero("Sales Entry", "Create a new transaction — calculations update live.", icon="📝")

df = get_df()

# ---------- Chat-style intro bubble ----------
bubble("👋 Hi! Fill in the form below. "
       "I'll calculate <b>Sales Amount, VAT, Net Sales</b> and "
       "<b>Outstanding</b> as you type.", role="bot")

with st.form("add_sale", clear_on_submit=True):
    with st.expander("🧾 Invoice Information", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            date     = st.date_input("Date", datetime.now())
            txn_id   = st.text_input("Transaction ID", value=f"TXN{len(df)+100000}")
            inv_no   = st.text_input("Invoice No",     value=f"INV{len(df)+1000}")
        with c2:
            customer = st.text_input("Customer Name")
            ctype    = st.selectbox("Customer Type", CUSTOMER_TYPES)
            executive= st.selectbox("Executive",     EXECUTIVES)
        with c3:
            zone     = st.selectbox("Sales Zone",    SALES_ZONES)
            channel  = st.selectbox("Sales Channel", SALES_CHANNELS)
            remarks  = st.text_area("Remarks", height=68)

    with st.expander("💰 Amounts & Payment", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            inv_val = st.number_input("Invoice Value", min_value=0.0, value=0.0)
            disc    = st.number_input("Discount",      min_value=0.0, value=0.0)
        with c2:
            sret     = st.number_input("Sales Return",    min_value=0.0, value=0.0)
            credited = st.number_input("Credited Amount", min_value=0.0, value=0.0)
        with c3:
            pm   = st.selectbox("Payment Method", PAYMENT_METHODS)
            bank = st.selectbox("Bank Name",      BANKS)

    # ---------- Live calculations ----------
    sales_amount = max(inv_val - disc, 0)
    sales_vat    = sales_amount * VAT_RATE
    net_sales    = sales_amount + sales_vat - sret
    outstanding  = net_sales - credited

    st.markdown("##### 🧮 Live Calculation")
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Sales Amount", f"৳ {sales_amount:,.2f}")
    p2.metric("VAT (5%)",     f"৳ {sales_vat:,.2f}")
    p3.metric("Net Sales",    f"৳ {net_sales:,.2f}")
    p4.metric("Outstanding",  f"৳ {outstanding:,.2f}")

    submitted = st.form_submit_button("✅ Save Sale", use_container_width=True)

    if submitted:
        new_row = {
            'Date':            pd.to_datetime(date),
            'Transaction ID':  txn_id,
            'Invoice No':      inv_no,
            'Customer Name':   customer or 'Unknown',
            'Customer Type':   ctype,
            'Executive':       executive,
            'Sales Zone':      zone,
            'Sales Channel':   channel,
            'Invoice Value':   inv_val,
            'Discount':        disc,
            'Sales Amount':    sales_amount,
            'Sales VAT':       sales_vat,
            'Sales Return':    sret,
            'Credited Amount': credited,
            'Payment Method':  pm,
            'Bank Name':       bank,
            'Remarks':         remarks,
            'Month Name':      pd.to_datetime(date).strftime('%B %Y')
        }
        persist(pd.concat([st.session_state.df, pd.DataFrame([new_row])],
                          ignore_index=True))
        bubble(f"✅ Sale <b>{inv_no}</b> saved! "
               f"Outstanding recorded as ৳{outstanding:,.2f}.", role="bot")
        toast_success(f"Sale {inv_no} added")
        st.balloons()

st.markdown("---")
with st.expander("🕓 Recently Added (last 10)", expanded=True):
    st.dataframe(st.session_state.df.tail(10), use_container_width=True)