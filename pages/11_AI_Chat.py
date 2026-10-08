"""
pages/11_AI_Chat.py — Chat with your sales data.
  • Rule-based assistant (works out of the box, no API key)
  • Optional OpenAI integration if OPENAI_API_KEY is set
"""
import os
import re
import streamlit as st
import pandas as pd
import plotly.express as px

from styles import inject_theme, hero
from utils import get_df

st.set_page_config(page_title="AI Chat", page_icon="🤖", layout="wide")
inject_theme()
hero("AI Chat with Data", "Ask questions in plain English about your sales data.", icon="🤖")

df = get_df()

# =====================================================================
# Session state
# =====================================================================
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant",
         "content": "👋 Hi! I'm your sales assistant. Try asking me:\n"
                    "- *What is the total sales?*\n"
                    "- *Top 5 customers by sales*\n"
                    "- *Show sales by executive*\n"
                    "- *What is the total outstanding?*\n"
                    "- *Sales by zone*"}
    ]

# =====================================================================
# Rule-based engine
# =====================================================================
def answer(q: str):
    """Return (text, plotly_fig or None)."""
    ql = q.lower().strip()

    total_sales     = df['Sales Amount'].sum()
    total_credited  = df['Credited Amount'].sum()
    total_out       = df['Outstanding'].sum()
    total_profit    = df['Profit'].sum()
    total_commission= df['Commission'].sum()
    n_customers     = df['Customer Name'].nunique()
    n_exec          = df['Executive'].nunique()

    # ---------- Total sales ----------
    if any(k in ql for k in ['total sales', 'overall sales', 'how much sales']):
        return f"💰 **Total sales** = ৳ {total_sales:,.0f}", None

    # ---------- Total outstanding ----------
    if 'outstanding' in ql and not any(k in ql for k in ['top', 'highest', 'by customer']):
        return f"📊 **Total outstanding** = ৳ {total_out:,.0f}", None

    # ---------- Total credited / collected ----------
    if any(k in ql for k in ['total credited', 'total collected', 'total paid']):
        return f"💳 **Total credited** = ৳ {total_credited:,.0f}", None

    # ---------- Profit ----------
    if 'profit' in ql:
        return f"📈 **Total profit** = ৳ {total_profit:,.0f}", None

    # ---------- Commission ----------
    if 'commission' in ql:
        return f"👨‍💼 **Total commission** = ৳ {total_commission:,.0f}", None

    # ---------- Counts ----------
    if 'how many customer' in ql or 'number of customer' in ql:
        return f"👥 **{n_customers}** unique customers.", None
    if 'how many executive' in ql or 'number of executive' in ql:
        return f"👨‍💼 **{n_exec}** executives.", None

    # ---------- Top N customers ----------
    m = re.search(r'top\s+(\d+)?\s*customer', ql)
    if m:
        n = int(m.group(1)) if m.group(1) else 5
        top = (df.groupby('Customer Name')['Sales Amount'].sum()
                 .nlargest(n).reset_index())
        fig = px.bar(top, x='Sales Amount', y='Customer Name',
                     orientation='h', title=f"Top {n} customers by sales",
                     color='Sales Amount', color_continuous_scale='viridis')
        return f"🏆 Here are the **top {n} customers** by sales:", fig

    # ---------- Top executives ----------
    m = re.search(r'top\s+(\d+)?\s*(executive|exec)', ql)
    if m:
        n = int(m.group(1)) if m.group(1) else 5
        top = (df.groupby('Executive')['Sales Amount'].sum()
                 .nlargest(n).reset_index())
        fig = px.bar(top, x='Executive', y='Sales Amount',
                     title=f"Top {n} executives by sales",
                     color='Sales Amount', color_continuous_scale='plasma')
        return f"🏆 Top {n} executives:", fig

    # ---------- Grouped charts: by <dimension> ----------
    dims = {
        'executive': 'Executive',
        'zone':      'Sales Zone',
        'channel':   'Sales Channel',
        'bank':      'Bank Name',
        'payment':   'Payment Method',
        'customer type': 'Customer Type',
    }
    for kw, col in dims.items():
        if f'by {kw}' in ql or f'{kw} breakdown' in ql:
            grp = (df.groupby(col)['Sales Amount'].sum()
                     .reset_index().sort_values('Sales Amount', ascending=False))
            fig = px.bar(grp, x=col, y='Sales Amount',
                         title=f"Sales by {col}", color='Sales Amount',
                         color_continuous_scale='blues')
            return f"📊 Sales by **{col}**:", fig

    # ---------- Highest outstanding customer ----------
    if any(k in ql for k in ['highest outstanding customer', 'top outstanding customer']):
        top = (df.groupby('Customer Name')['Outstanding'].sum()
                 .nlargest(10).reset_index())
        fig = px.bar(top, x='Customer Name', y='Outstanding',
                     title="Top 10 by outstanding",
                     color='Outstanding', color_continuous_scale='reds')
        return "🚨 Customers with highest outstanding:", fig

    # ---------- Monthly trend ----------
    if any(k in ql for k in ['monthly trend', 'monthly sales', 'trend']):
        monthly = (df.set_index('Date')['Sales Amount']
                     .resample('M').sum().reset_index())
        fig = px.line(monthly, x='Date', y='Sales Amount',
                      title="Monthly Sales Trend", markers=True)
        return "📈 Monthly sales trend:", fig

    # ---------- Help / fallback ----------
    return ("I'm a rule-based assistant. Try:\n"
            "- *total sales*\n"
            "- *total outstanding*\n"
            "- *top 5 customers*\n"
            "- *top 3 executives*\n"
            "- *sales by executive / zone / channel*\n"
            "- *monthly trend*\n"
            "- *how many customers*"), None


# =====================================================================
# Optional OpenAI integration
# =====================================================================
def openai_answer(question: str, history: list):
    try:
        from openai import OpenAI
    except ImportError:
        return "❌ Install `openai` to use the LLM.", None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None

    client = OpenAI(api_key=api_key)
    summary = (
        f"Sales data summary:\n"
        f"- Rows: {len(df)}\n"
        f"- Total sales: {df['Sales Amount'].sum():,.0f}\n"
        f"- Total credited: {df['Credited Amount'].sum():,.0f}\n"
        f"- Total outstanding: {df['Outstanding'].sum():,.0f}\n"
        f"- Customers: {df['Customer Name'].nunique()}\n"
        f"- Executives: {df['Executive'].nunique()}\n"
        f"- Date range: {df['Date'].min().date()} → {df['Date'].max().date()}\n"
    )
    messages = [{"role": "system",
                 "content": "You are a helpful sales analyst. "
                            "Answer concisely using only the data summary."},
                {"role": "user", "content": summary}]
    for h in history[-6:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": question})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, temperature=0.3, max_tokens=500)
        return resp.choices[0].message.content, None
    except Exception as e:
        return f"LLM error: {e}", None


# =====================================================================
# Sidebar toggle
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ AI Settings")
    use_llm = st.toggle("Use OpenAI (if API key set)", value=False)
    if st.button("🗑️ Clear chat"):
        st.session_state.chat_history = [st.session_state.chat_history[0]]
        st.rerun()

# =====================================================================
# Render conversation
# =====================================================================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "fig" in msg and msg["fig"] is not None:
            st.plotly_chart(msg["fig"], use_container_width=True)

# =====================================================================
# Input
# =====================================================================
prompt = st.chat_input("Ask something about your sales data…")

if prompt:
    # user
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # assistant
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            text, fig = None, None
            if use_llm:
                text, fig = openai_answer(prompt, st.session_state.chat_history)
            if text is None:
                text, fig = answer(prompt)
            st.markdown(text)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": text, "fig": fig}
    )

# =====================================================================
# Suggested quick questions
# =====================================================================
st.markdown("---")
st.markdown("#### 💡 Quick questions")
cols = st.columns(4)
suggestions = [
    "Total sales",
    "Total outstanding",
    "Top 5 customers",
    "Sales by executive",
    "Sales by zone",
    "Monthly trend",
    "How many customers",
    "Top 3 executives",
]
for i, s in enumerate(suggestions):
    if cols[i % 4].button(s, key=f"q_{i}", use_container_width=True):
        st.session_state.chat_history.append({"role": "user", "content": s})
        text, fig = answer(s)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": text, "fig": fig}
        )
        st.rerun()