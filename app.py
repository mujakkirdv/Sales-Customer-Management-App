"""
app.py — Entry point.
Uses Streamlit's modern st.navigation API for a grouped sidebar.
"""
import streamlit as st
from styles import inject_theme
from utils import DATA_PATH

st.set_page_config(
    page_title="Sales & Customer App",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_theme()

pages = {
    "🏠 Overview": [
        st.Page("pages/1_Dashboard.py",            title="Dashboard",  icon="📊", default=True),
        st.Page("pages/6_Analytics.py",            title="Analytics",  icon="📈"),
    ],
    "📉 Charts": [
        st.Page("pages/9_Native_Charts.py",        title="Native Charts", icon="📉"),
    ],
    "👥 Relationships": [
        st.Page("pages/2_Customer_Management.py",  title="Customers",  icon="👥"),
        st.Page("pages/3_Executive_Dashboard.py",  title="Executives", icon="👨‍💼"),
    ],
    "📝 Transactions": [
        st.Page("pages/4_Sales_Entry.py",          title="Sales Entry", icon="📝"),
        st.Page("pages/5_Reports.py",              title="Reports",     icon="📄"),
    ],
    "🤖 AI & ML": [
        st.Page("pages/10_Machine_Learning.py",    title="Machine Learning", icon="🤖"),
        st.Page("pages/11_AI_Chat.py",             title="AI Chat",          icon="💬"),
    ],
    "🗂️ Data": [
        st.Page("pages/7_Data_Management.py",      title="Data Management", icon="🗂️"),
        st.Page("pages/8_Edit_Data.py",            title="Edit Data",       icon="✏️"),
    ],
}

with st.sidebar:
    st.markdown(
        """
        <div style="padding:0.8rem 0.4rem 0.4rem 0.4rem;">
            <div style="display:flex;align-items:center;gap:0.6rem;">
                <div style="width:38px;height:38px;border-radius:10px;
                            background:linear-gradient(135deg,#6366f1,#06b6d4);
                            display:flex;align-items:center;justify-content:center;
                            font-size:1.2rem;">🚀</div>
                <div>
                    <div style="font-weight:700;font-size:1.05rem;">Sales App</div>
                    <div style="font-size:0.72rem;opacity:0.7;">v3.0 • AI Edition</div>
                </div>
            </div>
        </div>
        <hr style="border-color:#334155;margin:0.4rem 0 0.8rem 0;">
        """,
        unsafe_allow_html=True,
    )

nav = st.navigation(pages, position="sidebar")
nav.run()

with st.sidebar:
    st.markdown("---")
    st.caption(f"📁 `{DATA_PATH.split('/')[-1]}`")
    st.caption("Made with ❤️ using Streamlit")