import streamlit as st
import pandas as pd
from utils import get_df, persist, save_to_disk, DATA_PATH

from styles import inject_theme, hero
# ...
st.set_page_config(page_title="Data Management", page_icon="🗂️", layout="wide")
inject_theme()
hero("Data Management", "Upload, save, and manage your sales data", icon="🗂️")

st.set_page_config(page_title="Data Management", page_icon="🗂️", layout="wide")
st.subheader("🗂️ Data Management")
df = get_df()

# ---------- Stats ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows",       f"{len(df):,}")
k2.metric("Columns",    len(df.columns))
k3.metric("Date Range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
k4.metric("Memory",     f"{df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

st.caption(f"📁 Working file: `{DATA_PATH}`")

# ---------- Upload (replace in-memory only) ----------
st.markdown("---")
st.subheader("📤 Upload Replacement Data (Excel / CSV)")
uploaded = st.file_uploader("Upload Excel or CSV", type=['xlsx', 'xls', 'csv'])
if uploaded is not None:
    try:
        if uploaded.name.endswith('.csv'):
            new_df = pd.read_csv(uploaded)
        else:
            new_df = pd.read_excel(uploaded)
        if 'Date' in new_df.columns:
            new_df['Date'] = pd.to_datetime(new_df['Date'])
        if 'Month Name' not in new_df.columns and 'Date' in new_df.columns:
            new_df['Month Name'] = new_df['Date'].dt.strftime('%B %Y')
        persist(new_df)
        st.success("✅ Data loaded into session (in-memory).")
        st.dataframe(new_df.head())
    except Exception as e:
        st.error(f"Error reading file: {e}")

# ---------- Save to disk ----------
st.markdown("---")
st.subheader("💾 Save Current Data to Excel")
if st.button("💾 Write session data → sales_data_2025.xlsx", use_container_width=True):
    if save_to_disk(st.session_state.df):
        st.success(f"✅ Saved to `{DATA_PATH}`")
        st.cache_data.clear()

# ---------- Download ----------
st.markdown("---")
st.subheader("📥 Download Full Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv,
                   file_name="sales_data_2025_export.csv",
                   mime="text/csv")

# ---------- Preview ----------
st.markdown("---")
st.subheader("🔍 Data Preview")
st.dataframe(df.head(100), use_container_width=True)

# ---------- Danger zone ----------
st.markdown("---")
st.subheader("⚠️ Danger Zone")
c1, c2 = st.columns(2)
with c1:
    if st.button("🔄 Reload from Excel file", use_container_width=True):
        st.cache_data.clear()
        st.session_state.pop('df', None)
        st.success("Reloaded from disk.")
        st.rerun()
with c2:
    if st.button("🗑️ Clear All Rows (in memory)", use_container_width=True):
        persist(st.session_state.df.iloc[0:0])
        st.warning("All rows removed (in-memory only).")
        st.rerun()