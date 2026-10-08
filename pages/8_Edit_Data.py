import streamlit as st
import pandas as pd

from styles import inject_theme, hero, toast_success, toast_warning
from utils import get_df, persist, save_to_disk

st.set_page_config(page_title="Edit Data", page_icon="✏️", layout="wide")
inject_theme()

hero("Edit Data", "Modify rows inline, save to session, or write back to Excel.", icon="✏️")

df = get_df()

with st.expander("📋 Editable Data Table", expanded=True):
    edited = st.data_editor(
        st.session_state.df,
        use_container_width=True,
        num_rows="dynamic",
        key="editor",
    )

c1, c2, c3, c4 = st.columns(4)
with c1:
    if st.button("💾 Save to session", use_container_width=True):
        out = edited.copy()
        if 'Date' in out.columns:
            out['Date'] = pd.to_datetime(out['Date'])
        if 'Month Name' not in out.columns and 'Date' in out.columns:
            out['Month Name'] = out['Date'].dt.strftime('%B %Y')
        persist(out)
        toast_success("Saved to session")
        st.rerun()
with c2:
    if st.button("💾 Save → Excel", use_container_width=True):
        out = edited.copy()
        if 'Date' in out.columns:
            out['Date'] = pd.to_datetime(out['Date'])
        if 'Month Name' not in out.columns and 'Date' in out.columns:
            out['Month Name'] = out['Date'].dt.strftime('%B %Y')
        persist(out)
        if save_to_disk(out):
            toast_success("Written to sales_data_2025.xlsx")
            st.cache_data.clear()
with c3:
    if st.button("↩️ Discard", use_container_width=True):
        st.rerun()
with c4:
    csv = edited.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Export CSV", csv,
                       file_name="edited_data.csv", mime="text/csv",
                       use_container_width=True)

st.markdown("---")
with st.expander("🔎 Filter & Delete Rows"):
    c1, c2 = st.columns(2)
    cust_del = c1.selectbox("Filter by Customer",
                            ['-- All --'] + list(df['Customer Name'].unique()))
    inv_del  = c2.text_input("Filter by Invoice No (contains)")

    view = df.copy()
    if cust_del != '-- All --':
        view = view[view['Customer Name'] == cust_del]
    if inv_del:
        view = view[view['Invoice No'].astype(str).str.contains(inv_del, case=False)]

    st.dataframe(view.head(200), use_container_width=True)

    if st.button("🗑️ Delete Filtered Rows"):
        if cust_del == '-- All --' and not inv_del:
            toast_warning("Apply a filter before deleting")
        else:
            persist(df.drop(view.index))
            toast_success(f"Deleted {len(view)} rows")
            st.rerun()