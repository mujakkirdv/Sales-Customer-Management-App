"""
pages/5_Reports.py — Enterprise Reporting Engine
─────────────────────────────────────────────────
• 10 report types + custom grouping
• Sidebar filters (dates, executive, zone, channel, type, bank, payment)
• KPI strip with current vs previous period comparison
• Every report gets: table + chart + summary + CSV + Excel + Print
• Styled, formatted, ranked output
"""
import io
from datetime import timedelta

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
st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")
inject_theme()
hero("Reports Center",
     "Build, visualise, compare, and export any report — in one place.",
     icon="📄")

df = get_df()

# =====================================================================
# REPORT DEFINITIONS
# =====================================================================
REPORT_TYPES = {
    "Sales by Executive":      {"group": "Executive",      "metrics": ["Sales","Credited","Outstanding","Profit"]},
    "Sales by Zone":           {"group": "Sales Zone",     "metrics": ["Sales","Credited","Outstanding"]},
    "Sales by Channel":        {"group": "Sales Channel",  "metrics": ["Sales","Credited","Outstanding"]},
    "Sales by Customer Type":  {"group": "Customer Type",  "metrics": ["Sales","Credited","Outstanding"]},
    "Sales by Bank":           {"group": "Bank Name",      "metrics": ["Sales","Credited"]},
    "Sales by Payment Method": {"group": "Payment Method", "metrics": ["Sales","Credited"]},
    "Monthly Sales":           {"group": "Month Name",     "metrics": ["Sales","Credited","Outstanding","Invoices"]},
    "Top Customers":           {"group": "Customer Name",  "metrics": ["Sales","Credited","Outstanding"]},
    "Outstanding Aging":       {"group": "Customer Name",  "metrics": ["Outstanding","Invoices","Last_Date"]},
    "Returns Report":          {"group": "Customer Name",  "metrics": ["Returns","Invoices"]},
}

# =====================================================================
# SIDEBAR — GLOBAL FILTERS
# =====================================================================
with st.sidebar:
    st.markdown("### 🎛️ Report Filters")

    with st.expander("📅 Date Range", expanded=True):
        d1 = st.date_input("From", df['Date'].min().date(), key="rpt_from")
        d2 = st.date_input("To",   df['Date'].max().date(), key="rpt_to")

        # Quick range buttons
        qc1, qc2 = st.columns(2)
        if qc1.button("Last 30d", use_container_width=True):
            st.session_state.rpt_from = (df['Date'].max() - timedelta(days=30)).date()
            st.session_state.rpt_to   = df['Date'].max().date()
            st.rerun()
        if qc2.button("Last 90d", use_container_width=True):
            st.session_state.rpt_from = (df['Date'].max() - timedelta(days=90)).date()
            st.session_state.rpt_to   = df['Date'].max().date()
            st.rerun()

    with st.expander("🏢 Dimensions", expanded=False):
        f_exec    = st.multiselect("Executives",      df['Executive'].unique(),       df['Executive'].unique())
        f_zone    = st.multiselect("Sales Zone",      df['Sales Zone'].unique(),      df['Sales Zone'].unique())
        f_channel = st.multiselect("Sales Channel",   df['Sales Channel'].unique(),   df['Sales Channel'].unique())
        f_ctype   = st.multiselect("Customer Type",   df['Customer Type'].unique(),   df['Customer Type'].unique())

    with st.expander("💳 Payments", expanded=False):
        f_pm      = st.multiselect("Payment Method",  df['Payment Method'].unique(),  df['Payment Method'].unique())
        f_bank    = st.multiselect("Bank Name",       df['Bank Name'].unique(),       df['Bank Name'].unique())

    if st.button("🔄 Reset all filters", use_container_width=True):
        st.rerun()

# =====================================================================
# APPLY DATE FILTER + PREVIOUS-PERIOD WINDOW
# =====================================================================
d1_ts, d2_ts = pd.to_datetime(d1), pd.to_datetime(d2)
period_days  = (d2_ts - d1_ts).days or 1
prev_d1_ts   = d1_ts - timedelta(days=period_days)
prev_d2_ts   = d1_ts - timedelta(days=1)

base_mask = (
    df['Executive'].isin(f_exec) &
    df['Sales Zone'].isin(f_zone) &
    df['Sales Channel'].isin(f_channel) &
    df['Customer Type'].isin(f_ctype) &
    df['Payment Method'].isin(f_pm) &
    df['Bank Name'].isin(f_bank)
)

rdf      = df[base_mask & (df['Date'] >= d1_ts) & (df['Date'] <= d2_ts)].copy()
prev_rdf = df[base_mask & (df['Date'] >= prev_d1_ts) & (df['Date'] <= prev_d2_ts)].copy()

if rdf.empty:
    st.warning("⚠️ No data matches the current filters.")
    st.stop()

# =====================================================================
# KPI STRIP — CURRENT vs PREVIOUS
# =====================================================================
def pct_delta(cur, prev):
    if prev == 0:
        return None
    return (cur - prev) / prev * 100

cur_sales   = rdf['Sales Amount'].sum()
cur_credit  = rdf['Credited Amount'].sum()
cur_out     = rdf['Outstanding'].sum()
cur_profit  = rdf['Profit'].sum()
cur_inv     = rdf['Invoice No'].count()

prev_sales  = prev_rdf['Sales Amount'].sum()
prev_credit = prev_rdf['Credited Amount'].sum()
prev_out    = prev_rdf['Outstanding'].sum()
prev_profit = prev_rdf['Profit'].sum()
prev_inv    = prev_rdf['Invoice No'].count()

def fmt_delta(v):
    return None if v is None else f"{v:+.1f}%"

chips([
    (f"📦 {len(rdf):,} rows", ""),
    (f"📅 {d1:%d/%m/%y} → {d2:%d/%m/%y}", ""),
    (f"⚖️ vs previous {period_days}d", "amber"),
])

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("💰 Sales",       f"৳ {cur_sales:,.0f}",  fmt_delta(pct_delta(cur_sales, prev_sales)))
k2.metric("💳 Credited",    f"৳ {cur_credit:,.0f}", fmt_delta(pct_delta(cur_credit, prev_credit)))
k3.metric("📊 Outstanding", f"৳ {cur_out:,.0f}",    fmt_delta(pct_delta(cur_out, prev_out)))
k4.metric("📈 Profit",      f"৳ {cur_profit:,.0f}", fmt_delta(pct_delta(cur_profit, prev_profit)))
k5.metric("🧾 Invoices",    f"{cur_inv:,}",         fmt_delta(pct_delta(cur_inv, prev_inv)))

# =====================================================================
# REPORT SELECTOR
# =====================================================================
st.markdown("---")
c1, c2, c3 = st.columns([2, 1, 1])
rpt_type  = c1.selectbox("📋 Report Type", list(REPORT_TYPES.keys()))
top_n     = c2.number_input("Top N (for ranked reports)", 5, 500, 50, step=5)
sort_desc = c3.toggle("Sort descending", value=True)

# =====================================================================
# BUILD REPORT
# =====================================================================
def build_report(source, report_type, top_n):
    """Return a tidy DataFrame for the given report type."""
    if report_type == "Sales by Executive":
        out = source.groupby('Executive').agg(
            Sales=('Sales Amount', 'sum'),
            Credited=('Credited Amount', 'sum'),
            Outstanding=('Outstanding', 'sum'),
            Profit=('Profit', 'sum'),
            Invoices=('Invoice No', 'count'),
        ).reset_index()

    elif report_type == "Sales by Zone":
        out = source.groupby('Sales Zone').agg(
            Sales=('Sales Amount', 'sum'),
            Credited=('Credited Amount', 'sum'),
            Outstanding=('Outstanding', 'sum'),
            Invoices=('Invoice No', 'count'),
        ).reset_index()

    elif report_type == "Sales by Channel":
        out = source.groupby('Sales Channel').agg(
            Sales=('Sales Amount', 'sum'),
            Credited=('Credited Amount', 'sum'),
            Outstanding=('Outstanding', 'sum'),
            Invoices=('Invoice No', 'count'),
        ).reset_index()

    elif report_type == "Sales by Customer Type":
        out = source.groupby('Customer Type').agg(
            Sales=('Sales Amount', 'sum'),
            Credited=('Credited Amount', 'sum'),
            Outstanding=('Outstanding', 'sum'),
            Invoices=('Invoice No', 'count'),
        ).reset_index()

    elif report_type == "Sales by Bank":
        out = source.groupby('Bank Name').agg(
            Sales=('Sales Amount', 'sum'),
            Credited=('Credited Amount', 'sum'),
            Invoices=('Invoice No', 'count'),
        ).reset_index()

    elif report_type == "Sales by Payment Method":
        out = source.groupby('Payment Method').agg(
            Sales=('Sales Amount', 'sum'),
            Credited=('Credited Amount', 'sum'),
            Invoices=('Invoice No', 'count'),
        ).reset_index()

    elif report_type == "Monthly Sales":
        out = source.groupby('Month Name').agg(
            Sales=('Sales Amount', 'sum'),
            Credited=('Credited Amount', 'sum'),
            Outstanding=('Outstanding', 'sum'),
            Invoices=('Invoice No', 'count'),
        ).reset_index()
        # Sort chronologically
        order = source.drop_duplicates('Month Name').sort_values('Date')['Month Name'].tolist()
        out['Month Name'] = pd.Categorical(out['Month Name'], categories=order, ordered=True)
        out = out.sort_values('Month Name')

    elif report_type == "Top Customers":
        out = (source.groupby('Customer Name')
               .agg(Sales=('Sales Amount', 'sum'),
                    Credited=('Credited Amount', 'sum'),
                    Outstanding=('Outstanding', 'sum'),
                    Invoices=('Invoice No', 'count'))
               .reset_index()
               .sort_values('Sales', ascending=False)
               .head(top_n))

    elif report_type == "Outstanding Aging":
        out = (source[source['Outstanding'] > 0]
               .groupby('Customer Name')
               .agg(Outstanding=('Outstanding', 'sum'),
                    Invoices=('Invoice No', 'count'),
                    Last_Date=('Date', 'max'))
               .reset_index()
               .sort_values('Outstanding', ascending=False)
               .head(top_n))

    else:  # Returns Report
        out = (source[source['Sales Return'] > 0]
               .groupby('Customer Name')
               .agg(Returns=('Sales Return', 'sum'),
                    Invoices=('Invoice No', 'count'))
               .reset_index()
               .sort_values('Returns', ascending=False)
               .head(top_n))

    return out

# Build current + previous period reports
out      = build_report(rdf, rpt_type, top_n)
prev_out = build_report(prev_rdf, rpt_type, top_n)

# Sort
group_col = REPORT_TYPES[rpt_type]["group"]
if group_col in out.columns and rpt_type != "Monthly Sales":
    sort_col = "Sales" if "Sales" in out.columns else out.columns[1]
    out = out.sort_values(sort_col, ascending=not sort_desc).reset_index(drop=True)

# Add rank + % of total
rank_metric = "Sales" if "Sales" in out.columns else out.columns[1]
out.insert(0, "Rank", range(1, len(out) + 1))
if rank_metric in out.columns and out[rank_metric].sum() != 0:
    out["Share %"] = (out[rank_metric] / out[rank_metric].sum() * 100).round(2)

# Merge previous period for delta
if group_col in prev_out.columns and rank_metric in prev_out.columns:
    prev_map = prev_out.set_index(group_col)[rank_metric].to_dict()
    out["Prev " + rank_metric] = out[group_col].map(prev_map).fillna(0)
    out["Δ %"] = np.where(
        out["Prev " + rank_metric] > 0,
        (out[rank_metric] - out["Prev " + rank_metric])
            / out["Prev " + rank_metric] * 100,
        np.nan,
    ).round(1)

# =====================================================================
# TABS — TABLE · CHART · SUMMARY
# =====================================================================
tab_table, tab_chart, tab_summary = st.tabs(["📋 Table", "📊 Chart", "🔍 Summary"])

# ---------- TABS: TABLE ----------
with tab_table:
    st.markdown(f"#### {rpt_type}")

    # Column formatting
    money_cols = [c for c in out.columns
                  if any(k in c for k in ['Sales', 'Credited', 'Outstanding', 'Profit',
                                          'Returns', 'Prev'])]
    pct_cols   = [c for c in out.columns if c in ['Share %', 'Δ %']]
    date_cols  = [c for c in out.columns if 'Date' in c]

    fmt = {}
    for c in money_cols: fmt[c] = '৳{:,.0f}'
    for c in pct_cols:   fmt[c] = '{:+.1f}%' if c == 'Δ %' else '{:.1f}%'
    for c in date_cols:  fmt[c] = lambda x: x.strftime('%d/%m/%Y') if pd.notnull(x) else ''

    # Totals row (numeric columns only)
    numeric_cols = out.select_dtypes(include=np.number).columns.tolist()
    totals = {c: out[c].sum() for c in numeric_cols
              if c not in ['Rank', 'Share %', 'Δ %']}
    totals[group_col] = "TOTAL"
    if "Rank" in out.columns:      totals["Rank"] = ""
    if "Share %" in out.columns:   totals["Share %"] = 100.0
    if "Δ %" in out.columns:       totals["Δ %"] = np.nan

    totals_row = pd.DataFrame([totals])
    display = pd.concat([out, totals_row], ignore_index=True)

    styled = display.style.format(fmt, na_rep='—')

    # Colour scale on rank metric
    if rank_metric in out.columns:
        styled = styled.background_gradient(subset=[rank_metric], cmap='Blues')

    # Highlight the total row
    styled = styled.apply(
        lambda row: ['background-color: #f1f5f9; font-weight: 700;'
                     if row.get(group_col) == "TOTAL" else '' for _ in row],
        axis=1,
    )

    st.dataframe(styled, use_container_width=True, height=500)

    # ---------- Exports ----------
    st.markdown("##### 📥 Export this report")
    ec1, ec2, ec3, ec4 = st.columns(4)

    # Clean (unstyled) export uses `out` (no totals row) or `display`? Use display for parity
    export_df = display.copy()
    for c in date_cols:
        if c in export_df.columns:
            export_df[c] = export_df[c].astype(str)

    # CSV
    csv = export_df.to_csv(index=False).encode('utf-8')
    ec1.download_button(
        "📄 CSV",
        csv,
        file_name=f"{rpt_type.replace(' ', '_')}_{d1}_{d2}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # Excel (with real formatting via openpyxl)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Report')
    ec2.download_button(
        "📊 Excel",
        buf.getvalue(),
        file_name=f"{rpt_type.replace(' ', '_')}_{d1}_{d2}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    # JSON
    json_bytes = export_df.to_json(orient='records', indent=2).encode('utf-8')
    ec3.download_button(
        "🧾 JSON",
        json_bytes,
        file_name=f"{rpt_type.replace(' ', '_')}_{d1}_{d2}.json",
        mime="application/json",
        use_container_width=True,
    )

    # Print-friendly HTML
    html = f"""
    <html><head><title>{rpt_type}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2rem; }}
        h1 {{ color: #1e293b; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
        th, td {{ border: 1px solid #cbd5e1; padding: 6px 10px; text-align: right; }}
        th {{ background: #1e293b; color: #fff; }}
        tr:nth-child(even) {{ background: #f8fafc; }}
        tr:last-child {{ background: #f1f5f9; font-weight: bold; }}
    </style></head><body>
    <h1>{rpt_type}</h1>
    <p><b>Period:</b> {d1} → {d2} &nbsp; | &nbsp;
       <b>Records:</b> {len(rdf):,} &nbsp; | &nbsp;
       <b>Generated:</b> {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
    {export_df.to_html(index=False, border=0, float_format=lambda x: f"{x:,.2f}")}
    </body></html>
    """
    ec4.download_button(
        "🖨️ Print (HTML)",
        html.encode('utf-8'),
        file_name=f"{rpt_type.replace(' ', '_')}_{d1}_{d2}.html",
        mime="text/html",
        use_container_width=True,
    )

# ---------- TAB: CHART ----------
with tab_chart:
    st.markdown(f"#### 📊 Visualisation — {rpt_type}")

    if rpt_type == "Outstanding Aging":
        # Bar of top 15 customers by outstanding
        chart_df = out.head(15)
        fig = px.bar(chart_df, x='Outstanding', y=group_col,
                     orientation='h', color='Outstanding',
                     color_continuous_scale='reds',
                     title=f"Top 15 — {rpt_type}",
                     text='Outstanding')
        fig.update_traces(texttemplate='৳%{text:,.0f}', textposition='outside')
        fig.update_layout(height=520, coloraxis_showscale=False,
                          yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

    elif rpt_type == "Returns Report":
        chart_df = out.head(15)
        fig = px.bar(chart_df, x='Returns', y=group_col,
                     orientation='h', color='Returns',
                     color_continuous_scale='oranges',
                     title=f"Top 15 — {rpt_type}",
                     text='Returns')
        fig.update_traces(texttemplate='৳%{text:,.0f}', textposition='outside')
        fig.update_layout(height=520, coloraxis_showscale=False,
                          yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

    elif rpt_type == "Monthly Sales":
        fig = go.Figure()
        fig.add_trace(go.Bar(x=out['Month Name'], y=out['Sales'],
                             name='Sales', marker_color='#3b82f6'))
        fig.add_trace(go.Bar(x=out['Month Name'], y=out['Credited'],
                             name='Credited', marker_color='#10b981'))
        fig.add_trace(go.Scatter(x=out['Month Name'], y=out['Outstanding'],
                                 name='Outstanding', mode='lines+markers',
                                 line=dict(color='#ef4444', width=3)))
        fig.update_layout(barmode='group', height=460, hovermode='x unified',
                          title="Monthly Overview")
        st.plotly_chart(fig, use_container_width=True)

    elif rpt_type == "Sales by Payment Method":
        fig = px.pie(out, values='Credited', names='Payment Method',
                     hole=0.45, title="Credited by Payment Method",
                     color_discrete_sequence=px.colors.sequential.Tealgrn)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)

    elif rpt_type == "Sales by Bank":
        fig = px.bar(out, x='Bank Name', y='Credited',
                     color='Credited', color_continuous_scale='Blues',
                     title="Credited by Bank")
        fig.update_layout(height=460, xaxis_tickangle=-45,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Bar for most other reports (top 20 by rank_metric)
        top_df = out.head(20)
        fig = px.bar(top_df, x=rank_metric, y=group_col,
                     orientation='h', color=rank_metric,
                     color_continuous_scale='viridis',
                     title=f"Top {len(top_df)} — {rpt_type}",
                     text=rank_metric)
        fig.update_traces(texttemplate='৳%{text:,.0f}', textposition='outside')
        fig.update_layout(height=560, coloraxis_showscale=False,
                          yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

    # Compare vs previous period (grouped bar)
    if "Prev " + rank_metric in out.columns:
        st.markdown("##### ⚖️ Current vs Previous period")
        compare_df = out.head(15).melt(
            id_vars=[group_col],
            value_vars=[rank_metric, "Prev " + rank_metric],
            var_name="Period", value_name="Amount",
        )
        compare_df["Period"] = compare_df["Period"].replace({
            rank_metric: "Current",
            "Prev " + rank_metric: "Previous",
        })
        fig_cmp = px.bar(compare_df, x=group_col, y="Amount",
                         color="Period", barmode='group',
                         color_discrete_map={"Current": "#3b82f6",
                                             "Previous": "#94a3b8"},
                         title="Current vs Previous period")
        fig_cmp.update_layout(height=440, xaxis_tickangle=-45,
                              legend_title=None)
        st.plotly_chart(fig_cmp, use_container_width=True)

# ---------- TAB: SUMMARY ----------
with tab_summary:
    st.markdown(f"#### 🔍 Insights — {rpt_type}")

    # Top/bottom performers
    if len(out) >= 2 and rank_metric in out.columns:
        top    = out.iloc[0]
        bottom = out.iloc[-1]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.success(
                f"**🏆 Best: {top[group_col]}**\n\n"
                f"{rank_metric}: **৳{top[rank_metric]:,.0f}**  \n"
                f"Share: **{top.get('Share %', 0):.1f}%**"
            )
        with c2:
            st.error(
                f"**📉 Weakest: {bottom[group_col]}**\n\n"
                f"{rank_metric}: **৳{bottom[rank_metric]:,.0f}**  \n"
                f"Share: **{bottom.get('Share %', 0):.1f}%**"
            )
        with c3:
            avg = out[rank_metric].mean()
            median = out[rank_metric].median()
            st.info(
                f"**📊 Distribution**\n\n"
                f"Average: **৳{avg:,.0f}**  \n"
                f"Median: **৳{median:,.0f}**  \n"
                f"Entries: **{len(out)}**"
            )

    # Concentration: top 5 = X% of total
    if "Share %" in out.columns:
        top5 = out.head(5)["Share %"].sum()
        top10 = out.head(10)["Share %"].sum()
        st.markdown("##### 📈 Concentration analysis")
        st.markdown(
            f"- **Top 5** account for **{top5:.1f}%** of total {rank_metric}\n"
            f"- **Top 10** account for **{top10:.1f}%** of total {rank_metric}\n"
        )
        if top5 > 60:
            st.warning("⚠️ High concentration — top 5 drive more than 60% of results.")
        elif top5 < 30:
            st.info("ℹ️ Well distributed — no single group dominates.")

    # Delta insights
    if "Δ %" in out.columns:
        deltas = out.dropna(subset=["Δ %"]).sort_values("Δ %", ascending=False)
        if not deltas.empty:
            st.markdown("##### ⚖️ Biggest movers vs previous period")
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**📈 Top 5 growers**")
                st.dataframe(
                    deltas.head(5)[[group_col, rank_metric, "Δ %"]].style.format({
                        rank_metric: '৳{:,.0f}', 'Δ %': '{:+.1f}%'
                    }),
                    use_container_width=True,
                )
            with mc2:
                st.markdown("**📉 Top 5 decliners**")
                st.dataframe(
                    deltas.tail(5)[[group_col, rank_metric, "Δ %"]].style.format({
                        rank_metric: '৳{:,.0f}', 'Δ %': '{:+.1f}%'
                    }),
                    use_container_width=True,
                )

    # Raw data expander
    with st.expander("🔍 Raw filtered data (first 500 rows)", expanded=False):
        st.dataframe(rdf.head(500), use_container_width=True)

# =====================================================================
# FOOTER
# =====================================================================
st.markdown("---")
st.caption(
    f"📄 **{rpt_type}** · "
    f"Period **{d1:%d %b %Y} → {d2:%d %b %Y}** · "
    f"**{len(rdf):,}** records · "
    f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}"
)
toast_success(f"Report ready: {rpt_type}")