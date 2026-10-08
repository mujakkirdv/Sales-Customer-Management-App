"""
pages/9_Native_Charts.py — Streamlit native charts + PyDeck + ECharts.
Demonstrates: st.area_chart, st.line_chart, st.scatter_chart,
              st.pydeck_chart, st_echarts (streamlit-echarts).
"""
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from streamlit_echarts import st_echarts

from styles import inject_theme, hero
from utils import get_df

st.set_page_config(page_title="Native Charts", page_icon="📉", layout="wide")
inject_theme()
hero("Native Charts", "Streamlit & 3rd-party chart primitives on your sales data.", icon="📉")

df = get_df()

# =====================================================================
# Common slice selector
# =====================================================================
with st.expander("🎛️ Chart Controls", expanded=True):
    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Metric",
                          ['Sales Amount', 'Credited Amount', 'Outstanding',
                           'Sales Return', 'Profit', 'Commission'])
    agg = c2.selectbox("Aggregation", ['sum', 'mean', 'count'])
    freq = c3.selectbox("Frequency", ['D', 'W', 'M'], index=2,
                        format_func=lambda x: {'D':'Daily','W':'Weekly','M':'Monthly'}[x])

# Time series (used by area / line / scatter)
ts = (df.set_index('Date')[metric]
        .resample(freq).agg(agg)
        .reset_index()
        .rename(columns={metric: 'value'})
        .dropna())

# Grouped data (used by echarts)
group_col = st.selectbox("Group by (for ECharts)",
                         ['Executive', 'Sales Zone', 'Sales Channel',
                          'Customer Type', 'Bank Name', 'Payment Method'],
                         key="echarts_group")
grouped = (df.groupby(group_col)[metric].agg(agg)
             .reset_index().sort_values(metric, ascending=False).head(10))

# =====================================================================
# 1. st.area_chart
# =====================================================================
st.markdown("### 🟦 `st.area_chart` — trend over time")
with st.expander("📖 What it does", expanded=False):
    st.markdown(
        """
        `st.area_chart` renders a **filled line chart** directly from a
        DataFrame / Series — no Plotly or Altair required.
        Best for showing **cumulative volume** over time.
        """
    )
st.area_chart(ts.set_index('Date'), color="#3b82f6", height=320)
st.caption(f"`{agg}({metric})` grouped by `{freq}`")

# =====================================================================
# 2. st.line_chart
# =====================================================================
st.markdown("### 🟩 `st.line_chart` — multi-series line")
with st.expander("📖 What it does", expanded=False):
    st.markdown(
        """
        `st.line_chart` accepts a **wide DataFrame** and plots one line
        per column. Use it to compare multiple metrics side-by-side.
        """
    )
multi = (df.set_index('Date')
           .resample(freq)[['Sales Amount', 'Credited Amount', 'Outstanding']]
           .sum().reset_index().set_index('Date'))
st.line_chart(multi, height=320, color=["#3b82f6", "#10b981", "#ef4444"])

# =====================================================================
# 3. st.scatter_chart
# =====================================================================
st.markdown("### 🟨 `st.scatter_chart` — correlation view")
with st.expander("📖 What it does", expanded=False):
    st.markdown(
        """
        `st.scatter_chart` is a quick scatterplot. Pass `x`, `y`,
        and optionally `color` / `size` columns.
        Great for spotting relationships (e.g. discount vs sales).
        """
    )
sample = df.sample(min(3000, len(df)), random_state=1)
st.scatter_chart(
    sample,
    x='Discount',
    y='Sales Amount',
    color='Customer Type',
    size='Invoice Value',
    height=340,
)

# =====================================================================
# 4. st.pydeck_chart
# =====================================================================
st.markdown("### 🗺️ `st.pydeck_chart` — geospatial")
with st.expander("📖 What it does", expanded=False):
    st.markdown(
        """
        `st.pydeck_chart` renders a **PyDeck** (deck.gl) map.
        We use the `Latitude` / `Longitude` columns derived from
        `Sales Zone` and colour each bubble by sales volume.
        """
    )

if {'Latitude', 'Longitude'}.issubset(df.columns):
    geo = (df.groupby(['Sales Zone'])
             .agg(Latitude=('Latitude', 'mean'),
                  Longitude=('Longitude', 'mean'),
                  Sales=('Sales Amount', 'sum'),
                  Customers=('Customer Name', 'nunique'))
             .reset_index())

    max_sales = geo['Sales'].max() or 1
    geo['radius'] = (geo['Sales'] / max_sales * 30000 + 8000).astype(int)

    layer = pdk.Layer(
        "ScatterplotLayer",
        geo,
        get_position=["Longitude", "Latitude"],
        get_radius="radius",
        get_fill_color=["255 * Sales / 500000", "100", "200 - 255 * Sales / 500000"],
        pickable=True,
        auto_highlight=True,
    )

    view = pdk.ViewState(
        latitude=float(geo['Latitude'].mean()),
        longitude=float(geo['Longitude'].mean()),
        zoom=6,
        pitch=40,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        tooltip={"text": "{Sales Zone}\nSales: ৳{Sales}\nCustomers: {Customers}"},
        map_style="mapbox://styles/mapbox/light-v9",
    )
    st.pydeck_chart(deck, use_container_width=True)
    st.caption("Bubble size ∝ total sales · Colour gradient ∝ sales volume")
else:
    st.info("Add 'Latitude'/'Longitude' columns to enable the map.")

# =====================================================================
# 5. st_echarts
# =====================================================================
st.markdown("### 🟪 `st_echarts` — interactive Apache ECharts")
with st.expander("📖 What it does", expanded=False):
    st.markdown(
        """
        `streamlit-echarts` wraps **Apache ECharts**, giving you
        gauges, radar charts, tree-maps, heatmaps, and more — all
        declarative via a simple Python dict.
        """
    )

# --- 5a. Bar with gradient ---
opt_bar = {
    "title": {"text": f"{agg.title()} of {metric} by {group_col}"},
    "tooltip": {"trigger": "axis"},
    "xAxis": {"type": "category", "data": grouped[group_col].tolist()},
    "yAxis": {"type": "value"},
    "series": [{
        "data": grouped[metric].round(0).tolist(),
        "type": "bar",
        "itemStyle": {
            "color": {
                "type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                "colorStops": [
                    {"offset": 0, "color": "#6366f1"},
                    {"offset": 1, "color": "#06b6d4"},
                ],
            },
            "borderRadius": [6, 6, 0, 0],
        },
    }],
}
st_echarts(options=opt_bar, height="380px")

# --- 5b. Gauge + Pie side-by-side ---
c1, c2 = st.columns(2)
with c1:
    rate = (df['Credited Amount'].sum() / df['Net Sales'].sum() * 100) if df['Net Sales'].sum() else 0
    opt_gauge = {
        "series": [{
            "type": "gauge",
            "startAngle": 180, "endAngle": 0,
            "min": 0, "max": 100,
            "progress": {"show": True, "width": 18},
            "axisLine": {"lineStyle": {"width": 18}},
            "detail": {"valueAnimation": True, "formatter": "{value}%",
                       "fontSize": 26, "offsetCenter": [0, "30%"]},
            "data": [{"value": round(rate, 1), "name": "Collection Rate"}],
        }]
    }
    st_echarts(options=opt_gauge, height="320px")

with c2:
    zone_sales = (df.groupby('Sales Zone')['Sales Amount'].sum()
                    .reset_index().sort_values('Sales Amount', ascending=False))
    opt_pie = {
        "title": {"text": "Sales share by Zone", "left": "center"},
        "tooltip": {"trigger": "item"},
        "series": [{
            "type": "pie",
            "radius": ["40%", "70%"],
            "avoidLabelOverlap": False,
            "itemStyle": {"borderRadius": 8, "borderColor": "#fff", "borderWidth": 2},
            "label": {"show": True, "formatter": "{b}: {d}%"},
            "data": [{"value": round(v, 0), "name": z}
                     for z, v in zip(zone_sales['Sales Zone'], zone_sales['Sales Amount'])],
        }],
    }
    st_echarts(options=opt_pie, height="320px")

# --- 5c. Radar chart ---
radar_cats = ['Sales', 'Credited', 'Outstanding', 'Profit', 'Returns']
exec_agg = (df.groupby('Executive')
              .agg(Sales=('Sales Amount', 'sum'),
                   Credited=('Credited Amount', 'sum'),
                   Outstanding=('Outstanding', 'sum'),
                   Profit=('Profit', 'sum'),
                   Returns=('Sales Return', 'sum'))
              .head(5))

# Normalise 0-100 for the radar
norm = exec_agg.div(exec_agg.max().replace(0, 1)) * 100

opt_radar = {
    "title": {"text": "Executive Radar (top 5, normalised)"},
    "tooltip": {},
    "legend": {"data": norm.index.tolist(), "bottom": 0},
    "radar": {"indicator": [{"name": c, "max": 100} for c in radar_cats]},
    "series": [{
        "type": "radar",
        "data": [{"value": row.round(0).tolist(), "name": name}
                 for name, row in norm.iterrows()],
    }],
}
st_echarts(options=opt_radar, height="420px")