# 🚀 Sales & Customer Management App

An interactive, multi-page **Streamlit** dashboard for managing sales, customers,
executives, and payments — with built-in analytics, machine learning, and an
AI chat assistant that talks to your data.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.36+-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

This app is designed for sales teams who want a **single place** to:

- Track every sale, payment, and outstanding balance
- Monitor executive and customer performance
- Generate on-demand reports (CSV / Excel / JSON / Print)
- Visualise trends with 5 chart types (Plotly, PyDeck, ECharts, native)
- Forecast sales, segment customers, and predict churn with ML
- Ask questions in plain English and get instant answers + charts

All data is loaded from a single Excel file — no database required.

---

## ✨ Features

| Category | Highlights |
|---|---|
| **📊 Dashboard** | Multi-filter KPIs, 6 interactive charts, CSV export |
| **👥 Customer Management** | Per-customer analytics, payment-method mix, transaction history |
| **👨‍💼 Executive Dashboard** | Sales, collection rate, commission, per-exec drill-down |
| **📝 Sales Entry** | Live calculation of VAT / net sales / outstanding |
| **📄 Reports** | 10 report types · CSV · Excel · JSON · Print (HTML) |
| **📈 Analytics** | Trends, MoM growth, rankings, correlation, treemap, sunburst |
| **📉 Native Charts** | `st.area_chart`, `st.line_chart`, `st.scatter_chart`, `st.pydeck_chart`, `st_echarts` |
| **🤖 Machine Learning** | Sales forecast, K-Means segmentation, outstanding prediction, churn risk |
| **💬 AI Chat** | Rule-based chat with your data · optional OpenAI integration |
| **🗂️ Data Management** | Upload, download, reload, save back to Excel |
| **✏️ Edit Data** | Inline editing with `st.data_editor`, filter-based delete |
| **🎨 UI/UX** | Custom CSS theme, gradient hero banner, styled KPI cards, toasts, bubbles |

---

## 📁 Project Structure

```
Sales-Customer-Management-App/
│
├── app.py                       # Entry point — router using st.navigation
├── utils.py                     # Shared constants, data loader, helpers
├── styles.py                    # Global CSS theme + reusable UI components
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── pages/                       # Streamlit multi-page app
│   ├── 1_Dashboard.py
│   ├── 2_Customer_Management.py
│   ├── 3_Executive_Dashboard.py
│   ├── 4_Sales_Entry.py
│   ├── 5_Reports.py
│   ├── 6_Analytics.py
│   ├── 7_Data_Management.py
│   ├── 8_Edit_Data.py
│   ├── 9_Native_Charts.py
│   ├── 10_Machine_Learning.py
│   └── 11_AI_Chat.py
│
└── data/
    └── sales_data_2025.xlsx     # Required — your dataset
```

---

## ⚙️ Installation

### 1. Clone / download the project

```bash
git clone <your-repo-url>
cd Sales-Customer-Management-App
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it:

| OS | Command |
|---|---|
| Windows | `.venv\Scripts\activate` |
| macOS / Linux | `source .venv/bin/activate` |

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you hit `ModuleNotFoundError: No module named 'streamlit_echarts'`, install the
optional packages manually:

```bash
pip install streamlit-echarts pydeck
```

---

## 📊 Data Requirements

Place your dataset at:

```
data/sales_data_2025.xlsx
```

### Required columns

| Column | Type | Description |
|---|---|---|
| `Date` | date | Transaction date |
| `Transaction ID` | text | Unique transaction reference |
| `Invoice No` | text | Invoice number |
| `Customer Name` | text | Customer identifier |
| `Customer Type` | text | New · Regular · VIP · Corporate |
| `Executive` | text | Salesperson name |
| `Sales Zone` | text | A1 · A2 · B1 · B2 · C1 · C2 · D1 |
| `Sales Channel` | text | Online · Retail · Corporate · Distributor · Direct |
| `Invoice Value` | number | Gross invoice amount |
| `Discount` | number | Discount applied |
| `Sales Amount` | number | Invoice Value − Discount |
| `Sales VAT` | number | VAT (5 %) |
| `Sales Return` | number | Returned amount |
| `Credited Amount` | number | Amount collected |
| `Payment Method` | text | Cash · Card · Bank Transfer · Cheque · Mobile Banking |
| `Bank Name` | text | Bank used for the transaction |
| `Remarks` | text | Free-text notes |
| `Month Name` | text | Auto-generated if missing |

> If `Month Name` is missing, the app will auto-generate it from `Date`.

---

## ▶️ Running the App

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**.

---

## 🧮 Business Rules

All derived values are computed in `utils.enrich()`:

| Field | Formula |
|---|---|
| `Net Sales` | `Sales Amount + Sales VAT − Sales Return` |
| `Outstanding` | `Net Sales − Credited Amount` |
| `Commission` | `Credited Amount × 1 %` |
| `Profit` | `Sales Amount × 25 %` |
| `Latitude` / `Longitude` | Derived from `Sales Zone` (jittered for map display) |

---

## 🗺️ Pages at a Glance

| # | Page | Purpose |
|---|---|---|
| 1 | **Dashboard** | KPIs, filters, six interactive charts |
| 2 | **Customer Management** | Per-customer dashboard + all-customer analytics |
| 3 | **Executive Dashboard** | Executive sales, collection rate, drill-down |
| 4 | **Sales Entry** | Add new transactions with live calculations |
| 5 | **Reports** | 10 report types · 4 export formats · period comparison |
| 6 | **Analytics** | Trends, rankings, correlation, zone/channel |
| 7 | **Data Management** | Upload, download, reload, save to Excel |
| 8 | **Edit Data** | Inline editing, filter-based delete |
| 9 | **Native Charts** | `st.area/line/scatter_chart`, PyDeck, ECharts |
| 10 | **Machine Learning** | Forecast · Segmentation · Outstanding · Churn |
| 11 | **AI Chat** | Chat with your data (rule-based + optional LLM) |

---

## 🤖 Machine Learning

| Tab | Model | Purpose |
|---|---|---|
| **Sales Forecast** | `RandomForestRegressor` | Forecast next 1–12 months · R² · MAE |
| **Segmentation** | `KMeans` + `StandardScaler` | Group customers by RFM-like features |
| **Outstanding Prediction** | `RandomForestRegressor` | Predict outstanding from transaction attributes |
| **Churn Risk** | `RandomForestClassifier` | Identify at-risk customers with probability |

> Models train on-page — no pre-trained files required.

---

## 💬 AI Chat

Two modes:

1. **Rule-based (default)** — works instantly, no API key.
   Understands queries like:
   - *total sales*
   - *total outstanding*
   - *top 5 customers*
   - *sales by executive / zone / channel*
   - *monthly trend*
   - *how many customers*

2. **OpenAI (optional)** — set your API key and toggle in the sidebar:

   ```bash
   export OPENAI_API_KEY=sk-...       # macOS / Linux
   setx OPENAI_API_KEY "sk-..."       # Windows
   ```

---

## 🎨 Customisation

### Change theme colours

Edit `styles.py` — the CSS block at the top of `inject_theme()`.

### Add a new page

1. Create `pages/12_My_Page.py`.
2. Add these three lines at the top:

   ```python
   import streamlit as st
   from styles import inject_theme, hero
   from utils import get_df

   st.set_page_config(page_title="My Page", page_icon="✨", layout="wide")
   inject_theme()
   hero("My Page", "Subtitle", icon="✨")
   ```

3. Register it in `app.py` inside the `pages` dictionary.

Streamlit will pick it up automatically.

---

## 🛠️ Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: streamlit_echarts` | `pip install streamlit-echarts` |
| `ModuleNotFoundError: pydeck` | `pip install pydeck` |
| `FileNotFoundError: sales_data_2025.xlsx` | Place the Excel file in `data/` |
| `KeyError: 'Sales Amount'` | Check column names match the schema above exactly |
| Charts render blank | `pip install --upgrade plotly` |
| Streamlit version too old | `pip install --upgrade streamlit>=1.36` |

---

## 📦 Requirements

```
streamlit>=1.36.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
openpyxl>=3.1.0
pydeck>=0.9.0
streamlit-echarts>=0.4.0
scikit-learn>=1.3.0
openai>=1.30.0        # optional, only for LLM chat
```

---

## 📸 Screenshot Placeholders

> Add screenshots to a `docs/` folder and reference them here:

```markdown
![Dashboard](docs/dashboard.png)
![Reports](docs/reports.png)
![ML](docs/ml.png)
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push (`git push origin feature/awesome`)
5. Open a Pull Request

---

## 📜 License

Released under the **MIT License** — free for personal and commercial use.

---

## 🙏 Acknowledgements

Built with:

- [Streamlit](https://streamlit.io/) — app framework
- [Plotly](https://plotly.com/python/) — interactive charts
- [PyDeck](https://deckgl.readthedocs.io/) — geospatial visualisation
- [Apache ECharts](https://echarts.apache.org/) — advanced chart types
- [scikit-learn](https://scikit-learn.org/) — machine learning
- [OpenAI](https://openai.com/) — optional chat-with-data

---

<p align="center">
  Made with ❤️ using Streamlit · v3.0 · 2025
</p>