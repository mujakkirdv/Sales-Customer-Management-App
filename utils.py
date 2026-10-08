"""
utils.py — Shared constants, data loaders, and helpers.
Imported by app.py and every file inside pages/.

Loads the working dataset from:  data/sales_data_2025.xlsx
No sample data is generated — the Excel file MUST exist.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np

# =====================================================================
# CONSTANTS
# =====================================================================
BANKS = ['MBL CC', 'MBL CD', 'MBL NRB CD', 'MBL NRB CC', 'MBL WB CC',
         'MBL WB CD', 'BRAC BANK', 'City Bank', 'DBBL', 'HSBC', 'Cash']

PAYMENT_METHODS = ['Cash', 'Card', 'Bank Transfer', 'Cheque', 'Mobile Banking']

EXECUTIVES = ["ATM Nur Hossain Rumel", "Mynuddin Hasan hridoy",
              "Sujoy Kumar Biswas", "Sajib Kumar Biswas", "Sanjoy Hore",
              "Mohammad Sumon", "Al - Amin"]

CUSTOMER_TYPES  = ['New', 'Regular', 'VIP', 'Corporate']
SALES_ZONES     = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1']
SALES_CHANNELS  = ['Online', 'Retail', 'Corporate', 'Distributor', 'Direct']
REMARKS_POOL    = ['Paid in full', 'Partial payment', 'Credit sale',
                   'Cash sale', 'Follow-up required', 'Regular customer', '']

VAT_RATE        = 0.05
COMMISSION_RATE = 0.01
PROFIT_RATE     = 0.25

# =====================================================================
# GEO COORDINATES FOR SALES ZONES (used by st.pydeck_chart)
# =====================================================================
ZONE_COORDS = {
    'A1': (23.8103, 90.4125),   # Dhaka
    'A2': (22.3569, 91.7832),   # Chittagong
    'B1': (24.8949, 91.8687),   # Sylhet
    'B2': (23.7104, 90.4074),   # Dhaka (north)
    'C1': (24.3636, 88.6241),   # Rajshahi
    'C2': (22.8456, 89.5403),   # Khulna
    'D1': (25.7439, 89.2752),   # Rangpur
}

# =====================================================================
# DATA PATHS
# =====================================================================
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATA_PATH = os.path.join(DATA_DIR, "welburg_sales_dataset_2025.xlsx")

# =====================================================================
# DATA LOADER  (Excel only — no sample data)
# =====================================================================
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the working dataset from data/welburg_sales_dataset_2025.xlsx."""
    if not os.path.exists(DATA_PATH):
        st.error(
            f"❌ Data file not found: `{DATA_PATH}`\n\n"
            "Please place **welburg_sales_dataset_2025.xlsx** inside the `data/` folder "
            "and refresh the page."
        )
        st.stop()

    try:
        df = pd.read_excel(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Could not read `{DATA_PATH}`: {e}")
        st.stop()

    # ---- Normalise Date ----
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # ---- Ensure Month Name exists ----
    if 'Month Name' not in df.columns and 'Date' in df.columns:
        df['Month Name'] = df['Date'].dt.strftime('%B %Y')

    return df

# =====================================================================
# DERIVED COLUMNS
# =====================================================================
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add Net Sales, Outstanding, Commission, Profit, Latitude, Longitude."""
    df = df.copy()

    df['Net Sales']   = df['Sales Amount'] + df['Sales VAT'] - df['Sales Return']
    df['Outstanding'] = df['Net Sales'] - df['Credited Amount']
    df['Commission']  = df['Credited Amount'] * COMMISSION_RATE
    df['Profit']      = df['Sales Amount']    * PROFIT_RATE

    # ---- Geo coordinates derived from Sales Zone (jittered to avoid overlap) ----
    if 'Sales Zone' in df.columns:
        rng = np.random.default_rng(42)
        lats = df['Sales Zone'].map(
            lambda z: ZONE_COORDS.get(z, (23.8103, 90.4125))[0]
        )
        lons = df['Sales Zone'].map(
            lambda z: ZONE_COORDS.get(z, (23.8103, 90.4125))[1]
        )
        df['Latitude']  = lats + rng.normal(0, 0.15, len(df))
        df['Longitude'] = lons + rng.normal(0, 0.15, len(df))

    return df

def get_df() -> pd.DataFrame:
    """Return enriched dataframe from session state (initialised on first call)."""
    if 'df' not in st.session_state:
        st.session_state.df = load_data()
    return enrich(st.session_state.df)

def persist(new_df: pd.DataFrame) -> None:
    """Replace the working dataframe in session state."""
    st.session_state.df = new_df

def save_to_disk(df: pd.DataFrame) -> bool:
    """Persist the working dataframe back to the Excel file."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        # Drop helper columns before saving to keep the file clean
        drop_cols = [c for c in ['Net Sales', 'Outstanding', 'Commission',
                                 'Profit', 'Latitude', 'Longitude']
                     if c in df.columns]
        df.drop(columns=drop_cols).to_excel(DATA_PATH, index=False)
        return True
    except Exception as e:
        st.error(f"❌ Could not save to `{DATA_PATH}`: {e}")
        return False