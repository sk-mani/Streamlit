# ============================================================
# SUPERSTORE SALES DASHBOARD
# Author: Surya Kant Mani
# Description:
# End-to-end analytical dashboard using Streamlit.
# Combines Orders + People + Returns into one semantic model,
# applies business filters, computes YoY KPIs, and visualizes
# Monthly Sales and Profit Ratio comparisons using Altair.
# ============================================================

import streamlit as st
import pandas as pd
import altair as alt
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================
# This sets up the basic Streamlit page settings.
# 'wide' layout gives more horizontal space for charts (dashboard feel)
st.set_page_config(
    page_title="Superstore Sales Dashboard",
    layout="wide"
)

# Main title shown on top of dashboard
st.title("📊 Superstore Sales Dashboard")


# ============================================================
# DATA LOADING + SEMANTIC MODEL CREATION
# ============================================================
@st.cache_data
def load_data():
    """
    This function loads raw CSV files and converts them into a
    single analytical dataset (like a mini data warehouse model).

    Steps:
    1. Load Orders (fact table)
    2. Load People (dimension: region managers)
    3. Load Returns (behavioral flag)
    4. Merge everything into one dataframe
    """

    # Path of current script (Summary.py)
    base_path = os.path.dirname(__file__)

    # Build path to data folder inside the same project folder
    data_path = os.path.join(base_path, "data")

    # Read CSV files safely
    orders = pd.read_csv(os.path.join(data_path, "Superstore_Orders.csv"))
    people = pd.read_csv(os.path.join(data_path, "Superstore_People.csv"))
    returns = pd.read_csv(os.path.join(data_path, "Superstore_Returns.csv"))

    # --- Convert date columns to proper datetime ---
    # This is critical for time-based filtering and trend charts
    orders["Order Date"] = pd.to_datetime(orders["Order Date"])
    orders["Ship Date"] = pd.to_datetime(orders["Ship Date"])

    # --- Merge Orders with People ---
    # Business logic: Region manager is an attribute of the order’s region
    df = orders.merge(people, on="Region", how="left")

    # --- Merge Returns ---
    # Adds a 'Returned' flag to each order if it exists in Returns table
    df = df.merge(returns, on="Order ID", how="left")

    # Fill missing values: if order not present in returns → Not Returned
    df["Returned"] = df["Returned"].fillna("No")

    return df


# Load the unified dataset
df = load_data()


# ============================================================
# TIME DIMENSIONS (Year & Quarter)
# ============================================================
# Creating derived columns for easy time slicing in dashboard
df["Year"] = df["Order Date"].dt.year
df["Quarter"] = df["Order Date"].dt.to_period("Q").astype(str)


# ============================================================
# SIDEBAR FILTERS (BUSINESS CONTROLS)
# ============================================================
st.sidebar.header("🔎 Filter Data")

# --- Year Selection ---
# This becomes the primary comparison axis (current vs previous year)
selected_year = st.sidebar.selectbox(
    "Select Year",
    sorted(df["Year"].unique()),
    index = 3
)

# Quarter intentionally kept optional (future extensibility)
selected_quarter = "All"

# Region filter (single select for cleaner executive dashboard UX)
selected_region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + sorted(df["Region"].unique())
)

# Category filter
selected_category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + sorted(df["Category"].unique())
)

# Segment filter
selected_segment = st.sidebar.selectbox(
    "Select Segment",
    ["All"] + sorted(df["Segment"].unique())
)


# ============================================================
# APPLY FILTER LOGIC (CURRENT PERIOD DATASET)
# ============================================================
# First filter by selected year (core time slicing)
filtered_df = df[df["Year"] == selected_year]

# Apply business dimension filters only if user chooses specific values
if selected_quarter != "All":
    filtered_df = filtered_df[filtered_df["Quarter"] == selected_quarter]

if selected_region != "All":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

if selected_segment != "All":
    filtered_df = filtered_df[filtered_df["Segment"] == selected_segment]


# ============================================================
# PREVIOUS YEAR DATA (FOR YoY COMPARISON)
# ============================================================
# We create an identical filtered dataset but for previous year
previous_df = df[df["Year"] == selected_year - 1]

# Apply same dimension filters to keep comparison apples-to-apples
if selected_quarter != "All":
    previous_df = previous_df[previous_df["Quarter"] == selected_quarter]

if selected_region != "All":
    previous_df = previous_df[previous_df["Region"] == selected_region]

if selected_category != "All":
    previous_df = previous_df[previous_df["Category"] == selected_category]

if selected_segment != "All":
    previous_df = previous_df[previous_df["Segment"] == selected_segment]


# ============================================================
# KPI SECTION (EXECUTIVE SUMMARY METRICS)
# ============================================================
st.subheader("📌 Key Metrics Overview")

# --- Current Year Metrics ---
total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Profit"].sum()
total_orders = filtered_df["Order ID"].nunique()
return_rate = (filtered_df["Returned"] == "Yes").mean()

# --- Previous Year Metrics (for growth comparison) ---
prev_sales = previous_df["Sales"].sum()
prev_profit = previous_df["Profit"].sum()
prev_orders = previous_df["Order ID"].nunique()
prev_return_rate = (previous_df["Returned"] == "Yes").mean()


# --- Growth Calculation Utility ---
# Handles divide-by-zero cases gracefully
def calc_growth(curr, prev):
    if prev == 0:
        return 0
    return (curr - prev) / prev


sales_growth = calc_growth(total_sales, prev_sales)
profit_growth = calc_growth(total_profit, prev_profit)
orders_growth = calc_growth(total_orders, prev_orders)
return_growth = calc_growth(return_rate, prev_return_rate)


# --- Display KPI Cards ---
# Each metric shows: value + YoY growth
col1, col2, col3, col4 = st.columns(4, border=True)

col1.metric("💰 Total Sales", f"${total_sales:,.0f}", f"{sales_growth:.2%}")
col2.metric("📈 Total Profit", f"${total_profit:,.0f}", f"{profit_growth:.2%}")
col3.metric("🧾 Total Orders", total_orders, f"{orders_growth:.2%}")
col4.metric("🔁 Return Rate", f"{return_rate:.2%}", f"{return_growth:.2%}")


# ============================================================
# MONTHLY TREND PREPARATION
# ============================================================
# Extract month number to build monthly trend charts
filtered_df["Month"] = filtered_df["Order Date"].dt.month
previous_df["Month"] = previous_df["Order Date"].dt.month

# Aggregate monthly sales
current_monthly = filtered_df.groupby("Month")["Sales"].sum().reset_index()
previous_monthly = previous_df.groupby("Month")["Sales"].sum().reset_index()


# ============================================================
# MONTHLY PROFIT RATIO CALCULATION
# ============================================================
# Profit Ratio = Profit / Sales (true profitability indicator)
curr_profit_monthly = filtered_df.groupby("Month")[["Profit", "Sales"]].sum().reset_index()
prev_profit_monthly = previous_df.groupby("Month")[["Profit", "Sales"]].sum().reset_index()

# Safe calculation (avoid divide by zero when no sales exist)
curr_profit_monthly["Profit_Ratio"] = curr_profit_monthly.apply(
    lambda x: x["Profit"] / x["Sales"] if x["Sales"] != 0 else 0, axis=1
)

prev_profit_monthly["Profit_Ratio"] = prev_profit_monthly.apply(
    lambda x: x["Profit"] / x["Sales"] if x["Sales"] != 0 else 0, axis=1
)

# Merge current vs previous for bullet comparison charts
profit_ratio_df = curr_profit_monthly.merge(
    prev_profit_monthly,
    on="Month",
    how="outer",
    suffixes=("_curr", "_prev")
).fillna(0)


# ============================================================
# VISUALIZATION SECTION (SALES + PROFIT RATIO YoY)
# ============================================================
col1, col2 = st.columns(2, border=True)


# ---------------- SALES BULLET CHART ----------------
with col1:
    st.markdown("### 💰 Monthly Sales (YoY)")

    sales_bullet_df = current_monthly.merge(
        previous_monthly,
        on="Month",
        how="outer",
        suffixes=("_curr", "_prev")
    ).fillna(0)

    base_sales = alt.Chart(sales_bullet_df).encode(
        x=alt.X("Month:O", title="Month")
    )

    # Background bar → Previous Year
    prev_sales_bar = base_sales.mark_bar(size=40, opacity=0.3, color="lightgray").encode(
        y="Sales_prev:Q"
    )

    # Foreground bar → Current Year
    curr_sales_bar = base_sales.mark_bar(size=20, color="steelblue").encode(
        y="Sales_curr:Q"
    )

    st.altair_chart(prev_sales_bar + curr_sales_bar, width='stretch')


# ---------------- PROFIT RATIO BULLET CHART ----------------
with col2:
    st.markdown("### 📈 Monthly Profit Ratio (YoY)")

    base_ratio = alt.Chart(profit_ratio_df).encode(
        x=alt.X("Month:O", title="Month")
    )

    # Previous year ratio (background reference)
    prev_ratio_bar = base_ratio.mark_bar(size=40, opacity=0.3, color="lightgray").encode(
        y=alt.Y("Profit_Ratio_prev:Q", title="Profit Ratio")
    )

    # Current year ratio (colored based on profitability)
    curr_ratio_bar = base_ratio.mark_bar(size=20).encode(
        y="Profit_Ratio_curr:Q",
        color=alt.condition(
            alt.datum.Profit_Ratio_curr < 0,
            alt.value("red"),        # Loss-making months → red
            alt.value("seagreen")    # Profitable months → green
        )
    )

    st.altair_chart(prev_ratio_bar + curr_ratio_bar, width='stretch')
